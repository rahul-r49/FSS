"""Time the solver for a fixed problem size (for strong-scaling plots).

    mpiexec -n 4 python scripts/benchmark.py --n 1024 --steps 20 --csv results/scaling.csv

Appends one row (processes, n, seconds per time step) to the CSV. The reported
time is the slowest process, which is what determines the wall-clock time.
"""
import argparse
import csv
import os
import time

from mpi4py import MPI

from fss import make_grid, ShallowWater, IFRK4, initial


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3, help="keep the best of this many timings")
    p.add_argument("--csv", default="results/scaling.csv")
    a = p.parse_args()

    g = make_grid(a.n, mpi=True)
    m = ShallowWater(g, f=1.0, eps=1.0)
    stepper = IFRK4(m, dt=0.002, nu=1e-30, r=8)
    qk = g.to_spec(initial.random_vortical(g, seed=0))

    qk = stepper.run(qk, a.warmup)                      # warm-up (FFT plans, caches)
    best = float("inf")
    for _ in range(a.repeats):
        g.comm.Barrier()
        t0 = time.perf_counter()
        qk = stepper.run(qk, a.steps)
        g.comm.Barrier()
        best = min(best, (time.perf_counter() - t0) / a.steps)
    worst = g.comm.allreduce(best, op=MPI.MAX)

    if g.rank == 0:
        print(f"n={a.n}  processes={g.size}  time/step = {worst:.4f} s")
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        new = not os.path.exists(a.csv)
        with open(a.csv, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["processes", "n", "seconds_per_step"])
            w.writerow([g.size, a.n, f"{worst:.6f}"])


if __name__ == "__main__":
    main()
