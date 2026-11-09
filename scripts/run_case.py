"""Run one decaying wave+vortical case and save fields and spectrum.

    mpiexec -n 4 python scripts/run_case.py --f 1 --n 256 --T 5 --out data/f1

n must be divisible by the number of processes.
"""
import argparse
import os
import time

import numpy as np

from fss import make_grid, ShallowWater, IFRK4, initial, diagnostics


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--f", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--T", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--damp", type=float, default=50.0,
                   help="hyperviscous damping rate at the dealiasing cutoff")
    p.add_argument("--r", type=int, default=8, help="hyperviscosity order")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/run")
    a = p.parse_args()

    g = make_grid(a.n, mpi=True)
    root = g.rank == 0
    if root:
        os.makedirs(a.out, exist_ok=True)
        print(f"grid {a.n}x{a.n}, processes: {g.size}", flush=True)
    m = ShallowWater(g, f=a.f, eps=a.eps)
    nu = a.damp / g.k_cut ** (2 * a.r)
    stepper = IFRK4(m, a.dt, nu=nu, r=a.r)

    q0 = initial.random_vortical(g, seed=a.seed)
    qk = g.to_spec(q0)
    E0 = m.energy(q0)
    t0 = time.time()

    def log(i, qk):
        e = m.energy(g.to_phys(qk)) / E0          # collective: all ranks call it
        if root:
            print(f"t={i * a.dt:6.2f}  E/E0={e:.5f}  wall={time.time() - t0:5.1f}s", flush=True)

    n_steps = int(round(a.T / a.dt))
    qk = stepper.run(qk, n_steps, callback=log, every=max(1, n_steps // 10))

    k, E = diagnostics.energy_spectrum(g, qk)
    q = g.gather(g.to_phys(qk))
    neg_div = g.gather(-diagnostics.divergence(g, qk))
    if root:
        np.savez_compressed(os.path.join(a.out, "final.npz"), q=q, neg_div=neg_div,
                            k=k, E=E, f=a.f, eps=a.eps, n=a.n)
        print("saved", os.path.join(a.out, "final.npz"))


if __name__ == "__main__":
    main()
