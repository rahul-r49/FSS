"""Plot strong scaling from the CSV written by benchmark.py.

    python scripts/plot_scaling.py results/scaling_n1024.csv
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter


def main(path):
    with open(path) as fh:
        rows = sorted((int(r["processes"]), int(r["n"]), float(r["seconds_per_step"]))
                      for r in csv.DictReader(fh))
    procs = np.array([r[0] for r in rows])
    n = rows[0][1]
    t = np.array([r[2] for r in rows])
    speedup = t[0] / t
    efficiency = speedup * procs[0] / procs

    print(f"{'procs':>6} {'s/step':>10} {'speedup':>9} {'efficiency':>11}")
    for p, ti, s, e in zip(procs, t, speedup, efficiency):
        print(f"{p:6d} {ti:10.4f} {s:9.2f} {e:11.1%}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(procs, t, "o-", label="measured")
    ax[0].plot(procs, t[0] * procs[0] / procs, "r--", label="ideal (slope -1)")
    ax[0].set_xscale("log", base=2)
    ax[0].set_yscale("log")
    ax[0].set_xticks(procs)                                   # one tick per process count
    ax[0].xaxis.set_major_formatter(ScalarFormatter())
    ax[0].xaxis.set_minor_formatter(NullFormatter())
    ax[0].yaxis.set_major_formatter(ScalarFormatter())        # plain numbers, not 3x10^0
    ax[0].yaxis.set_minor_formatter(ScalarFormatter())
    ax[0].set_xlabel("MPI processes")
    ax[0].set_ylabel("time per step (s)")
    ax[0].set_title(f"Strong scaling, {n}$\\times${n} grid")
    ax[0].legend()

    ax[1].plot(procs, speedup, "o-", label="measured")
    ax[1].plot(procs, procs / procs[0], "r--", label="ideal")
    ax[1].set_xticks(procs)                                   # no fractional process counts
    ax[1].set_xlabel("MPI processes")
    ax[1].set_ylabel(f"speedup relative to {procs[0]} process" + ("" if procs[0] == 1 else "es"))
    ax[1].set_title("Speedup")
    ax[1].legend(loc="upper left")

    ax[2].plot(procs, 100 * efficiency, "o-", label="measured")
    ax[2].axhline(100, color="r", ls="--", label="ideal (100%)")
    ax[2].set_xticks(procs)
    ax[2].set_ylim(0, 110)
    ax[2].set_xlabel("MPI processes")
    ax[2].set_ylabel("parallel efficiency (%)")
    ax[2].set_title("Parallel efficiency")
    ax[2].legend(loc="lower left")
    plt.tight_layout()

    out = os.path.splitext(path)[0] + ".png"
    plt.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1])