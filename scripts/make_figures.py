"""Make the README figures from saved runs, plus a temporal convergence plot.

    python scripts/make_figures.py results/f1 results/f3
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fss import SpectralGrid, ShallowWater, IFRK4, initial


def fields_and_spectra(dirs):
    # file name built from the f and n of each run, so new runs get new names
    tag = "_".join(f"f{float(np.load(f'{d}/final.npz')['f']):g}_n{int(np.load(f'{d}/final.npz')['n'])}"
                   for d in dirs)
    out = f"results/fields_{tag}.png"
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(1, len(dirs) + 1, figsize=(5 * (len(dirs) + 1), 4.4))
    for j, d in enumerate(dirs):
        z = np.load(f"{d}/final.npz")
        im = ax[j].imshow(z["neg_div"].T, origin="lower", cmap="inferno",
                          extent=[0, 2 * np.pi, 0, 2 * np.pi])
        label = f"f = {float(z['f']):g}, N = {int(z['n'])}"
        ax[j].set_title(f"$-\\nabla\\cdot v$,  {label}")
        plt.colorbar(im, ax=ax[j], shrink=0.85)
        ax[-1].loglog(z["k"][1:], z["E"][1:], label=label)
        if j == 0:                                  # anchor the guide to the first run
            k_ref, E_ref = z["k"], z["E"]
    k0 = min(8, len(k_ref) - 1)
    kk = np.array([k0, 0.4 * k_ref[-1]])
    ax[-1].loglog(kk, 0.1 * E_ref[k0] * (kk / k0) ** -2,
                  "k--", lw=1, label="$k^{-2}$ guide")
    ax[-1].set_xlabel("k")
    ax[-1].set_ylabel("E(k)")
    ax[-1].legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


def temporal_convergence(out="results/temporal_convergence.png"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n, T = 32, 0.5
    g = SpectralGrid(n)
    m = ShallowWater(g, f=1.0, eps=1.0)
    q0 = g.to_spec(initial.random_vortical(g, k_peak=3, u_rms=0.3, seed=4))
    ref = IFRK4(m, T / 640).run(q0, 640)
    steps = np.array([5, 10, 20, 40, 80])
    err = [np.max(np.abs(IFRK4(m, T / s).run(q0, s) - ref)) for s in steps]
    dts = T / steps
    plt.figure(figsize=(5, 4))
    plt.loglog(dts, err, "o-", label="IFRK4")
    plt.loglog(dts, err[0] * (dts / dts[0]) ** 4, "k--", lw=1, label="slope 4")
    plt.xlabel("dt")
    plt.ylabel("max error vs fine reference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print("saved", out, "orders:", np.round(np.log2(np.array(err[:-1]) / np.array(err[1:])), 2))


if __name__ == "__main__":
    temporal_convergence()
    if len(sys.argv) > 1:
        fields_and_spectra(sys.argv[1:])