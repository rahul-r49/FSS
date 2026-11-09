"""Diagnostics: shell-summed spectra and divergence (serial or MPI grid)."""
import numpy as np


def energy_spectrum(grid, qk, fields=(0, 1, 2)):
    """Shell-summed spectrum E(k) of the chosen fields (default u, v, h).

    Shells have unit width in units of the fundamental wavenumber. The grid
    supplies ``weight`` (how many times each stored mode counts, which differs
    between the half-plane rfft2 layout and the full complex plane) and
    ``allsum`` (a sum across processes, trivial in serial).
    """
    n = grid.n
    shell = np.rint(np.sqrt(grid.k2) / grid.dk).astype(int).ravel()
    nshell = n // 2
    spec = np.zeros(nshell)
    for i in fields:
        e = 0.5 * grid.weight * np.abs(qk[i] / n**2) ** 2
        spec += np.bincount(shell, weights=e.ravel(), minlength=nshell + 1)[:nshell]
    return np.arange(nshell) * grid.dk, grid.allsum(spec)


def divergence(grid, qk):
    return grid.to_phys(grid.ddx(qk[0]) + grid.ddy(qk[1]))
