"""Initial conditions (identical for the serial and MPI grids).

Random fields are generated globally from the seed on every process and each
process keeps its own slab, so the MPI run starts from exactly the same state
as the serial run regardless of the number of processes.
"""
import numpy as np


def random_vortical(grid, k_peak=6, u_rms=0.5, seed=0):
    """Random large-scale incompressible flow from a streamfunction, h = 0.

    Not in geostrophic balance, so it radiates inertia-gravity waves
    (a mixed waves + vortical state).
    """
    rng = np.random.default_rng(seed)
    n = grid.n
    white = grid.local(rng.standard_normal((n, n)))
    psik = grid.to_spec(white) * ((grid.k2 < (k_peak * grid.dk) ** 2) & (grid.k2 > 0))
    u = -grid.to_phys(grid.ddy(psik))
    v = grid.to_phys(grid.ddx(psik))
    scale = u_rms / np.sqrt(grid.mean(u**2 + v**2))
    return np.stack([u * scale, v * scale, np.zeros_like(u)])


def cosine_wave(grid, amplitude=0.1):
    """h = A cos(x), u = v = 0 (exact linear solution known)."""
    h = amplitude * np.cos(grid.x)
    return np.stack([np.zeros_like(h), np.zeros_like(h), h])
