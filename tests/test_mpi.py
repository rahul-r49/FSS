"""Tests of the MPI grid against analytic results (run with plain `pytest`)."""
import numpy as np

from fss import make_grid, ShallowWater, IFRK4, initial


def test_parallel_grid_roundtrip():
    g = make_grid(32, mpi=True)
    a = np.random.default_rng(1).standard_normal((3, 32, 32))
    assert np.allclose(g.to_phys(g.to_spec(a)), a, atol=1e-12)


def test_parallel_linear_wave_matches_exact_solution():
    """eps = 0, h = A cos x: h(0, t) = A (f^2 + cos(w t)) / w^2, w^2 = f^2 + 1."""
    f, A, T, dt = 1.0, 0.1, 5.0, 0.01
    g = make_grid(32, mpi=True)
    m = ShallowWater(g, f=f, eps=0.0)
    qk = IFRK4(m, dt).run(g.to_spec(initial.cosine_wave(g, A)), int(round(T / dt)))
    h = g.gather(g.to_phys(qk))[2]
    w = np.sqrt(f**2 + 1)
    assert abs(h[0, 0] - A * (f**2 + np.cos(w * T)) / w**2) < 1e-8
