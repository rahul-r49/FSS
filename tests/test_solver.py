import numpy as np
import pytest

from fss import SpectralGrid, ShallowWater, IFRK4, initial


def make(n=32, f=1.0, eps=1.0, dt=0.01, nu=0.0):
    g = SpectralGrid(n)
    m = ShallowWater(g, f=f, eps=eps)
    return g, m, IFRK4(m, dt, nu=nu)


def test_transform_roundtrip():
    g = SpectralGrid(32)
    a = np.random.default_rng(1).standard_normal((3, 32, 32))
    assert np.allclose(g.to_phys(g.to_spec(a)), a, atol=1e-12)


def test_derivative_of_sine():
    g = SpectralGrid(32)
    a = np.sin(3 * g.x)
    da = g.to_phys(g.ddx(g.to_spec(a)))
    assert np.allclose(da, 3 * np.cos(3 * g.x), atol=1e-10)


def test_dealias_mask_removes_upper_third():
    g = SpectralGrid(48)
    assert g.mask[0, 0]
    assert not g.mask[24, 0]            # Nyquist
    assert not g.mask[0, 20]            # beyond n/3 = 16


def test_linear_wave_matches_exact_solution():
    """eps=0, h=A cos x: h(0,t) = A (f^2 + cos(w t)) / w^2, w^2 = f^2 + 1."""
    f, A, T, dt = 1.0, 0.1, 5.0, 0.01
    g, m, stepper = make(n=32, f=f, eps=0.0, dt=dt)
    qk = g.to_spec(initial.cosine_wave(g, A))
    qk = stepper.run(qk, int(round(T / dt)))
    h = g.to_phys(qk)[2]
    w = np.sqrt(f**2 + 1)
    exact = A * (f**2 + np.cos(w * T)) / w**2
    assert abs(h[0, 0] - exact) < 1e-8


def test_energy_and_mass_conserved_inviscid():
    g, m, stepper = make(n=32, dt=0.01)
    q0 = initial.random_vortical(g, k_peak=3, u_rms=0.3, seed=2)
    qk = stepper.run(g.to_spec(q0), 100)
    q1 = g.to_phys(qk)
    assert abs(m.energy(q1) - m.energy(q0)) / m.energy(q0) < 1e-4
    assert abs(m.mass(q1) - m.mass(q0)) < 1e-12


def test_viscosity_only_removes_energy():
    g, m, stepper = make(n=32, dt=0.01, nu=1e-6)
    q0 = initial.random_vortical(g, k_peak=3, u_rms=0.3, seed=3)
    qk = g.to_spec(q0)
    E_prev = m.energy(q0)
    for _ in range(5):
        qk = stepper.run(qk, 20)
        E = m.energy(g.to_phys(qk))
        assert E < E_prev
        E_prev = E


def test_same_seed_is_reproducible():
    g = SpectralGrid(32)
    a = initial.random_vortical(g, seed=7)
    b = initial.random_vortical(g, seed=7)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("n", [16, 32])
def test_run_is_finite(n):
    g, m, stepper = make(n=n, dt=0.01, nu=1e-8)
    qk = g.to_spec(initial.random_vortical(g, k_peak=3, u_rms=0.3))
    qk = stepper.run(qk, 50)
    assert np.all(np.isfinite(qk))
