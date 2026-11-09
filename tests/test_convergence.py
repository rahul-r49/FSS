"""Temporal and spatial convergence of the solver."""
import numpy as np

from fss import SpectralGrid, ShallowWater, IFRK4, initial


def solve(n, dt, T=0.5, k_peak=3, seed=4):
    g = SpectralGrid(n)
    m = ShallowWater(g, f=1.0, eps=1.0)
    qk = g.to_spec(initial.random_vortical(g, k_peak=k_peak, u_rms=0.3, seed=seed))
    qk = IFRK4(m, dt).run(qk, int(round(T / dt)))
    return g, qk


def smooth_state(g):
    """Resolution-independent initial condition built from a few fixed modes."""
    psi = 0.3 * np.sin(g.x) * np.cos(2 * g.y) + 0.2 * np.cos(2 * g.x + g.y)
    psik = g.to_spec(psi)
    u = -g.to_phys(g.ddy(psik))
    v = g.to_phys(g.ddx(psik))
    h = 0.1 * np.cos(g.x - g.y)
    return np.stack([u, v, h])


def low_modes(qk, n, kmax=6):
    """Normalised low-wavenumber coefficients, comparable across resolutions."""
    kx = np.r_[0:kmax, -kmax:0]
    return qk[:, kx, :kmax] / n**2


def run_smooth(n, dt=0.005, T=0.5):
    g = SpectralGrid(n)
    m = ShallowWater(g, f=1.0, eps=1.0)
    qk = g.to_spec(smooth_state(g))
    return IFRK4(m, dt).run(qk, int(round(T / dt)))


def test_fourth_order_in_time():
    _, ref = solve(32, 0.5 / 320)
    errs = []
    for steps in (10, 20, 40):
        _, qk = solve(32, 0.5 / steps)
        errs.append(np.max(np.abs(qk - ref)))
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert min(orders) > 3.5, f"observed orders {orders}"


def test_spectral_accuracy_in_space():
    """Error vs an n=48 reference shrinks fast as resolution grows."""
    ref = low_modes(run_smooth(48), 48)
    errs = [np.max(np.abs(low_modes(run_smooth(n), n) - ref)) for n in (12, 16, 24)]
    assert errs[2] < errs[1] < errs[0], errs
    assert errs[2] < errs[0] / 100, errs        # roughly geometric decay with n
