"""Integrating-factor RK4 (IFRK4) with hyperviscosity treated exactly.

The damping term -nu (-nabla^2)^r q is diagonal in Fourier space, so
exp(-nu k^(2r) dt) can be applied exactly and RK4 only sees the remaining
(non-stiff) terms. Unlike RK4 followed by a backward-Euler damping step, this
keeps fourth-order accuracy in time (verified in tests/test_convergence.py).
"""
import numpy as np


class IFRK4:
    def __init__(self, model, dt, nu=0.0, r=8):
        self.model = model
        self.dt = dt
        k2 = model.g.k2
        damp = nu * k2**r
        self.E = np.exp(-damp * dt)
        self.E2 = np.exp(-damp * dt / 2)

    def step(self, qk):
        N, dt, E, E2 = self.model.tendency, self.dt, self.E, self.E2
        a = N(qk)
        b = N(E2 * (qk + 0.5 * dt * a))
        c = N(E2 * qk + 0.5 * dt * b)
        d = N(E * qk + dt * E2 * c)
        return E * qk + dt / 6 * (E * a + 2 * E2 * (b + c) + d)

    def run(self, qk, n_steps, callback=None, every=1):
        for i in range(1, n_steps + 1):
            qk = self.step(qk)
            if callback is not None and i % every == 0:
                callback(i, qk)
        return qk
