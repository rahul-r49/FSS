"""Right-hand side of the rotating shallow-water equations (spectral space).

State q = (u, v, h), stored as an array of shape (3, n, n) in physical space
and (3, n, n//2+1) in spectral space.

    u_t =  f v - h_x - eps (u u_x + v u_y)
    v_t = -f u - h_y - eps (u v_x + v v_y)
    h_t = -(u_x + v_y) - eps [(h u)_x + (h v)_y]

eps is the nonlinearity (Rossby-like) parameter; eps = 0 gives the linear
inertia-gravity wave system, which has exact solutions used in the tests.
"""
import numpy as np


class ShallowWater:
    def __init__(self, grid, f=1.0, eps=1.0):
        self.g = grid
        self.f = f
        self.eps = eps

    def tendency(self, qk):
        """d(qk)/dt from the full right-hand side, dealiased, spectral in/out."""
        g, f, eps = self.g, self.f, self.eps
        uk, vk, hk = qk
        u, v, h = g.to_phys(qk)

        # gradients of all three fields in one batched inverse transform
        grads = g.to_phys(np.stack([g.ddx(qk), g.ddy(qk)]))      # (2, 3, n, n)
        ux, vx, hx = grads[0]
        uy, vy, hy = grads[1]

        nl = eps * np.stack([
            u * ux + v * uy,                        # advective terms
            u * vx + v * vy,
            g.to_phys(g.ddx(g.to_spec(h * u)) + g.ddy(g.to_spec(h * v))),
        ])
        lin = np.stack([f * v - hx, -f * u - hy, -(ux + vy)])
        return g.mask * g.to_spec(lin - nl)

    def energy(self, q):
        """Conserved by the inviscid equations: 1/2 <(1 + eps h)|v|^2 + h^2>."""
        u, v, h = q
        return 0.5 * self.g.mean((1 + self.eps * h) * (u**2 + v**2) + h**2)

    def mass(self, q):
        return self.g.mean(q[2])
