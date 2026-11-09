"""Periodic spectral grid (serial): wavenumbers, 2/3 dealiasing mask, transforms.

Fields are stored in physical space as (n, n) arrays and in spectral space
as rfft2 coefficients of shape (n, n//2 + 1). Any leading axes (for example a
stacked state of shape (3, n, n)) are transformed in one batched call.

The solver only talks to a grid through the small interface below, so the
MPI grid in ``pgrid.py`` is a drop-in replacement:

    to_spec, to_phys, ddx, ddy        transforms and derivatives
    k2, mask, dk, k_cut, weight       spectral-space arrays/constants
    x, y                              physical coordinates (local part)
    mean, allsum, local, gather       reductions and data movement
    rank, size                        MPI rank and number of processes
"""
import numpy as np


class SpectralGrid:
    rank = 0
    size = 1

    def __init__(self, n, length=2 * np.pi):
        if n % 2:
            raise ValueError("n must be even")
        self.n = n
        self.length = length
        coords = np.arange(n) * length / n
        self.x, self.y = np.meshgrid(coords, coords, indexing="ij")

        scale = 2 * np.pi / length
        kx = scale * np.fft.fftfreq(n, 1.0 / n)
        ky = scale * np.fft.rfftfreq(n, 1.0 / n)
        self.kx, self.ky = np.meshgrid(kx, ky, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2
        self.dk = scale

        # 2/3 rule: keep |k| < n/3 in each direction (also removes Nyquist)
        self.k_cut = scale * n / 3.0
        self.mask = (np.abs(self.kx) < self.k_cut) & (np.abs(self.ky) < self.k_cut)

        # rfft2 stores half of the ky plane; interior columns count twice
        self.weight = np.full(self.ky.shape, 2.0)
        self.weight[:, 0] = 1.0
        self.weight[:, -1] = 1.0

    # transforms --------------------------------------------------------
    def to_spec(self, a):
        return np.fft.rfft2(a, axes=(-2, -1))

    def to_phys(self, ak):
        return np.fft.irfft2(ak, s=(self.n, self.n), axes=(-2, -1))

    # spectral derivatives ---------------------------------------------
    def ddx(self, ak):
        return 1j * self.kx * ak

    def ddy(self, ak):
        return 1j * self.ky * ak

    # reductions and data movement (trivial in serial) -------------------
    def mean(self, a):
        """Global mean over the last two (physical) axes."""
        return np.mean(a, axis=(-2, -1))

    def allsum(self, a):
        return a

    def local(self, a_global):
        """The part of a global physical-space array owned by this process."""
        return a_global

    def gather(self, a_local):
        """Full physical-space array (on every process in serial)."""
        return a_local
