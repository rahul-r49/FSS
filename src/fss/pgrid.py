"""MPI slab-decomposed spectral grid.

Data layout (p = number of processes, nl = n / p):

    physical space   (..., nl, n)   process r owns x-rows  r*nl ... (r+1)*nl - 1
    spectral space   (..., n, nl)   process r owns ky-cols r*nl ... (r+1)*nl - 1
                                    (all kx; ky ordered as numpy.fft.fftfreq)

Forward transform (physical -> spectral):
    1. FFT along y        (y is complete on every process)
    2. cut the ky axis into p blocks and exchange them with Alltoall
       (afterwards x is complete and ky is split)
    3. FFT along x

The inverse runs the same steps backwards. Only these two transposes need
communication; every other operation in the solver is local.

This grid uses complex FFTs (fft along each axis) rather than rfft2, because
the half-plane of rfft2 (n/2 + 1 columns) does not divide evenly among
processes. n must be divisible by the number of processes.
"""
import numpy as np
from mpi4py import MPI


class ParallelGrid:
    def __init__(self, n, length=2 * np.pi, comm=None):
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        if n % 2:
            raise ValueError("n must be even")
        if n % self.size:
            raise ValueError(f"n={n} must be divisible by the number of processes ({self.size})")
        self.n = n
        self.length = length
        self.nl = nl = n // self.size
        self.rows = slice(self.rank * nl, (self.rank + 1) * nl)

        coords = np.arange(n) * length / n
        self.x, self.y = np.meshgrid(coords[self.rows], coords, indexing="ij")

        scale = 2 * np.pi / length
        k = scale * np.fft.fftfreq(n, 1.0 / n)
        self.kx, self.ky = np.meshgrid(k, k[self.rows], indexing="ij")     # (n, nl)
        self.k2 = self.kx**2 + self.ky**2
        self.dk = scale

        self.k_cut = scale * n / 3.0
        self.mask = (np.abs(self.kx) < self.k_cut) & (np.abs(self.ky) < self.k_cut)
        self.weight = np.ones(self.ky.shape)          # full complex plane

    # transforms --------------------------------------------------------
    def to_spec(self, a):
        n, nl, p = self.n, self.nl, self.size
        lead = a.shape[:-2]
        ak = np.fft.fft(a, axis=-1)                                    # (..., nl, n)
        send = np.ascontiguousarray(
            np.moveaxis(ak.reshape(*lead, nl, p, nl), -2, 0))         # (p, ..., nl_x, nl_ky)
        recv = np.empty_like(send)
        self.comm.Alltoall(send, recv)
        ak = np.moveaxis(recv, 0, -3).reshape(*lead, n, nl)           # (..., n, nl_ky)
        return np.fft.fft(ak, axis=-2)

    def to_phys(self, ak):
        n, nl, p = self.n, self.nl, self.size
        lead = ak.shape[:-2]
        a = np.fft.ifft(ak, axis=-2)                                   # (..., n, nl_ky)
        send = np.ascontiguousarray(
            np.moveaxis(a.reshape(*lead, p, nl, nl), -3, 0))          # (p, ..., nl_x, nl_ky)
        recv = np.empty_like(send)
        self.comm.Alltoall(send, recv)
        a = np.moveaxis(recv, 0, -2).reshape(*lead, nl, n)            # (..., nl_x, n)
        return np.fft.ifft(a, axis=-1).real

    # spectral derivatives ---------------------------------------------
    def ddx(self, ak):
        return 1j * self.kx * ak

    def ddy(self, ak):
        return 1j * self.ky * ak

    # reductions and data movement --------------------------------------
    def mean(self, a):
        s = np.sum(a, axis=(-2, -1))
        return self.comm.allreduce(s, op=MPI.SUM) / self.n**2

    def allsum(self, a):
        out = np.empty_like(a)
        self.comm.Allreduce(np.ascontiguousarray(a), out, op=MPI.SUM)
        return out

    def local(self, a_global):
        return a_global[..., self.rows, :]

    def gather(self, a_local):
        """Full physical-space array on rank 0 (None on other ranks)."""
        parts = self.comm.gather(np.ascontiguousarray(a_local), root=0)
        if self.rank == 0:
            return np.concatenate(parts, axis=-2)
        return None
