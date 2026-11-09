from .grid import SpectralGrid
from .model import ShallowWater
from .timestepping import IFRK4
from . import diagnostics, initial


def make_grid(n, mpi=False, **kw):
    """Serial SpectralGrid, or the MPI ParallelGrid when ``mpi=True``."""
    if mpi:
        from .pgrid import ParallelGrid        # imports mpi4py only when needed
        return ParallelGrid(n, **kw)
    return SpectralGrid(n, **kw)


__all__ = ["SpectralGrid", "ShallowWater", "IFRK4", "diagnostics", "initial", "make_grid"]
__version__ = "0.2.0"
