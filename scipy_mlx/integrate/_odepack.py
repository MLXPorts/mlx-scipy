"""
MLX stub for SciPy's compiled ODEPACK wrapper (`integrate._odepack`).

Upstream SciPy wraps the Fortran ODEPACK solvers via a compiled extension.
This MLX port does not ship that extension; we provide a stub so imports work.
"""

from __future__ import annotations


def odeint(*args, **kwargs):
    raise NotImplementedError("scipy_mlx.integrate.odeint is not yet implemented for MLX (missing ODEPACK backend).")

