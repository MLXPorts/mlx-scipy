"""
MLX stub for SciPy's compiled ODRPACK wrapper (`odr.__odrpack`).

Upstream SciPy provides a compiled wrapper around the Fortran ODRPACK library.
This MLX port does not ship the compiled extension; provide a stub so that the
high-level Python API can be imported.
"""

from __future__ import annotations


def odr(*args, **kwargs):
    raise NotImplementedError("scipy_mlx.odr is not yet implemented for MLX (missing ODRPACK backend).")


def _set_exceptions(*args, **kwargs):
    # Called by the pure-Python wrapper to register exception types.
    return None
