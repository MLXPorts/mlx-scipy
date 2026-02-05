"""
Stub for SciPy's compiled Fortran BLAS wrappers (`scipy.linalg._fblas`).

This module is normally a compiled extension. In this MLX port we provide an
importable placeholder so higher-level modules can be imported.
"""

from __future__ import annotations


class error(Exception):
    pass


__all__ = ["error"]

