"""
MLX stub for SciPy's SuperLU sparse direct solver extension.

Upstream SciPy provides SuperLU as a compiled extension. This MLX port does not
ship that extension; we provide a placeholder `SuperLU` class to keep imports
working.
"""

from __future__ import annotations


class SuperLU:  # pragma: no cover
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("scipy_mlx.sparse.linalg.SuperLU is not yet implemented for MLX.")

