"""
Stub for SciPy's compiled `cKDTree` implementation.

This MLX port currently does not include a full KD-tree. This module exists to
keep imports working; the implementation raises `NotImplementedError`.
"""

from __future__ import annotations

from typing import Any


class cKDTree:  # noqa: N801 (match SciPy API)
    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("scipy_mlx.spatial.cKDTree is not implemented for MLX.")


class cKDTreeNode:  # noqa: N801
    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("scipy_mlx.spatial.cKDTreeNode is not implemented for MLX.")
