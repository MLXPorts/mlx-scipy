"""
Stub for SciPy's compiled Qhull wrappers (`scipy.spatial._qhull`).
"""

from __future__ import annotations

from typing import Any


class QhullError(RuntimeError):
    pass


def _not_impl(*args: Any, **kwargs: Any):
    raise NotImplementedError("scipy_mlx.spatial._qhull is not implemented for MLX.")


# Common entrypoints referenced by spatial submodules
Delaunay = _not_impl
ConvexHull = _not_impl
Voronoi = _not_impl
HalfspaceIntersection = _not_impl

__all__ = [
    "QhullError",
    "Delaunay",
    "ConvexHull",
    "Voronoi",
    "HalfspaceIntersection",
]

