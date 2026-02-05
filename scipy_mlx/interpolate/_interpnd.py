"""
MLX stub for SciPy's compiled ND interpolation backend (`interpolate._interpnd`).

Upstream SciPy provides compiled implementations for several interpolators and
support utilities. This MLX port keeps the high-level API importable; most of
the heavy lifting is not yet implemented.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def _ndim_coords_from_arrays(points, ndim: int | None = None):
    """
    Convert a sequence of coordinate arrays into an (npoints, ndim) array.
    """
    if isinstance(points, (list, tuple)):
        arrs = [mx.reshape(mx.array(p), (-1,)) for p in points]
        return mx.stack(arrs, axis=-1)
    return mx.array(points)


class NDInterpolatorBase:  # pragma: no cover
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("NDInterpolatorBase is not yet implemented for MLX.")


class LinearNDInterpolator(NDInterpolatorBase):  # pragma: no cover
    pass


class NearestNDInterpolator(NDInterpolatorBase):  # pragma: no cover
    pass


class CloughTocher2DInterpolator(NDInterpolatorBase):  # pragma: no cover
    pass


def __getattr__(name: str) -> Any:  # pragma: no cover
    raise AttributeError(name)

