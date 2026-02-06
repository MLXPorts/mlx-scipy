"""
Very small placeholder for `numpy.ma` used in a handful of legacy tests.

Masked arrays are not implemented for MLX; this module exists to keep imports
working without external NumPy.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

masked = object()
nomask = None


class MaskedArray:  # pragma: no cover
    pass


def array(data: Any, dtype=None, **kwargs):
    if kwargs.get("mask", None) is not None:
        raise NotImplementedError("Masked arrays are not supported in the MLX port.")
    return mx.array(data, dtype=dtype)


masked_array = array


__all__ = [
    "MaskedArray",
    "array",
    "masked",
    "masked_array",
    "nomask",
]

