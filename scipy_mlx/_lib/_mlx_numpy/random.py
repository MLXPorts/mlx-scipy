"""
Minimal replacement for `numpy.random` for the MLX port.

This is not API-compatible with NumPy; it exists to keep legacy modules
importable without external NumPy.
"""

from __future__ import annotations

from typing import Any, Iterable

import mlx.core as mx


def _shape_from_size(size) -> tuple[int, ...]:
    if size is None:
        return ()
    if isinstance(size, int):
        return (int(size),)
    if isinstance(size, tuple):
        return tuple(int(s) for s in size)
    if isinstance(size, list):
        return tuple(int(s) for s in size)
    # Fallback for array-likes
    return tuple(int(s) for s in size)


def rand(*shape: int) -> mx.array:
    return mx.random.uniform(shape=tuple(int(s) for s in shape))


def random(size=None) -> mx.array:
    return mx.random.uniform(shape=_shape_from_size(size))


def uniform(low: Any = 0.0, high: Any = 1.0, size=None) -> mx.array:
    return mx.random.uniform(low, high, _shape_from_size(size))


class _Generator:
    def __init__(self, seed=None):
        if seed is not None:
            mx.random.seed(int(seed))

    def random(self, size=None) -> mx.array:
        return random(size=size)

    def uniform(self, low: Any = 0.0, high: Any = 1.0, size=None) -> mx.array:
        return uniform(low=low, high=high, size=size)

    def standard_normal(self, size=None) -> mx.array:
        return mx.random.normal(shape=_shape_from_size(size))


def default_rng(seed=None) -> _Generator:
    return _Generator(seed)


RandomState = _Generator


__all__ = [
    "RandomState",
    "default_rng",
    "rand",
    "random",
    "uniform",
]

