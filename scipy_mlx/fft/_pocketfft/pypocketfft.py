"""
MLX-backed replacement for SciPy's `pypocketfft` extension.

Upstream SciPy uses pocketfft (C++) via a compiled Python extension for fast
FFTs and helper routines like `good_size`. In this MLX port, we route FFTs
through `mlx.core.fft` and provide conservative fallbacks for helper functions.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import mlx.core as mx


def good_size(n: int) -> int:
    # Pocketfft chooses "fast" lengths; we conservatively return `n`.
    return int(n)


def prev_good_size(n: int) -> int:
    return int(n)


def _norm_scale(norm: int, n: int):
    if norm == 0:
        return mx.array(1.0)
    if norm == 1:
        return mx.divide(mx.array(1.0), mx.sqrt(mx.array(float(n))))
    if norm == 2:
        return mx.divide(mx.array(1.0), mx.array(float(n)))
    raise ValueError(f"invalid norm mode {norm!r}")


def _prod_shape(x: mx.array, axes: Sequence[int]) -> int:
    p = 1
    for ax in axes:
        p *= int(x.shape[ax])
    return p


def c2c(x: mx.array, axes: Sequence[int], forward: bool, norm: int, out=None, workers: int = 1):
    axes = tuple(int(a) for a in axes)
    y = mx.fft.fftn(x, axes=axes) if forward else mx.fft.ifftn(x, axes=axes)
    scale = _norm_scale(norm, _prod_shape(x, axes))
    return mx.multiply(y, scale)


def r2c(x: mx.array, axes: Sequence[int], forward: bool, norm: int, out=None, workers: int = 1):
    axes = tuple(int(a) for a in axes)
    y = mx.fft.rfftn(x, axes=axes)
    if not forward:
        # ihfft/ihfftn convention: conj(rfft(x))
        y = mx.conjugate(y)
    scale = _norm_scale(norm, _prod_shape(x, axes))
    return mx.multiply(y, scale)


def c2r(
    x: mx.array,
    axes: Sequence[int],
    lastsize: int,
    forward: bool,
    norm: int,
    out=None,
    workers: int = 1,
):
    axes = tuple(int(a) for a in axes)
    # Build `s` with the full real-domain shape along transformed axes.
    s = [int(x.shape[a]) for a in axes]
    s[-1] = int(lastsize)
    xx = mx.conjugate(x) if forward else x
    y = mx.fft.irfftn(xx, s=tuple(s), axes=axes)
    scale = _norm_scale(norm, math.prod(s))
    return mx.multiply(y, scale)


def dct(*args, **kwargs):
    raise NotImplementedError("scipy_mlx.fft DCT is not yet implemented for MLX.")


def dst(*args, **kwargs):
    raise NotImplementedError("scipy_mlx.fft DST is not yet implemented for MLX.")
