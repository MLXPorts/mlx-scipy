"""
Minimal MLX-backed subset of `array_api_extra` used by SciPy.

Only implements the small surface used in this repository (see `rg xpx.`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx


def atleast_nd(a, *, ndim: int, xp=mx):
    arr = xp.asarray(a) if hasattr(xp, "asarray") else xp.array(a)
    while arr.ndim < ndim:
        arr = expand_dims(arr, axis=0, xp=xp)
    return arr


def expand_dims(a, axis: int = 0, *, xp=mx):
    arr = xp.asarray(a) if hasattr(xp, "asarray") else xp.array(a)
    axis = axis % (arr.ndim + 1)
    new_shape = list(arr.shape)
    new_shape.insert(axis, 1)
    return xp.reshape(arr, new_shape)


def isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False, *, xp=mx):
    a = xp.asarray(a) if hasattr(xp, "asarray") else xp.array(a)
    b = xp.asarray(b) if hasattr(xp, "asarray") else xp.array(b)
    diff = xp.abs(a - b)
    tol = xp.array(atol) + xp.array(rtol) * xp.abs(b)
    out = diff <= tol
    if equal_nan:
        out = xp.logical_or(out, xp.logical_and(xp.isnan(a), xp.isnan(b)))
    return out


def sinc(x, *, xp=mx):
    x = xp.asarray(x) if hasattr(xp, "asarray") else xp.array(x)
    pix = xp.multiply(xp.array(mx.pi), x)
    # sin(pi x) / (pi x), with sinc(0)=1
    return xp.where(x == 0, xp.ones_like(x), xp.sin(pix) / pix)


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *, xp=mx):
    x = xp.asarray(m) if hasattr(xp, "asarray") else xp.array(m)
    if y is not None:
        y = xp.asarray(y) if hasattr(xp, "asarray") else xp.array(y)
        x = xp.concatenate((x, y), axis=0 if rowvar else 1)
    if x.ndim == 1:
        x = xp.reshape(x, (1, -1))
    if not rowvar and x.ndim > 1:
        x = xp.transpose(x)
    if ddof is None:
        ddof = 0 if bias else 1
    avg = xp.mean(x, axis=1, keepdims=True)
    x = x - avg
    fact = x.shape[1] - ddof
    return xp.matmul(x, xp.transpose(x)) / xp.array(fact)


def nunique(a, *, xp=mx):
    a = xp.reshape(xp.asarray(a) if hasattr(xp, "asarray") else xp.array(a), (-1,))
    if a.size == 0:
        return 0
    # unique via sort+diff
    s = xp.sort(a)
    if s.size == 1:
        return 1
    mask = xp.concatenate([xp.array([True], dtype=mx.bool_), s[1:] != s[:-1]])
    return int(xp.sum(mask))


def pad(a, pad_width, mode="constant", constant_values=0, *, xp=mx):
    if mode != "constant":
        raise NotImplementedError("Only constant padding is supported in MLX array_api_extra stub.")
    arr = xp.asarray(a) if hasattr(xp, "asarray") else xp.array(a)
    if isinstance(pad_width, int):
        pad_width = [(pad_width, pad_width)] * arr.ndim
    elif isinstance(pad_width, (tuple, list)) and len(pad_width) == 2 and isinstance(pad_width[0], int):
        pad_width = [tuple(pad_width)] * arr.ndim
    pads = pad_width
    out_shape = [p0 + s + p1 for (p0, p1), s in zip(pads, arr.shape)]
    out = xp.full(out_shape, xp.array(constant_values, dtype=arr.dtype))
    slices = tuple(slice(p0, p0 + s) for (p0, _), s in zip(pads, arr.shape))
    out[slices] = arr
    return out


def create_diagonal(v, offset: int = 0, *, xp=mx):
    v = xp.reshape(xp.asarray(v) if hasattr(xp, "asarray") else xp.array(v), (-1,))
    n = int(v.shape[0] + abs(offset))
    out = xp.zeros((n, n), dtype=v.dtype)
    if offset >= 0:
        i = xp.arange(n - offset)
        out[i, i + offset] = v
    else:
        k = -offset
        i = xp.arange(n - k)
        out[i + k, i] = v
    return out


def apply_where(condition, args, f_true: Callable[..., Any], f_false: Callable[..., Any] | None = None,
                *, fill_value=None, xp=mx):
    cond = xp.asarray(condition, dtype=mx.bool_) if hasattr(xp, "asarray") else xp.array(condition, dtype=mx.bool_)
    if not isinstance(args, tuple):
        args = (args,)
    true_val = f_true(*args)
    if fill_value is not None:
        false_val = xp.array(fill_value) if xp.isscalar(fill_value) else fill_value
    elif f_false is not None:
        false_val = f_false(*args)
    else:
        false_val = xp.zeros_like(true_val)
    return xp.where(cond, true_val, false_val)


@dataclass(frozen=True)
class _AtHelper:
    x: Any
    idx: Any = None

    def __getitem__(self, idx):
        return _AtHelper(self.x, idx)

    def set(self, values, copy: bool | None = None):
        _ = copy
        y = mx.array(self.x)
        y[self.idx] = values
        return y

    def add(self, values, copy: bool | None = None):
        _ = copy
        y = mx.array(self.x)
        y[self.idx] = y[self.idx] + values
        return y

    def multiply(self, values, copy: bool | None = None):
        _ = copy
        y = mx.array(self.x)
        y[self.idx] = y[self.idx] * values
        return y


def at(x, idx=None):
    helper = _AtHelper(x)
    return helper[idx] if idx is not None else helper


def lazy_apply(func: Callable[..., Any], *args, validate: bool = False, xp=mx, **kwargs):
    _ = (validate, xp)
    return func(*args, **kwargs)


lazy_xp_backends = ()


from . import testing  # noqa: E402

