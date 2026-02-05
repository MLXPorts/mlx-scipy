"""
MLX compatibility layer for NumPy-like APIs used across scipy_mlx.

This module provides small, MLX-backed shims for NumPy functions that are
referenced throughout the codebase. It also installs a few helpers on
``mlx.core`` to preserve existing call sites (e.g., ``mx.r_`` and
``mx.poly*`` utilities).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence
import math
import sys
from array import array as _py_array
import types as _types

import mlx.core as mx

_NATIVE_BYTEORDER = "<" if sys.byteorder == "little" else ">"
_ORIG_MX_FINFO = getattr(mx, "finfo", None)
_ORIG_MX_IINFO = getattr(mx, "iinfo", None)
_ORIG_MX_VECTORIZe = getattr(mx, "vectorize", None)

# MLX dtype surface area varies by version/platform. SciPy expects these names.
if not hasattr(mx, "complex128"):
    mx.complex128 = mx.complex64  # type: ignore[attr-defined]
if not hasattr(mx, "byte"):
    mx.byte = mx.int8  # type: ignore[attr-defined]
if not hasattr(mx, "ubyte"):
    mx.ubyte = mx.uint8  # type: ignore[attr-defined]
if not hasattr(mx, "short"):
    mx.short = mx.int16  # type: ignore[attr-defined]
if not hasattr(mx, "ushort"):
    mx.ushort = mx.uint16  # type: ignore[attr-defined]
if not hasattr(mx, "intc"):
    mx.intc = mx.int32  # type: ignore[attr-defined]
if not hasattr(mx, "uintc"):
    mx.uintc = mx.uint32  # type: ignore[attr-defined]
if not hasattr(mx, "intp"):
    mx.intp = mx.int64  # type: ignore[attr-defined]
if not hasattr(mx, "longlong"):
    mx.longlong = mx.int64  # type: ignore[attr-defined]
if not hasattr(mx, "ulonglong") and hasattr(mx, "uint64"):
    mx.ulonglong = mx.uint64  # type: ignore[attr-defined]
if not hasattr(mx, "longdouble"):
    # MLX typically doesn't have extended precision; map to float64.
    mx.longdouble = mx.float64  # type: ignore[attr-defined]
if not hasattr(mx, "clongdouble"):
    mx.clongdouble = mx.complex128  # type: ignore[attr-defined]
if not hasattr(mx, "rint") and hasattr(mx, "round"):
    mx.rint = mx.round  # type: ignore[attr-defined]
if not hasattr(mx, "deg2rad"):
    mx.deg2rad = lambda x: mx.multiply(mx.array(x), mx.array(math.pi / 180.0))  # type: ignore[attr-defined]
if not hasattr(mx, "rad2deg"):
    mx.rad2deg = lambda x: mx.multiply(mx.array(x), mx.array(180.0 / math.pi))  # type: ignore[attr-defined]
if not hasattr(mx, "trunc") and hasattr(mx, "floor") and hasattr(mx, "ceil"):
    mx.trunc = lambda x: mx.where(mx.greater_equal(mx.array(x), mx.array(0.0)), mx.floor(x), mx.ceil(x))  # type: ignore[attr-defined]
if not hasattr(mx, "absolute") and hasattr(mx, "abs"):
    mx.absolute = mx.abs  # type: ignore[attr-defined]


class MxDType:
    """Minimal dtype wrapper with numpy-like attributes used in scipy_mlx."""

    def __init__(self, char: str, itemsize: int, mx_dtype, byteorder: str = "="):
        self.char = char
        self.itemsize = int(itemsize)
        self.byteorder = byteorder
        self.mx_dtype = mx_dtype
        # Minimal numpy-like scalar constructor used in a few call sites.
        if char in ("b", "h", "i", "l", "q", "B", "H", "I", "Q", "?"):
            self.type = int if char != "?" else bool
        elif char in ("f", "d", "e"):
            self.type = float
        elif char in ("F", "D"):
            self.type = complex
        else:
            self.type = object

    def newbyteorder(self, order: str):
        if order in ("B", ">"):
            byteorder = ">"
        elif order in ("L", "<"):
            byteorder = "<"
        elif order in ("=", "|"):
            byteorder = "="
        else:
            byteorder = order
        return MxDType(self.char, self.itemsize, self.mx_dtype, byteorder=byteorder)

    def __repr__(self) -> str:
        return f"MxDType(char={self.char!r}, itemsize={self.itemsize}, byteorder={self.byteorder!r})"

    @property
    def kind(self) -> str:
        # Match NumPy's `dtype.kind` semantics for common numeric types.
        if self.char == "?":
            return "b"
        if self.char in ("b", "h", "i", "l", "q"):
            return "i"
        if self.char in ("B", "H", "I", "Q"):
            return "u"
        if self.char in ("e", "f", "d"):
            return "f"
        if self.char in ("F", "D"):
            return "c"
        return "V"


_CHAR_TO_MX = {
    "?": mx.bool_,
    "b": mx.int8,
    "B": mx.uint8,
    "h": mx.int16,
    "H": mx.uint16,
    "i": mx.int32,
    "I": mx.uint32,
    "l": mx.int64,
    "q": mx.int64,
    "Q": getattr(mx, "uint64", None),
    "f": mx.float32,
    "d": mx.float64,
    "F": mx.complex64,
    "D": mx.complex128,
    "c": mx.uint8,
    "S": mx.uint8,
}

_STR_TO_CHAR = {
    "bool": ("?", 1),
    "bool_": ("?", 1),
    "int8": ("b", 1),
    "uint8": ("B", 1),
    "int16": ("h", 2),
    "uint16": ("H", 2),
    "int32": ("i", 4),
    "uint32": ("I", 4),
    "int64": ("q", 8),
    "uint64": ("Q", 8),
    "float16": ("e", 2),
    "float32": ("f", 4),
    "float64": ("d", 8),
    "complex64": ("F", 8),
    "complex128": ("D", 16),
    "f2": ("e", 2),
    "f4": ("f", 4),
    "f8": ("d", 8),
    "i1": ("b", 1),
    "u1": ("B", 1),
    "i2": ("h", 2),
    "u2": ("H", 2),
    "i4": ("i", 4),
    "u4": ("I", 4),
    "i8": ("q", 8),
    "u8": ("Q", 8),
    "c8": ("F", 8),
    "c16": ("D", 16),
}


def _as_mx_dtype(dt):
    if dt is None:
        return None
    if isinstance(dt, MxDType):
        return dt.mx_dtype
    if dt in _CHAR_TO_MX:
        return _CHAR_TO_MX[dt]
    try:
        return mx.Dtype(dt)
    except Exception:
        pass
    try:
        out = mx.dtype(dt)
        return out.mx_dtype if isinstance(out, MxDType) else out
    except Exception:
        pass
    return dtype(dt).mx_dtype


def _parse_dtype_string(spec: str):
    if not spec:
        raise TypeError("empty dtype spec")
    byteorder = "="
    if spec[0] in "<>=|":
        byteorder, spec = spec[0], spec[1:]
    spec = spec.strip()
    if spec in _STR_TO_CHAR:
        char, itemsize = _STR_TO_CHAR[spec]
        return char, itemsize, byteorder
    if len(spec) == 1:
        # single char code
        char = spec
        itemsize = {
            "?": 1, "b": 1, "B": 1, "c": 1, "S": 1,
            "h": 2, "H": 2,
            "i": 4, "I": 4,
            "l": 8, "q": 8, "Q": 8,
            "f": 4, "d": 8,
            "F": 8, "D": 16,
        }.get(char)
        if itemsize is None:
            raise TypeError(f"unsupported dtype spec {spec!r}")
        return char, itemsize, byteorder
    # like 'i4', 'f8', 'u2'
    char = spec[0]
    size = int(spec[1:])
    return char, size, byteorder


# ---------------------------
# Basic array helpers
# ---------------------------

def array(obj, dtype=None):
    return mx.array(obj, dtype=_as_mx_dtype(dtype))


def asarray(obj, dtype=None):
    return mx.array(obj, dtype=_as_mx_dtype(dtype))


def asanyarray(obj, dtype=None):
    return mx.array(obj, dtype=_as_mx_dtype(dtype))

def ascontiguousarray(obj, dtype=None):
    # MLX doesn't expose explicit contiguity controls; treat this as `array`.
    return mx.array(obj, dtype=_as_mx_dtype(dtype))


def zeros(shape, dtype=None):
    return mx.zeros(shape, dtype=_as_mx_dtype(dtype))


def ones(shape, dtype=None):
    return mx.ones(shape, dtype=_as_mx_dtype(dtype))


def empty(shape, dtype=None):
    # MLX doesn't provide an uninitialized array; fall back to zeros.
    return mx.zeros(shape, dtype=_as_mx_dtype(dtype))


def arange(*args, **kwargs):
    return mx.arange(*args, **kwargs)


def linspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    dtype = _as_mx_dtype(dtype)
    if endpoint:
        return mx.linspace(start, stop, num, dtype=dtype, axis=axis)
    if num <= 0:
        return mx.array([], dtype=dtype)
    # Create one extra sample and drop the last.
    res = mx.linspace(start, stop, num + 1, dtype=dtype, axis=axis)
    return res[:-1]


def reshape(a, newshape):
    return mx.reshape(a, newshape)


def ravel(a):
    return mx.reshape(a, (-1,))


def atleast_1d(a):
    arr = mx.array(a)
    if arr.ndim == 0:
        return mx.reshape(arr, (1,))
    return arr


def atleast_2d(a):
    arr = mx.array(a)
    if arr.ndim == 0:
        return mx.reshape(arr, (1, 1))
    if arr.ndim == 1:
        return mx.reshape(arr, (1, arr.shape[0]))
    return arr


def atleast_3d(a):
    arr = mx.array(a)
    if arr.ndim == 0:
        return mx.reshape(arr, (1, 1, 1))
    if arr.ndim == 1:
        return mx.reshape(arr, (1, arr.shape[0], 1))
    if arr.ndim == 2:
        return mx.reshape(arr, (1,) + arr.shape)
    return arr


def hstack(tup):
    arrays = [ravel(mx.array(a)) for a in tup]
    return mx.concatenate(arrays, axis=0)


def vstack(tup):
    arrays = [atleast_2d(a) for a in tup]
    return mx.concatenate(arrays, axis=0)


def squeeze(a, axis=None):
    return mx.squeeze(a, axis=axis)


def transpose(a, axes=None):
    return mx.transpose(a, axes) if axes is not None else mx.transpose(a)


def shape(a):
    return mx.array(a).shape


def isscalar(x):
    return mx.isscalar(x)


newaxis = None


# ---------------------------
# Math helpers
# ---------------------------

abs = mx.abs
add = mx.add
sqrt = mx.sqrt
exp = mx.exp
log = mx.log
log10 = mx.log10
log1p = mx.log1p
expm1 = mx.expm1
sin = mx.sin
cos = mx.cos
tan = mx.tan
sinh = mx.sinh
cosh = mx.cosh
tanh = mx.tanh
power = mx.power
floor = mx.floor
ceil = mx.ceil
mod = getattr(mx, "mod", None) or getattr(mx, "remainder", None)

def diff(a, n=1, axis=-1):
    if hasattr(mx, "diff"):
        return mx.diff(a, n=n, axis=axis)
    arr = mx.array(a)
    for _ in range(n):
        sl1 = [slice(None)] * arr.ndim
        sl2 = [slice(None)] * arr.ndim
        sl1[axis] = slice(1, None)
        sl2[axis] = slice(0, -1)
        arr = arr[tuple(sl1)] - arr[tuple(sl2)]
    return arr


def around(a, decimals=0):
    if decimals == 0:
        return mx.round(a)
    factor = mx.power(mx.array(10.0), mx.array(decimals))
    return mx.round(mx.multiply(a, factor)) / factor


# ---------------------------
# Logic/helpers
# ---------------------------

logical_and = mx.logical_and
logical_or = mx.logical_or
logical_not = mx.logical_not
def logical_xor(a, b):
    if hasattr(mx, "logical_xor"):
        return mx.logical_xor(a, b)
    a = mx.array(a, dtype=mx.bool_)
    b = mx.array(b, dtype=mx.bool_)
    return mx.not_equal(a, b)


def extract(condition, arr):
    cond = mx.array(condition, dtype=mx.bool_)
    return mx.array(arr)[cond]


# ---------------------------
# Reductions
# ---------------------------

sum = mx.sum
prod = mx.prod
mean = mx.mean
std = mx.std
var = mx.var
min = mx.min
max = mx.max
amin = mx.min
amax = mx.max
def unique(a, return_index=False, return_inverse=False, return_counts=False, axis=None):
    if axis is not None:
        raise NotImplementedError("unique(axis=...) not supported in MLX compat")
    if hasattr(mx, "unique"):
        return mx.unique(a, return_index=return_index, return_inverse=return_inverse, return_counts=return_counts)
    arr = mx.reshape(mx.array(a), (-1,))
    if arr.size == 0:
        out = mx.array([], dtype=arr.dtype)
        rets = [out]
        if return_index:
            rets.append(mx.array([], dtype=mx.int64))
        if return_inverse:
            rets.append(mx.array([], dtype=mx.int64))
        if return_counts:
            rets.append(mx.array([], dtype=mx.int64))
        return tuple(rets) if len(rets) > 1 else out
    sort_idx = mx.argsort(arr)
    sorted_arr = arr[sort_idx]
    first = mx.array([True], dtype=mx.bool_)
    if sorted_arr.size == 1:
        mask = first
    else:
        mask = mx.concatenate([first, mx.not_equal(sorted_arr[1:], sorted_arr[:-1])], axis=0)
    uniq = sorted_arr[mask]
    rets = [uniq]
    if return_index or return_inverse or return_counts:
        pos = mx.nonzero(mask)[0]
    if return_index:
        rets.append(sort_idx[pos])
    if return_inverse:
        # Inverse mapping via Python loop (small arrays only).
        inv_sorted = (mx.cumsum(mask.astype(mx.int64)) - 1).astype(mx.int64)
        inv = [0] * int(arr.size)
        inv_sorted_list = inv_sorted.tolist()
        sort_idx_list = sort_idx.tolist()
        for j, orig_i in enumerate(sort_idx_list):
            inv[orig_i] = inv_sorted_list[j]
        rets.append(mx.array(inv, dtype=mx.int64))
    if return_counts:
        pos_list = pos.tolist()
        pos_list.append(int(sorted_arr.size))
        counts = [pos_list[i + 1] - pos_list[i] for i in range(len(pos_list) - 1)]
        rets.append(mx.array(counts, dtype=mx.int64))
    return tuple(rets) if len(rets) > 1 else uniq


def count_nonzero(a, axis=None):
    return mx.sum(mx.not_equal(a, 0), axis=axis)


# ---------------------------
# Comparison
# ---------------------------

where = mx.where
clip = mx.clip
equal = mx.equal
not_equal = mx.not_equal
greater = mx.greater
greater_equal = mx.greater_equal
less = mx.less
less_equal = mx.less_equal
isfinite = mx.isfinite
isinf = mx.isinf
isnan = mx.isnan


# ---------------------------
# Indexing helpers
# ---------------------------

argsort = mx.argsort
argmax = mx.argmax
argmin = mx.argmin
sort = mx.sort


# ---------------------------
# Linear algebra helpers
# ---------------------------

def dot(a, b):
    return mx.matmul(a, b)


def vdot(a, b):
    return mx.vdot(a, b) if hasattr(mx, "vdot") else mx.sum(mx.conjugate(a) * b)

def inner(a, b):
    return mx.inner(a, b) if hasattr(mx, "inner") else mx.sum(a * b)

def outer(a, b):
    return mx.outer(a, b) if hasattr(mx, "outer") else mx.multiply(a[..., None], b[None, ...])


# ---------------------------
# Special constants
# ---------------------------

pi = mx.pi
inf = mx.inf
nan = mx.nan
float32 = getattr(mx, "float32", None)
float64 = getattr(mx, "float64", None)
complex64 = getattr(mx, "complex64", None)
intp = mx.int64
inexact = getattr(mx, "inexact", None)


# ---------------------------
# Masked assignment helpers
# ---------------------------

def putmask(a, mask, values):
    mask = mx.array(mask, dtype=mx.bool_)
    a[mask] = values
    return a


def place(a, mask, values):
    mask = mx.array(mask, dtype=mx.bool_)
    a[mask] = values
    return a


# ---------------------------
# Vectorize helper
# ---------------------------

def vectorize(pyfunc: Callable[..., Any], otypes=None, signature=None):
    # Avoid infinite recursion: this module may patch `mx.vectorize` to point to
    # this shim, so `hasattr(mx, "vectorize")` is not a safe guard.
    if _ORIG_MX_VECTORIZe is not None and _ORIG_MX_VECTORIZe is not vectorize:
        try:
            return _ORIG_MX_VECTORIZe(pyfunc, otypes=otypes, signature=signature)
        except TypeError:
            # Some MLX builds may expose a smaller signature.
            return _ORIG_MX_VECTORIZe(pyfunc)

    def _vectorized(*args, **kwargs):
        arrays = [mx.array(arg) for arg in args]
        if hasattr(mx, "broadcast_arrays"):
            arrays = mx.broadcast_arrays(*arrays)
        flat_arrays = [mx.reshape(arr, (-1,)) for arr in arrays]
        # Fallback to Python iteration
        try:
            iter_args = [arr.tolist() for arr in flat_arrays]
        except Exception:
            iter_args = [list(arr) for arr in flat_arrays]
        results = [pyfunc(*vals, **kwargs) for vals in zip(*iter_args)]
        out = mx.array(results)
        return mx.reshape(out, arrays[0].shape)

    return _vectorized


# ---------------------------
# Reduction helpers
# ---------------------------

def median(a, axis=None, keepdims=False):
    a = mx.array(a)
    if axis is None:
        a = mx.reshape(a, (-1,))
        axis = 0
    axis = int(axis)
    s = mx.sort(a, axis=axis)
    n = int(s.shape[axis])
    if n == 0:
        out = mx.array(float("nan"))
    elif n % 2 == 1:
        out = mx.take(s, n // 2, axis=axis)
    else:
        lo = mx.take(s, n // 2 - 1, axis=axis)
        hi = mx.take(s, n // 2, axis=axis)
        out = mx.divide(mx.add(lo, hi), mx.array(2.0))
    if keepdims:
        out = mx.expand_dims(out, axis=axis)
    return out


# ---------------------------
# Polynomial helpers (NumPy-style)
# ---------------------------

def _poly1d_coeffs(p):
    p = mx.array(p)
    if p.ndim == 0:
        return mx.reshape(p, (1,))
    return mx.reshape(p, (-1,))


def poly(seq_of_zeros):
    roots = _poly1d_coeffs(seq_of_zeros)
    # Start with polynomial 1.
    coeffs = mx.array([1.0])
    for r in roots.tolist():
        r = mx.array(r)
        out = mx.zeros((coeffs.shape[0] + 1,), dtype=mx.result_type(coeffs, r) if hasattr(mx, "result_type") else coeffs.dtype)
        out[:-1] = out[:-1] + coeffs
        out[1:] = out[1:] - r * coeffs
        coeffs = out
    return coeffs


def polyadd(a1, a2):
    a1 = _poly1d_coeffs(a1)
    a2 = _poly1d_coeffs(a2)
    if a1.shape[0] < a2.shape[0]:
        a1 = mx.concatenate([mx.zeros((a2.shape[0] - a1.shape[0],), dtype=a1.dtype), a1])
    elif a2.shape[0] < a1.shape[0]:
        a2 = mx.concatenate([mx.zeros((a1.shape[0] - a2.shape[0],), dtype=a2.dtype), a2])
    return a1 + a2


def polysub(a1, a2):
    a1 = _poly1d_coeffs(a1)
    a2 = _poly1d_coeffs(a2)
    if a1.shape[0] < a2.shape[0]:
        a1 = mx.concatenate([mx.zeros((a2.shape[0] - a1.shape[0],), dtype=a1.dtype), a1])
    elif a2.shape[0] < a1.shape[0]:
        a2 = mx.concatenate([mx.zeros((a1.shape[0] - a2.shape[0],), dtype=a2.dtype), a2])
    return a1 - a2


def polymul(a1, a2):
    a1 = _poly1d_coeffs(a1)
    a2 = _poly1d_coeffs(a2)
    out_dtype = mx.result_type(a1, a2) if hasattr(mx, "result_type") else a1.dtype
    out = mx.zeros((a1.shape[0] + a2.shape[0] - 1,), dtype=out_dtype)
    for i in range(a1.shape[0]):
        out[i:i + a2.shape[0]] = out[i:i + a2.shape[0]] + a1[i] * a2
    return out


def polydiv(u, v):
    u = _poly1d_coeffs(u)
    v = _poly1d_coeffs(v)
    if v.shape[0] == 0 or mx.all(v == 0):
        raise ZeroDivisionError("polynomial division")
    m = u.shape[0] - 1
    n = v.shape[0] - 1
    if m < n:
        return mx.zeros((1,), dtype=u.dtype), u
    q = mx.zeros((m - n + 1,), dtype=u.dtype)
    r = u.copy()
    for k in range(m - n + 1):
        d = r[k] / v[0]
        q = q.at[k].set(d) if hasattr(q, "at") else _set_index(q, k, d)
        r = (r.at[k:k + n + 1].set(r[k:k + n + 1] - d * v)
             if hasattr(r, "at") else _set_slice(r, k, n + 1, d, v))
    return q, r


def polyder(p, m=1):
    p = _poly1d_coeffs(p)
    if m <= 0:
        return p
    for _ in range(m):
        n = p.shape[0]
        if n <= 1:
            return mx.zeros((1,), dtype=p.dtype)
        powers = mx.arange(n - 1, 0, -1)
        p = p[:-1] * powers
    return p


def polyint(p, m=1, k=None):
    p = _poly1d_coeffs(p)
    if m <= 0:
        return p
    if k is None:
        k = 0
    for _ in range(m):
        n = p.shape[0]
        powers = mx.arange(n, 0, -1)
        p = mx.concatenate([p / powers, mx.array([k], dtype=p.dtype)])
        k = 0
    return p


def polyval(p, x):
    p = _poly1d_coeffs(p)
    x = mx.array(x)
    y = mx.zeros_like(x) + p[0]
    for c in p[1:]:
        y = y * x + c
    return y


def polyfit(x, y, deg):
    x = mx.array(x)
    y = mx.array(y)
    if x.ndim != 1:
        x = mx.reshape(x, (-1,))
    # Vandermonde matrix with descending powers
    powers = [mx.power(x, i) for i in range(deg, -1, -1)]
    vander = mx.stack(powers, axis=1)
    from scipy_mlx.linalg import lstsq
    coeffs, *_ = lstsq(vander, y)
    return coeffs


class poly1d:
    def __init__(self, c_or_r, r=False):
        if r:
            coeffs = poly(c_or_r)
        else:
            coeffs = _poly1d_coeffs(c_or_r)
        self.coeffs = mx.array(coeffs)
        self.coef = self.coeffs

    @property
    def order(self):
        return int(self.coeffs.shape[0] - 1)

    def __call__(self, x):
        return polyval(self.coeffs, x)

    def __repr__(self):
        return f"poly1d({self.coeffs})"

    def _binary_op(self, other, op):
        if isinstance(other, poly1d):
            other_coeffs = other.coeffs
        else:
            other_coeffs = other
        return poly1d(op(self.coeffs, other_coeffs))

    def __add__(self, other):
        return self._binary_op(other, polyadd)

    def __radd__(self, other):
        return self._binary_op(other, polyadd)

    def __sub__(self, other):
        return self._binary_op(other, polysub)

    def __rsub__(self, other):
        return poly1d(polysub(other, self.coeffs))

    def __mul__(self, other):
        return self._binary_op(other, polymul)

    def __rmul__(self, other):
        return self._binary_op(other, polymul)

    def __truediv__(self, other):
        if mx.isscalar(other):
            return poly1d(self.coeffs / other)
        q, r = polydiv(self.coeffs, other)
        return poly1d(q), poly1d(r)


class Polynomial:
    """Minimal Polynomial class compatible with numpy.polynomial.Polynomial."""
    def __init__(self, coef):
        self.coef = mx.array(coef)

    def __call__(self, x):
        x = mx.array(x)
        coef = self.coef
        if coef.ndim != 1:
            coef = mx.reshape(coef, (-1,))
        if coef.shape[0] == 0:
            return mx.zeros_like(x)
        y = mx.zeros_like(x) + coef[-1]
        for c in coef[-2::-1]:
            y = y * x + c
        return y

    def _align(self, other):
        other_coef = other.coef if isinstance(other, Polynomial) else mx.array(other)
        n = self.coef.shape[0]
        m = other_coef.shape[0]
        if n < m:
            pad = mx.zeros((m - n,), dtype=self.coef.dtype)
            a = mx.concatenate([self.coef, pad])
            b = other_coef
        elif m < n:
            pad = mx.zeros((n - m,), dtype=other_coef.dtype)
            a = self.coef
            b = mx.concatenate([other_coef, pad])
        else:
            a, b = self.coef, other_coef
        return a, b

    def __add__(self, other):
        a, b = self._align(other)
        return Polynomial(a + b)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        a, b = self._align(other)
        return Polynomial(a - b)

    def __rsub__(self, other):
        a, b = self._align(other)
        return Polynomial(b - a)

    def __mul__(self, other):
        if mx.isscalar(other):
            return Polynomial(self.coef * other)
        other_coef = other.coef if isinstance(other, Polynomial) else mx.array(other)
        a = self.coef
        b = other_coef
        if hasattr(mx, "result_type"):
            out_dtype = mx.result_type(a, b)
        else:
            out_dtype = a.dtype
        out = mx.zeros((a.shape[0] + b.shape[0] - 1,), dtype=out_dtype)
        for i in range(a.shape[0]):
            out[i:i + b.shape[0]] = out[i:i + b.shape[0]] + a[i] * b
        return Polynomial(out)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not mx.isscalar(other):
            raise TypeError("Polynomial division only supports scalar divisors.")
        return Polynomial(self.coef / other)

    def __pow__(self, power: int):
        if power < 0:
            raise ValueError("Power must be non-negative.")
        result = Polynomial([1])
        base = self
        for _ in range(power):
            result = result * base
        return result


# ---------------------------
# mx.r_ helper
# ---------------------------

def _as_1d_array(obj):
    arr = mx.array(obj)
    if arr.ndim == 0:
        return mx.reshape(arr, (1,))
    return mx.reshape(arr, (-1,))


def _concat_1d(*arrays):
    return mx.concatenate([_as_1d_array(a) for a in arrays], axis=0)


class _RClass:
    def __getitem__(self, key):
        if isinstance(key, tuple):
            parts = [self._handle_part(part) for part in key]
            return _concat_1d(*parts)
        return self._handle_part(key)

    def _handle_part(self, part):
        if isinstance(part, slice):
            return self._handle_slice(part)
        return mx.array(part)

    def _handle_slice(self, slc: slice):
        start = 0 if slc.start is None else slc.start
        stop = slc.stop
        step = 1 if slc.step is None else slc.step
        if isinstance(step, complex):
            num = int(abs(step.imag))
            return mx.linspace(start, stop, num)
        return mx.arange(start, stop, step)


r_ = _RClass()


# ---------------------------
# Misc utilities
# ---------------------------

def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
    x = mx.array(m)
    if y is not None:
        y = mx.array(y)
        x = mx.concatenate((x, y), axis=0 if rowvar else 1)
    if x.ndim == 1:
        x = mx.reshape(x, (1, -1))
    if not rowvar and x.ndim > 1:
        x = mx.transpose(x)
    if ddof is None:
        ddof = 0 if bias else 1
    avg = mx.mean(x, axis=1, keepdims=True)
    x = x - avg
    fact = x.shape[1] - ddof
    return mx.matmul(x, mx.transpose(x)) / mx.array(fact)


# ---------------------------
# Internal helpers for polydiv
# ---------------------------

def _set_index(arr, idx, value):
    arr[idx] = value
    return arr


def _set_slice(arr, start, length, d, v):
    arr[start:start + length] = arr[start:start + length] - d * v
    return arr


# ---------------------------
# Patch missing helpers into mx
# ---------------------------

mx.r_ = r_
mx.poly1d = poly1d
mx.polyval = polyval
mx.polyfit = polyfit
mx.polyadd = polyadd
mx.polysub = polysub
mx.polymul = polymul
mx.polydiv = polydiv
mx.polyder = polyder
mx.polyint = polyint
mx.poly = poly
mx.cov = cov
mx.Polynomial = Polynomial
if not hasattr(mx, "polynomial"):
    mx.polynomial = _types.SimpleNamespace(Polynomial=Polynomial)

# Additional NumPy-like aliases
mx.unique = unique
if _ORIG_MX_VECTORIZe is None:
    mx.vectorize = vectorize
if not hasattr(mx, "median"):
    mx.median = median

# NumPy-style newaxis
if not hasattr(mx, "newaxis"):
    mx.newaxis = None  # type: ignore[attr-defined]

# Minimal `mx.lib.stride_tricks` and `mx.pad` used in a few modules.
if not hasattr(mx, "lib"):
    mx.lib = _types.SimpleNamespace()  # type: ignore[attr-defined]
if not hasattr(mx.lib, "stride_tricks"):  # type: ignore[attr-defined]
    mx.lib.stride_tricks = _types.SimpleNamespace()  # type: ignore[attr-defined]


def _sliding_window_view(x, window_shape, axis=-1, writeable=False):
    _ = writeable
    x = mx.array(x)
    if isinstance(window_shape, (tuple, list)):
        if len(window_shape) != 1:
            raise NotImplementedError("Only 1D window_shape supported in MLX stride_tricks stub.")
        window_shape = int(window_shape[0])
    window_shape = int(window_shape)
    axis = int(axis)
    if axis < 0:
        axis += x.ndim
    if axis != x.ndim - 1:
        x_t = moveaxis(x, axis, -1)
        out = _sliding_window_view(x_t, window_shape, axis=-1, writeable=writeable)
        return moveaxis(out, -2, axis)
    n = x.shape[-1]
    w = window_shape
    if w <= 0 or w > n:
        raise ValueError("window_shape must be in [1, x.shape[axis]]")
    m = n - w + 1
    windows = [x[..., i:i + w] for i in range(m)]
    return mx.stack(windows, axis=-2)


def _pad_constant(x, pad_width, constant_values=0):
    x = mx.array(x)
    if isinstance(pad_width, int):
        pad_width = [(pad_width, pad_width)] * x.ndim
    elif isinstance(pad_width, (tuple, list)) and len(pad_width) == 2 and isinstance(pad_width[0], int):
        pad_width = [tuple(pad_width)] * x.ndim
    out = x
    for axis, (before, after) in enumerate(pad_width):
        if before:
            shape = list(out.shape)
            shape[axis] = before
            left = mx.multiply(mx.ones(shape, dtype=out.dtype), mx.array(constant_values, dtype=out.dtype))
            out = mx.concatenate([left, out], axis=axis)
        if after:
            shape = list(out.shape)
            shape[axis] = after
            right = mx.multiply(mx.ones(shape, dtype=out.dtype), mx.array(constant_values, dtype=out.dtype))
            out = mx.concatenate([out, right], axis=axis)
    return out


def pad(x, pad_width, mode="constant", constant_values=0, **kwargs):
    _ = kwargs
    if mode != "constant":
        raise NotImplementedError("Only constant padding is supported in mx.pad stub.")
    return _pad_constant(x, pad_width, constant_values=constant_values)


mx.lib.stride_tricks.sliding_window_view = _sliding_window_view  # type: ignore[attr-defined]
mx.lib.stride_tricks.as_strided = None  # type: ignore[attr-defined]
if not hasattr(mx, "pad"):
    mx.pad = pad  # type: ignore[attr-defined]

# Minimal masked array support (MLX-backed, limited semantics).
nomask = False


class _MaskedConstant:
    def __repr__(self) -> str:
        return "masked"


masked = _MaskedConstant()


def _normalize_mask(mask, shape):
    if mask is nomask or mask is False or mask is None:
        return mx.zeros(shape, dtype=mx.bool_)
    if mask is True:
        return mx.ones(shape, dtype=mx.bool_)
    mask_arr = mx.array(mask, dtype=mx.bool_)
    if mask_arr.shape != shape and hasattr(mx, "broadcast_to"):
        mask_arr = mx.broadcast_to(mask_arr, shape)
    return mask_arr


class MaskedArray:
    __array_priority__ = 1000

    def __init__(self, data, mask=nomask, dtype=None, copy=False, subok=False, ndmin=0):
        _ = subok  # unused, kept for signature compatibility
        data = mx.array(data, dtype=_as_mx_dtype(dtype))
        if ndmin and data.ndim < ndmin:
            new_shape = (1,) * (ndmin - data.ndim) + data.shape
            data = mx.reshape(data, new_shape)
        if copy:
            data = mx.array(data)
        self.data = data
        self._mask = _normalize_mask(mask, data.shape)

    @property
    def mask(self):
        return self._mask

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def copy(self):
        return MaskedArray(self.data, mask=self._mask, copy=True)

    def filled(self, fill_value=0):
        fill = mx.array(fill_value, dtype=self.data.dtype)
        return mx.where(self._mask, fill, self.data)

    def compressed(self):
        valid = mx.logical_not(self._mask)
        return mx.reshape(self.data[valid], (-1,))

    def count(self, axis=None):
        valid = mx.logical_not(self._mask)
        return mx.sum(valid, axis=axis)

    def ravel(self):
        return MaskedArray(mx.reshape(self.data, (-1,)), mask=mx.reshape(self._mask, (-1,)))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return MaskedArray(mx.reshape(self.data, shape), mask=mx.reshape(self._mask, shape))

    def transpose(self, axes=None):
        data = mx.transpose(self.data, axes) if axes is not None else mx.transpose(self.data)
        mask = mx.transpose(self._mask, axes) if axes is not None else mx.transpose(self._mask)
        return MaskedArray(data, mask=mask)

    @property
    def T(self):
        return self.transpose()

    def astype(self, dtype):
        return MaskedArray(self.data.astype(_as_mx_dtype(dtype)), mask=self._mask)

    def view(self, _type=None):
        if _type is None or _type is MaskedArray:
            return self
        if _type is array:
            return self.data
        return _type(self)

    def __getitem__(self, item):
        data = self.data[item]
        mask = self._mask[item]
        if mx.isscalar(data) and bool(mask):
            return masked
        return MaskedArray(data, mask=mask) if hasattr(data, "shape") else data

    def __setitem__(self, key, value):
        if value is masked:
            self._mask[key] = True
            return
        self.data[key] = value
        self._mask[key] = False

    def _binary_op(self, other, op):
        other_ma = _as_masked(other)
        data = op(self.data, other_ma.data)
        mask = mx.logical_or(self._mask, other_ma._mask)
        return MaskedArray(data, mask=mask)

    def __add__(self, other):
        return self._binary_op(other, mx.add)

    def __radd__(self, other):
        return self._binary_op(other, mx.add)

    def __sub__(self, other):
        return self._binary_op(other, mx.subtract)

    def __rsub__(self, other):
        other_ma = _as_masked(other)
        data = mx.subtract(other_ma.data, self.data)
        mask = mx.logical_or(self._mask, other_ma._mask)
        return MaskedArray(data, mask=mask)

    def __mul__(self, other):
        return self._binary_op(other, mx.multiply)

    def __rmul__(self, other):
        return self._binary_op(other, mx.multiply)

    def __truediv__(self, other):
        return self._binary_op(other, mx.divide)

    def __neg__(self):
        return MaskedArray(mx.negative(self.data), mask=self._mask)


def isMaskedArray(a):
    return isinstance(a, MaskedArray)


def _as_masked(a):
    if a is masked:
        return MaskedArray(mx.array(0.0), mask=True)
    return a if isinstance(a, MaskedArray) else MaskedArray(a)


def _unary_ma(op, a):
    a = _as_masked(a)
    return MaskedArray(op(a.data), mask=a._mask)


def _binary_ma(op, a, b):
    a = _as_masked(a)
    b = _as_masked(b)
    data = op(a.data, b.data)
    mask = mx.logical_or(a._mask, b._mask)
    return MaskedArray(data, mask=mask)


def _masked_mean(a, axis=None):
    valid = mx.logical_not(a._mask)
    count = mx.sum(valid, axis=axis)
    data = mx.where(valid, a.data, mx.zeros_like(a.data))
    total = mx.sum(data, axis=axis)
    return mx.where(count == 0, mx.nan, total / count)


def _masked_var(a, axis=None, ddof=0):
    valid = mx.logical_not(a._mask)
    count = mx.sum(valid, axis=axis)
    mean = _masked_mean(a, axis=axis)
    if axis is None:
        mean = mx.array(mean)
    data = mx.where(valid, a.data, mx.zeros_like(a.data))
    diff = data - mean
    num = mx.sum(diff * diff, axis=axis)
    denom = count - ddof
    return mx.where(denom <= 0, mx.nan, num / denom)


def _masked_std(a, axis=None, ddof=0):
    return mx.sqrt(_masked_var(a, axis=axis, ddof=ddof))


def ma_array(a, mask=nomask, dtype=None, copy=False, subok=False, ndmin=0):
    return MaskedArray(a, mask=mask, dtype=dtype, copy=copy, subok=subok, ndmin=ndmin)


def ma_asanyarray(a, dtype=None):
    return a if isinstance(a, MaskedArray) else MaskedArray(a, dtype=dtype)


def ma_asarray(a, dtype=None):
    return ma_asanyarray(a, dtype=dtype)


def ma_masked_array(a, mask=nomask, dtype=None, copy=False, subok=False, ndmin=0):
    return MaskedArray(a, mask=mask, dtype=dtype, copy=copy, subok=subok, ndmin=ndmin)


def ma_getmask(a):
    return a._mask if isinstance(a, MaskedArray) else nomask


def ma_mask_or(m1, m2, shrink=True):
    _ = shrink
    return mx.logical_or(mx.array(m1, dtype=mx.bool_), mx.array(m2, dtype=mx.bool_))


def ma_where(cond, x, y):
    x = _as_masked(x)
    y = _as_masked(y)
    cond = mx.array(cond, dtype=mx.bool_)
    data = mx.where(cond, x.data, y.data)
    mask = mx.where(cond, x._mask, y._mask)
    return MaskedArray(data, mask=mask)


def ma_fix_invalid(a, copy=True):
    a = _as_masked(a)
    mask = mx.logical_or(a._mask, mx.logical_or(mx.isnan(a.data), mx.isinf(a.data)))
    return MaskedArray(a.data, mask=mask, copy=copy)


def ma_masked_invalid(a):
    return ma_fix_invalid(a, copy=False)


def ma_masked_equal(a, value):
    a = _as_masked(a)
    mask = mx.logical_or(a._mask, mx.equal(a.data, value))
    return MaskedArray(a.data, mask=mask)


def ma_masked_values(a, value):
    return ma_masked_equal(a, value)


def ma_masked_less(a, value):
    a = _as_masked(a)
    mask = mx.logical_or(a._mask, mx.less(a.data, value))
    return MaskedArray(a.data, mask=mask)


def ma_masked_less_equal(a, value):
    a = _as_masked(a)
    mask = mx.logical_or(a._mask, mx.less_equal(a.data, value))
    return MaskedArray(a.data, mask=mask)


def ma_masked_greater(a, value):
    a = _as_masked(a)
    mask = mx.logical_or(a._mask, mx.greater(a.data, value))
    return MaskedArray(a.data, mask=mask)


def ma_masked_greater_equal(a, value):
    a = _as_masked(a)
    mask = mx.logical_or(a._mask, mx.greater_equal(a.data, value))
    return MaskedArray(a.data, mask=mask)


def ma_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    arr = _as_masked(arr)
    axis = axis % arr.ndim
    data = moveaxis(arr.data, axis, 0)
    mask = moveaxis(arr._mask, axis, 0)
    out_data = []
    out_mask = []
    for i in range(data.shape[0]):
        res = func1d(MaskedArray(data[i], mask=mask[i]), *args, **kwargs)
        res_ma = _as_masked(res)
        out_data.append(res_ma.data)
        out_mask.append(res_ma._mask)
    stacked = mx.stack(out_data, axis=0)
    stacked_mask = mx.stack(out_mask, axis=0)
    stacked = moveaxis(stacked, 0, axis)
    stacked_mask = moveaxis(stacked_mask, 0, axis)
    return MaskedArray(stacked, mask=stacked_mask)


def ma_sort(a, axis=-1):
    a = _as_masked(a)
    data = mx.sort(a.filled(mx.inf), axis=axis)
    mask = mx.sort(a._mask, axis=axis)
    return MaskedArray(data, mask=mask)


def ma_median(a, axis=None):
    a = _as_masked(a)
    data = a.compressed() if axis is None else ma_sort(a, axis=axis).data
    if axis is None:
        n = data.shape[0]
        if n == 0:
            return masked
        mid = n // 2
        if n % 2 == 1:
            return data[mid]
        return (data[mid - 1] + data[mid]) / 2
    # axis-specific median
    n = data.shape[axis]
    mid = n // 2
    if n % 2 == 1:
        return mx.take(data, mid, axis=axis)
    return (mx.take(data, mid - 1, axis=axis) + mx.take(data, mid, axis=axis)) / 2


def ma_vstack(tup):
    arrays = [_as_masked(a) for a in tup]
    data = mx.concatenate([a.data for a in arrays], axis=0)
    mask = mx.concatenate([a._mask for a in arrays], axis=0)
    return MaskedArray(data, mask=mask)


def ma_column_stack(tup):
    arrays = [_as_masked(a) for a in tup]
    data = mx.stack([a.data for a in arrays], axis=1)
    mask = mx.stack([a._mask for a in arrays], axis=1)
    return MaskedArray(data, mask=mask)


def ma_minimum_reduce(a, axis=None):
    a = _as_masked(a)
    data = a.filled(mx.inf)
    return mx.min(data, axis=axis)


def ma_maximum_reduce(a, axis=None):
    a = _as_masked(a)
    data = a.filled(-mx.inf)
    return mx.max(data, axis=axis)


def ma_arange(*args, **kwargs):
    return MaskedArray(mx.arange(*args, **kwargs))


def ma_empty(shape, dtype=None, mask=nomask):
    data = mx.zeros(shape, dtype=_as_mx_dtype(dtype))
    return MaskedArray(data, mask=mask)


def ma_zeros(shape, dtype=None, mask=nomask):
    data = mx.zeros(shape, dtype=_as_mx_dtype(dtype))
    return MaskedArray(data, mask=mask)


def ma_ones(shape, dtype=None, mask=nomask):
    data = mx.ones(shape, dtype=_as_mx_dtype(dtype))
    return MaskedArray(data, mask=mask)


def ma_mask_cols(a):
    a = _as_masked(a)
    col_mask = mx.any(a._mask, axis=0)
    mask = mx.logical_or(a._mask, col_mask[None, :])
    return MaskedArray(a.data, mask=mask)


def ma_mask_rowcols(a, axis=0):
    a = _as_masked(a)
    row_mask = mx.any(a._mask, axis=1)
    col_mask = mx.any(a._mask, axis=0)
    if axis == 0:
        mask = mx.logical_or(a._mask, row_mask[:, None])
        mask = mx.logical_or(mask, col_mask[None, :])
    else:
        mask = mx.logical_or(a._mask, col_mask[None, :])
    return MaskedArray(a.data, mask=mask)


def ma_corrcoef(a, rowvar=True):
    a = _as_masked(a)
    data = a.filled(mx.nan) if mx.any(a._mask) else a.data
    if hasattr(mx, "corrcoef"):
        return mx.corrcoef(data, rowvar=rowvar)
    c = cov(data, rowvar=rowvar)
    diag = mx.sqrt(mx.diag(c))
    return c / mx.outer(diag, diag)


def ma_allclose(a, b, rtol=1e-7, atol=0.0):
    a = _as_masked(a)
    b = _as_masked(b)
    return mx.allclose(a.filled(mx.nan), b.filled(mx.nan), rtol=rtol, atol=atol)


ma = _types.SimpleNamespace(
    array=ma_array,
    asarray=ma_asarray,
    asanyarray=ma_asanyarray,
    masked_array=ma_masked_array,
    getmask=ma_getmask,
    mask_or=ma_mask_or,
    where=ma_where,
    fix_invalid=ma_fix_invalid,
    masked_invalid=ma_masked_invalid,
    masked_equal=ma_masked_equal,
    masked_values=ma_masked_values,
    masked_less=ma_masked_less,
    masked_less_equal=ma_masked_less_equal,
    masked_greater=ma_masked_greater,
    masked_greater_equal=ma_masked_greater_equal,
    apply_along_axis=ma_apply_along_axis,
    sort=ma_sort,
    median=ma_median,
    vstack=ma_vstack,
    column_stack=ma_column_stack,
    arange=ma_arange,
    empty=ma_empty,
    zeros=ma_zeros,
    ones=ma_ones,
    mask_cols=ma_mask_cols,
    mask_rowcols=ma_mask_rowcols,
    corrcoef=ma_corrcoef,
    minimum=_types.SimpleNamespace(reduce=ma_minimum_reduce),
    maximum=_types.SimpleNamespace(reduce=ma_maximum_reduce),
    allclose=ma_allclose,
    sqrt=lambda a: _unary_ma(mx.sqrt, a),
    power=lambda a, b: _binary_ma(mx.power, a, b),
    log=lambda a: _unary_ma(mx.log, a),
    exp=lambda a: _unary_ma(mx.exp, a),
    abs=lambda a: _unary_ma(mx.abs, a),
    sign=lambda a: _unary_ma(mx.sign, a),
    count=lambda a, axis=None: _as_masked(a).count(axis=axis),
    compressed=lambda a: _as_masked(a).compressed(),
    filled=lambda a, fill_value=0: _as_masked(a).filled(fill_value),
    var=lambda a, axis=None, ddof=0: _masked_var(_as_masked(a), axis=axis, ddof=ddof),
    std=lambda a, axis=None, ddof=0: _masked_std(_as_masked(a), axis=axis, ddof=ddof),
    MaskedArray=MaskedArray,
    masked=masked,
    nomask=nomask,
    isMaskedArray=isMaskedArray,
)

mx.ma = ma

# dtype/typing helpers
def dtype(obj):
    if isinstance(obj, MxDType):
        return obj
    if isinstance(obj, str):
        char, itemsize, byteorder = _parse_dtype_string(obj)
        mx_dtype = _CHAR_TO_MX.get(char)
        if mx_dtype is None:
            raise TypeError(f"Unsupported dtype spec {obj!r}")
        return MxDType(char, itemsize, mx_dtype, byteorder=byteorder)
    if isinstance(obj, type):
        # Python scalar types
        if obj is bool:
            return MxDType("?", 1, mx.bool_)
        if obj is int:
            return MxDType("q", 8, mx.int64)
        if obj is float:
            return MxDType("d", 8, mx.float64)
        if obj is complex:
            return MxDType("D", 16, mx.complex128)
    # MLX dtype objects
    try:
        if obj == mx.float32:
            return MxDType("f", 4, mx.float32)
        if obj == mx.float64:
            return MxDType("d", 8, mx.float64)
        if obj == mx.int8:
            return MxDType("b", 1, mx.int8)
        if obj == mx.uint8:
            return MxDType("B", 1, mx.uint8)
        if obj == mx.int16:
            return MxDType("h", 2, mx.int16)
        if obj == mx.uint16:
            return MxDType("H", 2, mx.uint16)
        if obj == mx.int32:
            return MxDType("i", 4, mx.int32)
        if obj == mx.uint32:
            return MxDType("I", 4, mx.uint32)
        if obj == mx.int64:
            return MxDType("q", 8, mx.int64)
        if obj == mx.complex64:
            return MxDType("F", 8, mx.complex64)
        if obj == mx.complex128:
            return MxDType("D", 16, mx.complex128)
        if obj == mx.bool_:
            return MxDType("?", 1, mx.bool_)
    except Exception:
        pass
    # Array-like
    try:
        arr = mx.array(obj)
        return dtype(arr.dtype)
    except Exception as exc:
        raise TypeError(f"No dtype constructor available for {obj!r}") from exc


# Patch dtype constructor into mx for call sites that expect `mx.dtype(...)`.
if not hasattr(mx, "dtype"):
    mx.dtype = dtype  # type: ignore[attr-defined]

# Patch missing array constructors/helpers that SciPy code expects.
if not hasattr(mx, "empty"):
    mx.empty = empty  # type: ignore[attr-defined]
if not hasattr(mx, "asarray"):
    mx.asarray = asarray  # type: ignore[attr-defined]
if not hasattr(mx, "asanyarray"):
    mx.asanyarray = asanyarray  # type: ignore[attr-defined]
if not hasattr(mx, "ascontiguousarray"):
    mx.ascontiguousarray = ascontiguousarray  # type: ignore[attr-defined]

def finfo(dt):
    if _ORIG_MX_FINFO is not None:
        mx_dt = _as_mx_dtype(dt)
        base = _ORIG_MX_FINFO(mx_dt)

        class _Finfo:
            __slots__ = ("dtype", "eps", "max", "min", "tiny", "smallest_normal")

            def __init__(self, base_obj, tiny):
                self.dtype = base_obj.dtype
                self.eps = base_obj.eps
                self.max = base_obj.max
                self.min = base_obj.min
                self.tiny = tiny
                self.smallest_normal = tiny

        # MLX `finfo` doesn't expose `tiny`; provide best-effort constants.
        if mx_dt == mx.float16:
            tiny = 6.1035156e-05
        elif mx_dt == mx.float32:
            tiny = 1.17549435e-38
        elif mx_dt == mx.float64:
            tiny = 2.2250738585072014e-308
        else:
            # Default to a reasonable lower bound.
            tiny = 0.0

        return _Finfo(base, tiny)
    raise TypeError("No finfo available")

def iinfo(dt):
    if _ORIG_MX_IINFO is not None:
        return _ORIG_MX_IINFO(_as_mx_dtype(dt))
    raise TypeError("No iinfo available")

def issubdtype(dt, kind):
    if hasattr(mx, "issubdtype"):
        return mx.issubdtype(_as_mx_dtype(dt), kind)
    return False

def result_type(*args):
    if hasattr(mx, "result_type"):
        return mx.result_type(*args)
    # Best-effort fallback: compute from arrays
    arrays = [mx.array(a) for a in args]
    return arrays[0].dtype if arrays else mx.float32


# Patch finfo/iinfo wrappers into mx so call sites can use `mx.finfo(float)`.
if _ORIG_MX_FINFO is not None:
    mx.finfo = finfo  # type: ignore[assignment]
if _ORIG_MX_IINFO is not None:
    mx.iinfo = iinfo  # type: ignore[assignment]

def frombuffer(buffer, dtype=None, count=-1, offset=0):
    if hasattr(mx, "frombuffer"):
        return mx.frombuffer(buffer, dtype=_as_mx_dtype(dtype), count=count, offset=offset)
    dtype_fn = globals()["dtype"]
    dt = dtype if isinstance(dtype, MxDType) else dtype_fn(dtype or "b")
    key = (dt.char, dt.itemsize)
    typecode_map = {
        ("b", 1): "b",
        ("B", 1): "B",
        ("c", 1): "B",
        ("h", 2): "h",
        ("H", 2): "H",
        ("i", 4): "i",
        ("I", 4): "I",
        ("l", 8): "q",
        ("q", 8): "q",
        ("Q", 8): "Q",
        ("f", 4): "f",
        ("d", 8): "d",
    }
    if key not in typecode_map:
        raise TypeError(f"Unsupported dtype for frombuffer: {dt}")
    typecode = typecode_map[key]
    mv = memoryview(buffer)[offset:]
    if count is not None and count >= 0:
        mv = mv[:count * dt.itemsize]
    arr = _py_array(typecode)
    arr.frombytes(mv.tobytes() if isinstance(mv, memoryview) else bytes(mv))
    if dt.byteorder in ("<", ">") and dt.byteorder != _NATIVE_BYTEORDER and dt.itemsize > 1:
        arr.byteswap()
    return mx.array(arr, dtype=dt.mx_dtype)

def searchsorted(a, v, side="left", sorter=None):
    if hasattr(mx, "searchsorted"):
        return mx.searchsorted(a, v, side=side)
    arr = mx.array(a)
    vals = mx.array(v)
    if arr.ndim != 1:
        arr = mx.reshape(arr, (-1,))
    if sorter is not None:
        arr = arr[sorter]
    if vals.ndim == 0:
        vals = mx.reshape(vals, (1,))
    comp = arr[None, :] <= vals[..., None] if side == "right" else arr[None, :] < vals[..., None]
    idx = mx.sum(comp, axis=-1)
    return mx.reshape(idx, v.shape) if hasattr(v, "shape") else idx

def moveaxis(a, source, destination):
    if hasattr(mx, "moveaxis"):
        return mx.moveaxis(a, source, destination)
    arr = mx.array(a)
    ndim = arr.ndim
    if isinstance(source, int):
        source = (source,)
    if isinstance(destination, int):
        destination = (destination,)
    if len(source) != len(destination):
        raise ValueError("source and destination must have the same number of axes")
    source = [(s + ndim) % ndim for s in source]
    destination = [(d + ndim) % ndim for d in destination]
    order = [ax for ax in range(ndim) if ax not in source]
    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)
    return mx.transpose(arr, order)

def zeros_like(a, dtype=None):
    return mx.zeros_like(a, dtype=_as_mx_dtype(dtype))

def concatenate(tup, axis=0):
    return mx.concatenate(tup, axis=axis)

def eye(n, m=None, k=0, dtype=None):
    if m is None:
        m = n
    return mx.eye(n, m, k=k, dtype=_as_mx_dtype(dtype))

def triu(m, k=0):
    return mx.triu(m, k=k) if hasattr(mx, "triu") else _triu_fallback(m, k=k)

def _triu_fallback(m, k=0):
    arr = mx.array(m)
    rows, cols = arr.shape[-2], arr.shape[-1]
    r = mx.arange(rows)[:, None]
    c = mx.arange(cols)[None, :]
    mask = c - r >= k
    return mx.where(mask, arr, mx.zeros_like(arr))

def real(a):
    return mx.real(a) if hasattr(mx, "real") else mx.array(a).real

def imag(a):
    return mx.imag(a) if hasattr(mx, "imag") else mx.array(a).imag

little_endian = "little" if sys.byteorder == "little" else "big"


def add_newdoc(module, name, doc):
    """Minimal replacement for numpy.lib.add_newdoc."""
    try:
        import sys
        obj = sys.modules.get(module)
        if obj is None:
            return
        for part in name.split('.'):
            obj = getattr(obj, part)
        obj.__doc__ = doc
    except Exception:
        # Best effort only.
        return
