"""
Minimal replacement for `numpy.testing` for the MLX port.

Only a small subset is provided - enough to keep legacy modules importable.
"""

from __future__ import annotations

import platform
import sys
import unittest
from typing import Any

import mlx.core as mx

HAS_REFCOUNT = hasattr(sys, "gettotalrefcount")
IS_PYPY = platform.python_implementation() == "PyPy"
TestCase = unittest.TestCase
verbose = 0


def _as_mx(x: Any) -> mx.array:
    return x if isinstance(x, mx.array) else mx.array(x)


def assert_(expr: Any, msg: str | None = None) -> None:
    if not expr:
        raise AssertionError("" if msg is None else msg)


def assert_equal(actual: Any, desired: Any, err_msg: str = "", verbose: bool = True) -> None:
    if isinstance(actual, mx.array) or isinstance(desired, mx.array):
        ok = bool(mx.array_equal(_as_mx(actual), _as_mx(desired)))
    else:
        ok = actual == desired
    if not ok:
        raise AssertionError(err_msg or f"{actual!r} != {desired!r}")


def assert_array_equal(actual: Any, desired: Any, err_msg: str = "", verbose: bool = True) -> None:
    if not bool(mx.array_equal(_as_mx(actual), _as_mx(desired))):
        raise AssertionError(err_msg or "arrays are not equal")


def assert_allclose(
    actual: Any,
    desired: Any,
    rtol: float = 1e-7,
    atol: float = 0.0,
    err_msg: str = "",
    verbose: bool = True,
) -> None:
    if not bool(mx.allclose(_as_mx(actual), _as_mx(desired), rtol=rtol, atol=atol)):
        raise AssertionError(err_msg or "arrays are not close")


def assert_almost_equal(
    actual: Any,
    desired: Any,
    decimal: int = 7,
    err_msg: str = "",
    verbose: bool = True,
) -> None:
    tol = 10.0 ** (-int(decimal))
    assert_allclose(actual, desired, rtol=0.0, atol=tol, err_msg=err_msg, verbose=verbose)


def assert_array_almost_equal(
    actual: Any,
    desired: Any,
    decimal: int = 6,
    err_msg: str = "",
    verbose: bool = True,
) -> None:
    tol = 10.0 ** (-int(decimal))
    assert_allclose(actual, desired, rtol=0.0, atol=tol, err_msg=err_msg, verbose=verbose)


def assert_approx_equal(
    actual: Any,
    desired: Any,
    significant: int = 7,
    err_msg: str = "",
    verbose: bool = True,
) -> None:
    tol = 10.0 ** (-int(significant))
    assert_allclose(actual, desired, rtol=0.0, atol=tol, err_msg=err_msg, verbose=verbose)


def assert_array_less(actual: Any, desired: Any, err_msg: str = "", verbose: bool = True) -> None:
    a = _as_mx(actual)
    d = _as_mx(desired)
    if not bool(mx.all(mx.less(a, d))):
        raise AssertionError(err_msg or "actual is not strictly less than desired")


def assert_array_almost_equal_nulp(
    actual: Any,
    desired: Any,
    nulp: int = 1,
    err_msg: str = "",
    verbose: bool = True,
) -> None:
    a = _as_mx(actual)
    d = _as_mx(desired)
    eps = mx.finfo(a.dtype).eps if hasattr(mx, "finfo") else 1e-7
    atol = float(nulp) * float(eps)
    if not bool(mx.allclose(a, d, rtol=0.0, atol=atol)):
        raise AssertionError(err_msg or "arrays differ more than nulp")


__all__ = [
    "HAS_REFCOUNT",
    "IS_PYPY",
    "TestCase",
    "verbose",
    "assert_",
    "assert_allclose",
    "assert_almost_equal",
    "assert_approx_equal",
    "assert_array_almost_equal",
    "assert_array_almost_equal_nulp",
    "assert_array_equal",
    "assert_array_less",
    "assert_equal",
]

