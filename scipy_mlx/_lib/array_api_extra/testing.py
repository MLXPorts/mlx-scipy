"""
Testing helpers for array_api_extra.

This is a minimal stub to satisfy SciPy's imports in this MLX fork.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, TypeVar

T = TypeVar("T")


def lazy_xp_function(func: Callable[..., T]) -> Callable[..., T]:
    # In upstream this wraps functions for lazy backends (e.g. JAX). For MLX we
    # keep the function unchanged.
    return func


@contextmanager
def patch_lazy_xp_functions():
    # No-op for MLX-only.
    yield

