"""
Minimal MLX-first subset of SciPy's vendored `array_api_compat`.

SciPy's upstream `array_api_compat` supports many array backends (NumPy, CuPy,
PyTorch, JAX, Dask, array_api_strict, ...). For this repository we target MLX
(`mlx.core`) as the runtime backend, so we provide only the small surface area
needed by `scipy_mlx._lib._array_api` and related helpers.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import mlx.core as mx


def is_array_api_obj(x: Any) -> bool:
    # MLX arrays are Array-API-like for our purposes.
    return hasattr(x, "__class__") and getattr(x.__class__, "__module__", "").startswith("mlx")


def is_lazy_array(x: Any) -> bool:
    # MLX uses lazy execution in many cases; treat arrays as lazy.
    return is_array_api_obj(x)


def is_numpy_array(x: Any) -> bool:
    return False


def is_cupy_array(x: Any) -> bool:
    return False


def is_torch_array(x: Any) -> bool:
    return False


def is_jax_array(x: Any) -> bool:
    return False


def is_dask_array(x: Any) -> bool:
    return False


def size(x: Any) -> int:
    return int(mx.array(x).size)


def device(x: Any) -> str:
    # MLX doesn't expose device strings like other frameworks; keep it simple.
    return "cpu"


def is_numpy_namespace(xp: ModuleType) -> bool:
    return False


def is_cupy_namespace(xp: ModuleType) -> bool:
    return False


def is_torch_namespace(xp: ModuleType) -> bool:
    return False


def is_jax_namespace(xp: ModuleType) -> bool:
    return False


def is_dask_namespace(xp: ModuleType) -> bool:
    return False


def is_array_api_strict_namespace(xp: ModuleType) -> bool:
    return False


def array_namespace(*xs: Any) -> ModuleType:
    # MLX-only: always return mlx.core.
    return mx
