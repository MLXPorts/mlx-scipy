"""
MLX stub for SciPy's compiled dfitpack extension (`interpolate._dfitpack`).

This module normally contains Fortran wrappers used by FITPACK. We provide just
enough structure for import-time code paths (notably `types.intvar.dtype`) while
raising `NotImplementedError` for computational routines.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from scipy_mlx._lib._mlx_compat import dtype as _dtype


types = SimpleNamespace(
    intvar=SimpleNamespace(dtype=_dtype("i4")),
)


def __getattr__(name: str) -> Any:  # pragma: no cover
    raise NotImplementedError(
        f"scipy_mlx.interpolate._dfitpack.{name} is not yet implemented for MLX."
    )

