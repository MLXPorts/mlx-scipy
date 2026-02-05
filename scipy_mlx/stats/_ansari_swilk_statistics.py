"""
MLX stub for SciPy's compiled `_ansari_swilk_statistics` extension.

This extension provides fast helpers for certain statistical tests in upstream
SciPy. It is not yet implemented in this MLX port.
"""

from __future__ import annotations

from typing import Any, Tuple


def gscale(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    raise NotImplementedError("scipy_mlx.stats.gscale is not implemented for MLX.")


def swilk(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    raise NotImplementedError("scipy_mlx.stats.swilk is not implemented for MLX.")


__all__ = ["gscale", "swilk"]

