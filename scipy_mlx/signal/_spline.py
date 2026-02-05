"""
Stub for the compiled `scipy.signal._spline` extension.
"""

from __future__ import annotations

from typing import Any


def sepfir2d(*args: Any, **kwargs: Any):
    raise NotImplementedError("scipy_mlx.signal.sepfir2d is not implemented for MLX.")


def symiirorder1_ic(*args: Any, **kwargs: Any):
    raise NotImplementedError("scipy_mlx.signal.symiirorder1_ic is not implemented for MLX.")


def symiirorder2_ic_fwd(*args: Any, **kwargs: Any):
    raise NotImplementedError("scipy_mlx.signal.symiirorder2_ic_fwd is not implemented for MLX.")


def symiirorder2_ic_bwd(*args: Any, **kwargs: Any):
    raise NotImplementedError("scipy_mlx.signal.symiirorder2_ic_bwd is not implemented for MLX.")


__all__ = [
    "sepfir2d",
    "symiirorder1_ic",
    "symiirorder2_ic_fwd",
    "symiirorder2_ic_bwd",
]
