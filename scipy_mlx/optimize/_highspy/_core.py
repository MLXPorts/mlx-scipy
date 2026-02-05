"""
MLX stub for the HiGHS Python bindings (`optimize._highspy._core`).
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(
            f"scipy_mlx.optimize._highspy._core.{name} is not yet implemented for MLX."
        )
    _fn.__name__ = name
    return _fn


def __getattr__(name: str) -> Any:  # pragma: no cover
    return _unimplemented(name)

