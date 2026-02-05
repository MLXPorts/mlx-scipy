"""
MLX stub for SciPy's compiled MINPACK extension (`optimize._minpack`).
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(f"scipy_mlx.optimize._minpack.{name} is not yet implemented for MLX.")
    _fn.__name__ = name
    return _fn


hybrd = _unimplemented("hybrd")
hybrj = _unimplemented("hybrj")
lmdif = _unimplemented("lmdif")
lmder = _unimplemented("lmder")


def __getattr__(name: str) -> Any:  # pragma: no cover
    return _unimplemented(name)

