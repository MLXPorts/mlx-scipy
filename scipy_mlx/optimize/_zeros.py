"""
MLX stub for SciPy's compiled root-finding helpers (`optimize._zeros`).
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(f"scipy_mlx.optimize._zeros.{name} is not yet implemented for MLX.")
    _fn.__name__ = name
    return _fn


bisect = _unimplemented("bisect")
ridder = _unimplemented("ridder")
brentq = _unimplemented("brentq")
brenth = _unimplemented("brenth")
newton = _unimplemented("newton")
toms748 = _unimplemented("toms748")


def __getattr__(name: str) -> Any:  # pragma: no cover
    return _unimplemented(name)

