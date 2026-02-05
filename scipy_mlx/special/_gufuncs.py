"""
Stub for SciPy's compiled gufuncs in `scipy.special`.

Provides placeholder symbols so `scipy_mlx.special` can be imported from the
source tree without compiled extensions.
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(f"scipy_mlx.special.{name} is not yet implemented for MLX.")
    _fn.__name__ = name
    return _fn


_lqn = _unimplemented("_lqn")
_lqmn = _unimplemented("_lqmn")
_rctj = _unimplemented("_rctj")
_rcty = _unimplemented("_rcty")

__all__ = ["_lqn", "_lqmn", "_rctj", "_rcty"]

