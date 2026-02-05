"""
MLX stub for SciPy's compiled piecewise polynomial backend (`interpolate._ppoly`).
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(f"scipy_mlx.interpolate._ppoly.{name} is not yet implemented for MLX.")
    _fn.__name__ = name
    return _fn


evaluate = _unimplemented("evaluate")
integrate = _unimplemented("integrate")
fix_continuity = _unimplemented("fix_continuity")
real_roots = _unimplemented("real_roots")
evaluate_bernstein = _unimplemented("evaluate_bernstein")
evaluate_nd = _unimplemented("evaluate_nd")


def __getattr__(name: str) -> Any:  # pragma: no cover
    return _unimplemented(name)

