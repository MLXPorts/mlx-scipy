"""
MLX stub for SciPy's compiled ndimage extension (`ndimage._nd_image`).

SciPy's ndimage relies on a compiled C extension for performance. This MLX port
does not ship compiled extensions; provide a stub so that the high-level Python
API modules can be imported. Calling any of these functions will raise
`NotImplementedError`.
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(f"scipy_mlx.ndimage._nd_image.{name} is not yet implemented for MLX.")

    _fn.__name__ = name
    return _fn


def __getattr__(name: str) -> Any:  # pragma: no cover
    return _unimplemented(name)

