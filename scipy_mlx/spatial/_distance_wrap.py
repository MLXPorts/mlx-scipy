"""
MLX stub for SciPy's spatial distance wrap extension.

Upstream SciPy ships a compiled extension (`_distance_wrap`) that provides
low-level output-buffer-based implementations for several distance kernels.

This MLX port keeps the pure-Python APIs importable; the compiled functions
are not available, so we provide stubs that raise `NotImplementedError` when
called.
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(
            f"scipy_mlx.spatial._distance_wrap.{name} is not yet implemented for MLX."
        )

    _fn.__name__ = name
    return _fn


def __getattr__(name: str) -> Any:  # pragma: no cover
    return _unimplemented(name)

