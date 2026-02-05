"""
MLX stub for SciPy's compiled `cluster._vq` extension.

Upstream SciPy provides a compiled extension for vector quantization (VQ) and
K-means kernels. This MLX port does not include compiled extensions; we provide
stubs so that the pure-Python wrappers can be imported.
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(f"scipy_mlx.cluster._vq.{name} is not yet implemented for MLX.")

    _fn.__name__ = name
    return _fn


def __getattr__(name: str) -> Any:  # pragma: no cover
    return _unimplemented(name)

