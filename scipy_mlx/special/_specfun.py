"""
Stub module for SciPy's `_specfun` extension.

The upstream SciPy implementation is a compiled extension generated from
`_specfun.pyx`. This MLX port currently provides a minimal placeholder so the
pure-Python special-function wrappers can be imported.
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(f"scipy_mlx.special._specfun.{name} is not implemented for MLX.")
    _fn.__name__ = name
    return _fn


__getattr__ = lambda name: _unimplemented(name)

