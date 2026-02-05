"""
Stub for SciPy's `_comb` extension.

Upstream SciPy uses a compiled implementation; this MLX source-tree port uses
placeholders to keep imports working.
"""

from __future__ import annotations

from typing import Any


def _comb_int(*args: Any, **kwargs: Any):
    raise NotImplementedError("scipy_mlx.special._comb_int is not implemented for MLX.")


__all__ = ["_comb_int"]

