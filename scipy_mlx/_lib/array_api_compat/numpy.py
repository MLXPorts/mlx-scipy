"""
MLX-backed replacement for SciPy's `array_api_compat.numpy` namespace.

Upstream SciPy uses this module as the default backend when Array API support
is disabled. In this MLX fork we treat that "numpy compat" namespace as MLX.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["mx"]

# Provide an Array API version marker for code paths that expect it.
__array_api_version__ = "2023.12"


def __getattr__(name: str):
    return getattr(mx, name)

