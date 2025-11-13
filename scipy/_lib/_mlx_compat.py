"""
MLX compatibility layer for scipy.

This module provides compatibility functions to use MLX (Apple's machine learning
framework) as a backend for scipy operations, similar to how NumPy is used.

MLX is optimized for Apple Silicon and provides a NumPy-like API.
"""

import sys
from typing import Any, TypeGuard, cast
from collections.abc import Hashable
from functools import lru_cache

__all__ = ['is_mlx_array', 'is_mlx_namespace', 'mlx_available']


# Check if MLX is available
try:
    import mlx.core as mx
    mlx_available = True
except (ImportError, OSError):
    # OSError occurs when MLX shared library is not available
    mx = None
    mlx_available = False


@lru_cache(100)
def _issubclass_fast(cls: type, modname: str, clsname: str) -> bool:
    """Fast subclass check without importing the module if not loaded."""
    try:
        mod = sys.modules[modname]
    except KeyError:
        return False
    parent_cls = getattr(mod, clsname, None)
    if parent_cls is None:
        return False
    return issubclass(cls, parent_cls)


def is_mlx_array(x: object) -> bool:
    """
    Return True if `x` is an MLX array.

    This function does not import MLX if it has not already been imported
    and is therefore cheap to use.

    Parameters
    ----------
    x : object
        Object to check

    Returns
    -------
    bool
        True if x is an MLX array, False otherwise

    Examples
    --------
    >>> import mlx.core as mx  # doctest: +SKIP
    >>> a = mx.array([1, 2, 3])  # doctest: +SKIP
    >>> is_mlx_array(a)  # doctest: +SKIP
    True
    >>> is_mlx_array([1, 2, 3])  # doctest: +SKIP
    False
    """
    if not mlx_available or 'mlx.core' not in sys.modules:
        return False
    
    cls = cast(Hashable, type(x))
    return _issubclass_fast(cls, "mlx.core", "array")


def is_mlx_namespace(xp: Any) -> bool:
    """
    Return True if `xp` is the MLX namespace.

    Parameters
    ----------
    xp : module
        Module to check

    Returns
    -------
    bool
        True if xp is mlx.core module, False otherwise
    """
    if not mlx_available:
        return False
    
    return xp.__name__ == 'mlx.core'
