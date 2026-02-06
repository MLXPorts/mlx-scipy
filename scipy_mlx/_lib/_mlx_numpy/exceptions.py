"""
Minimal replacement for `numpy.exceptions` for the MLX port.
"""

from __future__ import annotations


class ComplexWarning(RuntimeWarning):
    """Raised when casting complex values to real."""


class VisibleDeprecationWarning(DeprecationWarning):
    """Compatibility warning; used in legacy SciPy tests."""


__all__ = [
    "ComplexWarning",
    "VisibleDeprecationWarning",
]

