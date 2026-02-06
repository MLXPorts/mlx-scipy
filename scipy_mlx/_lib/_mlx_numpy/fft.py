"""
Minimal replacement for `numpy.fft` for the MLX port.
"""

from __future__ import annotations

from typing import Any

import mlx.core.fft as _fft


def __getattr__(name: str) -> Any:
    return getattr(_fft, name)


__all__ = []

