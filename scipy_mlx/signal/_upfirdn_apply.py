"""
Stub for the compiled `_upfirdn_apply` extension.

Used by `scipy_mlx.signal._upfirdn`. This MLX port does not yet provide a
high-performance implementation.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any


class mode_enum(IntEnum):
    constant = 0
    symmetric = 1
    reflect = 2
    edge = 3
    wrap = 4


def _output_len(*args: Any, **kwargs: Any) -> int:
    raise NotImplementedError("scipy_mlx.signal._output_len is not implemented for MLX.")


def _apply(*args: Any, **kwargs: Any):
    raise NotImplementedError("scipy_mlx.signal._apply is not implemented for MLX.")

