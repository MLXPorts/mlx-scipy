"""
Build configuration stub for scipy_mlx.

Upstream SciPy generates this module at build/install time. For this MLX-first
source tree we provide a minimal implementation so `import scipy_mlx` works
from the repo checkout.
"""

from __future__ import annotations


def show() -> None:
    print("scipy_mlx: source-tree build config stub (no compiled build info).")

