"""
MLX stub for SciPy's compiled QUADPACK wrapper (`integrate._quadpack`).

SciPy's `quad` family uses compiled wrappers around the Fortran QUADPACK
library. This MLX port does not ship those extensions.
"""

from __future__ import annotations


def __getattr__(name):  # pragma: no cover
    raise NotImplementedError(
        f"scipy_mlx.integrate._quadpack.{name} is not yet implemented for MLX."
    )

