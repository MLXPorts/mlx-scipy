"""
MLX stub for SciPy's compiled ``_biasedurn`` extension.

Upstream SciPy ships efficient implementations of Fisher's and Wallenius'
noncentral hypergeometric distributions in a compiled extension. This MLX port
does not yet provide these algorithms; the classes below exist to keep import
chains working for `scipy_mlx.stats`.
"""

from __future__ import annotations

from typing import Any


class _BaseBiasedUrn:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, name: str) -> Any:  # pragma: no cover
        raise NotImplementedError(
            "scipy_mlx.stats._biasedurn is not implemented for MLX; "
            f"attribute {name!r} is unavailable."
        )


class _PyFishersNCHypergeometric(_BaseBiasedUrn):
    pass


class _PyWalleniusNCHypergeometric(_BaseBiasedUrn):
    pass


class _PyStochasticLib3(_BaseBiasedUrn):
    pass


__all__ = [
    "_PyFishersNCHypergeometric",
    "_PyWalleniusNCHypergeometric",
    "_PyStochasticLib3",
]

