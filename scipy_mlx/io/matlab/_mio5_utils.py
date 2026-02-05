"""
MLX stub for SciPy's compiled MAT v5 utilities (`io.matlab._mio5_utils`).

SciPy's MAT-file reader uses a Cython extension for performance. This MLX port
keeps the public I/O APIs importable, but does not provide a full MAT v5 parser.
"""

from __future__ import annotations


class VarReader5:  # pragma: no cover
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("scipy_mlx.io.matlab VarReader5 is not yet implemented for MLX.")

