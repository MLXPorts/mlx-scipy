"""
MLX stub for SciPy's compiled sparse tools (`sparse._sparsetools`).

Upstream SciPy provides many sparse-format kernels as compiled extensions.
This MLX port does not ship those extensions; we provide stubs so that the
pure-Python sparse matrix classes can be imported.
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(f"scipy_mlx.sparse._sparsetools.{name} is not yet implemented for MLX.")

    _fn.__name__ = name
    return _fn


# Names imported at module import time across scipy_mlx.sparse
csr_hstack = _unimplemented("csr_hstack")

csr_tocsc = _unimplemented("csr_tocsc")
expandptr = _unimplemented("expandptr")

dia_matmat = _unimplemented("dia_matmat")
dia_matvec = _unimplemented("dia_matvec")
dia_matvecs = _unimplemented("dia_matvecs")
dia_tocsr = _unimplemented("dia_tocsr")

coo_tocsr = _unimplemented("coo_tocsr")
coo_todense = _unimplemented("coo_todense")
coo_todense_nd = _unimplemented("coo_todense_nd")
coo_matvec = _unimplemented("coo_matvec")
coo_matvec_nd = _unimplemented("coo_matvec_nd")
coo_matmat_dense = _unimplemented("coo_matmat_dense")
coo_matmat_dense_nd = _unimplemented("coo_matmat_dense_nd")

csr_tobsr = _unimplemented("csr_tobsr")
csr_count_blocks = _unimplemented("csr_count_blocks")
get_csr_submatrix = _unimplemented("get_csr_submatrix")
csr_sample_values = _unimplemented("csr_sample_values")
csr_sample_offsets = _unimplemented("csr_sample_offsets")
csr_todense = _unimplemented("csr_todense")
csr_row_index = _unimplemented("csr_row_index")
csr_row_slice = _unimplemented("csr_row_slice")
csr_column_index1 = _unimplemented("csr_column_index1")
csr_column_index2 = _unimplemented("csr_column_index2")
csr_diagonal = _unimplemented("csr_diagonal")
csr_has_canonical_format = _unimplemented("csr_has_canonical_format")
csr_eliminate_zeros = _unimplemented("csr_eliminate_zeros")
csr_sum_duplicates = _unimplemented("csr_sum_duplicates")
csr_has_sorted_indices = _unimplemented("csr_has_sorted_indices")
csr_sort_indices = _unimplemented("csr_sort_indices")
csr_matmat_maxnnz = _unimplemented("csr_matmat_maxnnz")
csr_matmat = _unimplemented("csr_matmat")

bsr_matvec = _unimplemented("bsr_matvec")
bsr_matvecs = _unimplemented("bsr_matvecs")
bsr_matmat = _unimplemented("bsr_matmat")
bsr_transpose = _unimplemented("bsr_transpose")
bsr_sort_indices = _unimplemented("bsr_sort_indices")
bsr_tocsr = _unimplemented("bsr_tocsr")


def __getattr__(name: str) -> Any:  # pragma: no cover
    return _unimplemented(name)

