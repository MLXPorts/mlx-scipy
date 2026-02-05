import mlx.core as mx
from typing import TypeVar, Any

NDArray = Any

# this mimicks the `ctypedef fused lapack_t`
_LapackT = TypeVar("_LapackT", mx.float32, mx.float64, mx.complex64, mx.complex128)

def lu_dispatcher(a: NDArray, u: NDArray, piv: NDArray,
                  permute_l: bool) -> None: ...
