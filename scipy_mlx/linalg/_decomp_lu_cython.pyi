import mlx.core as mx
from numpy.typing import NDArray
from typing import TypeVar

# this mimicks the `ctypedef fused lapack_t`
_LapackT = TypeVar("_LapackT", mx.float32, mx.float64, mx.complex64, mx.complex128)

def lu_dispatcher(a: NDArray[_LapackT], u: NDArray[_LapackT], piv: NDArray[mx.integer],
                  permute_l: bool) -> None: ...
