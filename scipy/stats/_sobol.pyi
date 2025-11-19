import mlx.core as mx
from scipy._lib._util import IntNumber
from typing import Literal

def _initialize_v(
    v : mx.array,
    dim : IntNumber,
    bits: IntNumber
) -> None: ...

def _cscramble (
    dim : IntNumber,
    bits: IntNumber,
    ltm : mx.array,
    sv: mx.array
) -> None: ...

def _fill_p_cumulative(
    p: mx.array,
    p_cumulative: mx.array
) -> None: ...

def _draw(
    n : IntNumber,
    num_gen: IntNumber,
    dim: IntNumber,
    scale: float,
    sv: mx.array,
    quasi: mx.array,
    sample: mx.array
    ) -> None: ...

def _fast_forward(
    n: IntNumber,
    num_gen: IntNumber,
    dim: IntNumber,
    sv: mx.array,
    quasi: mx.array
    ) -> None: ...

def _categorize(
    draws: mx.array,
    p_cumulative: mx.array,
    result: mx.array
    ) -> None: ...

_MAXDIM: Literal[21201]
_MAXDEG: Literal[18]

def _test_find_index(
    p_cumulative: mx.array,
    size: int, 
    value: float
    ) -> int: ...
