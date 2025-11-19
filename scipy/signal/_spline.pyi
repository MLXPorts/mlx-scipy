
import mlx.core as mx
from numpy.typing import NDArray

FloatingArray = NDArray[mx.float32] | NDArray[mx.float64]
ComplexArray = NDArray[mx.complex64] | NDArray[mx.complex128]
FloatingComplexArray = FloatingArray | ComplexArray


def symiirorder1_ic(signal: FloatingComplexArray,
                    c0: float,
                    z1: float,
                    precision: float) -> FloatingComplexArray:
    ...


def symiirorder2_ic_fwd(signal: FloatingArray,
                        r: float,
                        omega: float,
                        precision: float) -> FloatingArray:
    ...


def symiirorder2_ic_bwd(signal: FloatingArray,
                        r: float,
                        omega: float,
                        precision: float) -> FloatingArray:
    ...


def sepfir2d(input: FloatingComplexArray,
             hrow: FloatingComplexArray,
             hcol: FloatingComplexArray) -> FloatingComplexArray:
    ...
