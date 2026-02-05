
import mlx.core as mx
from typing import Any

NDArray = Any
FloatingArray = Any
ComplexArray = Any
FloatingComplexArray = Any


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
