from typing import Any

NDArray = Any

def bandwidth(a: NDArray) -> tuple[int, int]: ...

def issymmetric(
    a: NDArray,
    atol: None | float = ...,
    rtol: None | float = ...,
) -> bool: ...

def ishermitian(
    a: NDArray,
    atol: None | float = ...,
    rtol: None | float = ...,
) -> bool: ...
