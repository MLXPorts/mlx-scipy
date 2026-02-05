from typing import Any

NDArray = Any

def pick_pade_structure(a: NDArray) -> tuple[int, int]: ...

def pade_UV_calc(Am: NDArray, m: int) -> int: ...
