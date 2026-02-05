"""
Pure-Python fallback for SciPy's `_ccallback_c` extension.

Upstream SciPy ships a compiled extension for PyCapsule handling. For this MLX
source-tree implementation we provide a minimal stub so imports succeed.

Note: This does *not* provide actual PyCapsule interoperability.
"""

from __future__ import annotations


def check_capsule(obj) -> bool:
    # No PyCapsule support in the pure-python fallback.
    return False


def get_raw_capsule(func, signature, context):
    # Store the pieces in a tuple; `_ccallback.LowLevelCallable.signature`
    # retrieves the signature via `get_capsule_signature`.
    return (func, signature, context)


def get_capsule_signature(item) -> str | None:
    if isinstance(item, tuple) and len(item) >= 2:
        return item[1]
    return None

