"""
Minimal utilities for MATLAB I/O.

Upstream SciPy includes additional utilities for MAT-file parsing. This MLX port
keeps the public API importable; for now these helpers are conservative no-ops.
"""

from __future__ import annotations


def squeeze_element(arr):
    return arr


def chars_to_strings(arr):
    return arr

