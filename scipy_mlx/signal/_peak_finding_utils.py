"""
Pure-MLX fallback implementations for SciPy's peak-finding Cython helpers.

Upstream SciPy provides `_peak_finding_utils` as a compiled extension for speed.
This MLX port includes Python implementations sufficient for basic use and to
keep import chains functional.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import mlx.core as mx


class PeakPropertyWarning(RuntimeWarning):
    pass


def _local_maxima_1d(x):
    """Find local maxima in a 1D signal.

    Returns
    -------
    peaks, left_edges, right_edges : mx.array
        For this fallback implementation, `left_edges` and `right_edges` are
        equal to `peaks` (plateaus are not handled specially).
    """
    x = mx.reshape(mx.array(x), (-1,))
    n = int(x.shape[0])
    if n < 3:
        empty = mx.array([], dtype=mx.int64)
        return empty, empty, empty

    mid = x[1:-1]
    mask = mx.logical_and(mx.greater(mid, x[:-2]), mx.greater(mid, x[2:]))
    peaks = mx.add(mx.nonzero(mask)[0], mx.array(1))
    return peaks, peaks, peaks


def _select_by_peak_distance(peaks, priority, distance):
    """Select peaks separated by at least `distance` samples (greedy)."""
    peaks = mx.array(peaks, dtype=mx.int64)
    priority = mx.array(priority)
    n = int(peaks.shape[0])
    if n == 0:
        return mx.array([], dtype=mx.bool_)

    p_list = peaks.tolist()
    pr_list = priority.tolist()
    order = sorted(range(n), key=lambda i: pr_list[i], reverse=True)
    keep = [True] * n
    dist = int(distance)

    for idx in order:
        if not keep[idx]:
            continue
        p = p_list[idx]
        for j in range(n):
            if j == idx or not keep[j]:
                continue
            if abs(p_list[j] - p) < dist:
                keep[j] = False

    return mx.array(keep, dtype=mx.bool_)


def _peak_prominences(x, peaks, wlen=None):
    """Compute peak prominences (naive O(n*k) fallback)."""
    x = mx.reshape(mx.array(x), (-1,))
    peaks = mx.array(peaks, dtype=mx.int64)
    n = int(x.shape[0])
    wlen = None if wlen is None else int(wlen)

    x_list = x.tolist()
    peaks_list = peaks.tolist()

    prominences = []
    left_bases = []
    right_bases = []

    for p in peaks_list:
        peak_val = x_list[p]
        left = 0 if wlen is None else max(0, p - wlen // 2)
        right = (n - 1) if wlen is None else min(n - 1, p + wlen // 2)

        # Search for the lowest valley on each side until a higher/equal value blocks.
        i = p
        left_min = peak_val
        left_min_i = p
        while i > left:
            i -= 1
            v = x_list[i]
            if v > peak_val:
                break
            if v < left_min:
                left_min = v
                left_min_i = i

        i = p
        right_min = peak_val
        right_min_i = p
        while i < right:
            i += 1
            v = x_list[i]
            if v > peak_val:
                break
            if v < right_min:
                right_min = v
                right_min_i = i

        base_val = max(left_min, right_min)
        prominences.append(peak_val - base_val)
        left_bases.append(left_min_i)
        right_bases.append(right_min_i)

    return (
        mx.array(prominences),
        mx.array(left_bases, dtype=mx.int64),
        mx.array(right_bases, dtype=mx.int64),
    )


def _peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases):
    """Compute peak widths at a relative height (naive fallback)."""
    x = mx.reshape(mx.array(x), (-1,))
    peaks = mx.array(peaks, dtype=mx.int64)
    prominences = mx.array(prominences)
    left_bases = mx.array(left_bases, dtype=mx.int64)
    right_bases = mx.array(right_bases, dtype=mx.int64)
    rel_height = float(rel_height)

    x_list = x.tolist()
    peaks_list = peaks.tolist()
    prom_list = prominences.tolist()
    lb_list = left_bases.tolist()
    rb_list = right_bases.tolist()

    widths = []
    width_heights = []
    left_ips = []
    right_ips = []

    for p, prom, lb, rb in zip(peaks_list, prom_list, lb_list, rb_list):
        peak_val = x_list[p]
        h = peak_val - prom * rel_height
        width_heights.append(h)

        # Left intersection point
        li = float(lb)
        for i in range(p, lb, -1):
            if x_list[i] < h:
                x0, x1 = x_list[i], x_list[i + 1]
                if x1 == x0:
                    li = float(i)
                else:
                    li = i + (h - x0) / (x1 - x0)
                break

        # Right intersection point
        ri = float(rb)
        for i in range(p, rb):
            if x_list[i] < h:
                x0, x1 = x_list[i], x_list[i - 1]
                if x1 == x0:
                    ri = float(i)
                else:
                    ri = i - (h - x0) / (x1 - x0)
                break

        left_ips.append(li)
        right_ips.append(ri)
        widths.append(ri - li)

    return (
        mx.array(widths),
        mx.array(width_heights),
        mx.array(left_ips),
        mx.array(right_ips),
    )


__all__ = [
    "PeakPropertyWarning",
    "_local_maxima_1d",
    "_select_by_peak_distance",
    "_peak_prominences",
    "_peak_widths",
]

