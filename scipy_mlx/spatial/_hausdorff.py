"""
MLX implementation of directed Hausdorff distance.

Upstream SciPy implements this in C for performance and includes an optimized
algorithm with random shuffling. For this MLX-first port, we provide a
deterministic, straightforward implementation sufficient for correctness and
to keep higher-level imports functional.
"""

from __future__ import annotations

import mlx.core as mx


def directed_hausdorff(u: mx.array, v: mx.array, rng=None):
    """
    Compute the directed Hausdorff distance from `u` to `v`.

    Parameters
    ----------
    u, v : mx.array
        Arrays of shape (M, N) and (O, N).
    rng : ignored
        Present for API-compatibility.

    Returns
    -------
    d : mx.array
        Scalar array with the directed Hausdorff distance.
    index_1 : mx.array
        Scalar array index into `u` corresponding to the Hausdorff pair.
    index_2 : mx.array
        Scalar array index into `v` corresponding to the Hausdorff pair.
    """
    u = mx.array(u)
    v = mx.array(v)

    # Pairwise squared distances: (M, O)
    diff = mx.subtract(mx.expand_dims(u, axis=1), mx.expand_dims(v, axis=0))
    dist_sq = mx.sum(mx.multiply(diff, diff), axis=2)

    # For each u[i], find nearest v[j].
    min_dist_sq = mx.min(dist_sq, axis=1)
    j_all = mx.argmin(dist_sq, axis=1)

    # The directed Hausdorff distance is max_i min_j d(u[i], v[j]).
    i = mx.argmax(min_dist_sq)
    j = mx.take(j_all, i)
    d = mx.sqrt(mx.take(min_dist_sq, i))
    return d, i, j

