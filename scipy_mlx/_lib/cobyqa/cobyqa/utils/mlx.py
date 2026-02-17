import contextlib

import mlx.core as mx


MLX_ARRAY_TYPE = type(mx.array(0.0))


@contextlib.contextmanager
def mx_printoptions(*args, **kwargs):
    """
    Compatibility context manager for printing options.

    MLX does not currently expose a ``printoptions`` context manager. When
    absent, this is a no-op context manager.
    """
    if hasattr(mx, "printoptions"):
        with mx.printoptions(*args, **kwargs):
            yield
    else:
        with contextlib.nullcontext():
            yield


def mx_array(x):
    return mx.asarray(x)


def mx_copy(x):
    return mx.array(x)


def _as_1d(array):
    arr = mx_array(array)
    if arr.ndim == 0:
        return mx.reshape(arr, (1,))
    return arr


def _as_2d_row(arr):
    array = mx_array(arr)
    if array.ndim == 0:
        return mx.reshape(array, (1, 1))
    if array.ndim == 1:
        return mx.reshape(array, (1, -1))
    return array


def _as_2d_col(arr):
    array = mx_array(arr)
    if array.ndim == 0:
        return mx.reshape(array, (1, 1))
    if array.ndim == 1:
        return mx.reshape(array, (-1, 1))
    return array


def mx_vstack(blocks):
    """
    MLX replacement for ``numpy.vstack``.
    """
    return mx.concatenate([_as_2d_row(block) for block in blocks], axis=0)


def mx_hstack(blocks):
    """
    MLX replacement for ``numpy.hstack``.
    """
    return mx.concatenate([_as_2d_col(block) for block in blocks], axis=1)


def mx_block(blocks):
    """
    MLX replacement for ``numpy.block``.
    """
    if not isinstance(blocks, (list, tuple)):
        raise TypeError("`blocks` must be a list or tuple.")
    if len(blocks) == 0:
        raise ValueError("`blocks` must be non-empty.")
    if isinstance(blocks[0], (list, tuple)):
        # A list of rows.
        return mx_vstack([
            mx_hstack(row) if isinstance(row, (list, tuple)) else _as_2d_row(row)
            for row in blocks
        ])
    return mx_hstack(blocks)


def mx_full_like(arr, fill_value):
    """
    MLX replacement for ``numpy.full_like``.
    """
    array = mx_array(arr)
    return mx.full(
        array.shape,
        mx.array(fill_value, dtype=array.dtype),
        dtype=array.dtype,
    )


def mx_flatnonzero(x):
    """
    MLX replacement for ``numpy.flatnonzero``.
    """
    idxs = mx.nonzero(_as_1d(x) != 0)[0]
    return idxs


def mx_r_(arrays):
    """
    MLX replacement for ``numpy.r_``.
    """
    return mx.concatenate([_as_1d(array) for array in arrays], axis=0)


def mx_c_(arrays):
    """
    MLX replacement for ``numpy.c_``.
    """
    return mx_hstack([_as_2d_col(array) for array in arrays])


def _finite_masked(arr, fill_value):
    array = mx_array(arr)
    finite = mx.isfinite(array)
    if array.ndim == 0:
        if mx.all(finite):
            return array
        return mx.array(fill_value, dtype=array.dtype)
    count = mx.sum(mx.astype(finite, mx.int32), axis=None)
    if count == 0:
        return mx.array(mx.nan, dtype=array.dtype)
    return mx.where(finite, array, fill_value)


def mx_nanmin(x, axis=None):
    """
    MLX replacement for ``numpy.nanmin``.
    """
    array = mx_array(x)
    if array.ndim == 0:
        return array
    if axis is None:
        finite = mx.isfinite(array)
        if mx.sum(mx.astype(finite, mx.int32), axis=None) == 0:
            return mx.array(mx.nan, dtype=array.dtype)
        return mx.min(mx.where(finite, array, mx.inf), axis=None)
    result = mx.min(_finite_masked(array, mx.inf), axis=axis)
    finite_count = mx.sum(
        mx.astype(mx.isfinite(array), mx.int32),
        axis=axis,
    )
    return mx.where(finite_count == 0, mx.nan, result)


def mx_nanmax(x, axis=None):
    """
    MLX replacement for ``numpy.nanmax``.
    """
    array = mx_array(x)
    if array.ndim == 0:
        return array
    if axis is None:
        finite = mx.isfinite(array)
        if mx.sum(mx.astype(finite, mx.int32), axis=None) == 0:
            return mx.array(mx.nan, dtype=array.dtype)
        return mx.max(mx.where(finite, array, -mx.inf), axis=None)
    result = mx.max(_finite_masked(array, -mx.inf), axis=axis)
    finite_count = mx.sum(
        mx.astype(mx.isfinite(array), mx.int32),
        axis=axis,
    )
    return mx.where(finite_count == 0, mx.nan, result)
