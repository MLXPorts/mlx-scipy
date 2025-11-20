"""
MLX Compatibility Layer

Provides missing MLX functions that don't exist in mlx.core
but are needed for NumPy compatibility in scipy.linalg.
"""

import mlx.core as mx




def asarray_chkfinite(a, dtype=None, order=None):
    """
    Convert input to MLX array, checking for NaNs and Infs.

    Parameters
    ----------
    a : array_like
        Input data
    dtype : data-type, optional
        Target dtype
    order : ignored
        Not used, kept for NumPy compatibility

    Returns
    -------
    out : array
        MLX array

    Raises
    ------
    ValueError
        If input contains NaNs or Infs
    """
    arr = mx.array(a, dtype=dtype)
    if not mx.all(mx.isfinite(arr)):
        raise ValueError("array must not contain infs or NaNs")
    return arr


def empty_like(prototype, dtype=None, order=None, subok=None, shape=None):
    """
    Return a new uninitialized array with same shape and type as prototype.

    Parameters
    ----------
    prototype : array_like
        The shape and dtype of prototype define these same attributes of the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order, subok : ignored
        Not used, kept for NumPy compatibility
    shape : int or sequence of ints, optional
        Overrides the shape of the result.

    Returns
    -------
    out : array
        Array of uninitialized data with same shape and type as prototype.
    """
    if shape is None:
        shape = prototype.shape
    if dtype is None:
        dtype = prototype.dtype

    # MLX doesn't have empty, use zeros as fallback
    return mx.zeros(shape, dtype=dtype)


def flatnonzero(a):
    """
    Return indices that are non-zero in the flattened version of a.

    This is equivalent to mx.nonzero(mx.ravel(a))[0].

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    res : array
        Output array, containing the indices of the elements of a.ravel()
        that are non-zero.
    """
    flat = mx.reshape(a, (-1,))
    return mx.where(flat != 0)[0]


def iscomplex(x):
    """
    Returns a bool array, where True if input element is complex.

    What is tested is whether the input has a non-zero imaginary part.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : array of bools
        Output array.
    """
    if mx.issubdtype(x.dtype, mx.complexfloating):
        return mx.not_equal(x.imag, 0)
    else:
        return mx.zeros(x.shape, dtype=mx.bool_)


def iscomplexobj(x):
    """
    Check for a complex type or an array of complex numbers.

    The type of the input is checked, not the value. Even if the input
    has an imaginary part equal to zero, iscomplexobj evaluates to True.

    Parameters
    ----------
    x : any
        The input can be of any type and shape.

    Returns
    -------
    iscomplexobj : bool
        The return value, True if x is of a complex type or has at least
        one complex element.
    """
    try:
        dtype = x.dtype
        return mx.issubdtype(dtype, mx.complexfloating)
    except AttributeError:
        # Not an array, check if it's a complex Python type
        return isinstance(x, complex)


# Add these functions to mx namespace for convenience
mx.array_chkfinite = asarray_chkfinite
mx.empty_like = empty_like
mx.flatnonzero = flatnonzero
mx.iscomplex = iscomplex
mx.iscomplexobj = iscomplexobj
