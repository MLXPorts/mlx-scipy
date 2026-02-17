import mlx.core as mx


EPS = mx.finfo(mx.float64).eps


def get_arrays_tol(*arrays):
    """
    Get a relative tolerance for a set of arrays.

    Parameters
    ----------
    *arrays: tuple
        Set of `numpy.ndarray` to get the tolerance for.

    Returns
    -------
    float
        Relative tolerance for the set of arrays.

    Raises
    ------
    ValueError
        If no array is provided.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided.")
    weight = mx.array(1.0, dtype=mx.float64)
    size = 0
    for array in arrays:
        arr = mx.asarray(array, dtype=mx.float64)
        finite_abs = mx.where(mx.isfinite(arr), mx.abs(arr), 0.0)
        weight = mx.maximum(weight, mx.max(finite_abs))
        size = max(size, arr.size)
    return 10.0 * EPS * max(size, 1.0) * weight


def exact_1d_array(x, message):
    """
    Preprocess a 1-dimensional array.

    Parameters
    ----------
    x : array_like
        Array to be preprocessed.
    message : str
        Error message if `x` cannot be interpreter as a 1-dimensional array.

    Returns
    -------
    `numpy.ndarray`
        Preprocessed array.
    """
    x = mx.atleast_1d(mx.squeeze(mx.asarray(x, dtype=mx.float64)))
    if x.ndim != 1:
        raise ValueError(message)
    return x


def exact_2d_array(x, message):
    """
    Preprocess a 2-dimensional array.

    Parameters
    ----------
    x : array_like
        Array to be preprocessed.
    message : str
        Error message if `x` cannot be interpreter as a 2-dimensional array.

    Returns
    -------
    `numpy.ndarray`
        Preprocessed array.
    """
    x = mx.atleast_2d(mx.asarray(x, dtype=mx.float64))
    if x.ndim != 2:
        raise ValueError(message)
    return x
