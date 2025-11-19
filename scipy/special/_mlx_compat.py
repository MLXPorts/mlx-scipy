"""
MLX Compatibility Layer for scipy.special

Provides missing MLX functions that don't exist in mlx.core
but are needed for NumPy compatibility in scipy.special.
"""

import mlx.core as mx


def place(arr, mask, vals):
    """
    Change elements of an array based on conditional and input values.

    Similar to ``arr[mask] = vals``.

    Parameters
    ----------
    arr : array
        Array to put data into.
    mask : array (bool)
        Boolean mask array. Must be the same size as arr.
    vals : scalar or array
        Values to put into arr. Only the first N elements are used, where
        N is the number of True values in mask.

    Returns
    -------
    None : modifies arr in place through MLX's array semantics

    Notes
    -----
    In MLX, we cannot truly modify arrays in place, so this returns a new array.
    The calling code should assign the result back.
    """
    # Convert inputs to MLX arrays
    arr = mx.array(arr)
    mask = mx.array(mask, dtype=mx.bool_)

    # Use mx.where to conditionally select values
    return mx.where(mask, vals, arr)


def extract(condition, arr):
    """
    Return the elements of an array that satisfy some condition.

    This is equivalent to ``arr[condition]``.

    Parameters
    ----------
    condition : array (bool)
        Boolean array with same shape as arr
    arr : array
        Input array

    Returns
    -------
    extract : array
        Rank 1 array of values from arr where condition is True.
    """
    arr = mx.array(arr)
    condition = mx.array(condition, dtype=mx.bool_)
    return arr[condition]


def sinc(x):
    """
    Return the sinc function.

    The sinc function is defined as::

        sinc(x) = sin(pi*x) / (pi*x)    for x != 0
        sinc(0) = 1

    Parameters
    ----------
    x : array
        Input array

    Returns
    -------
    out : array
        sinc(x), which has the same shape as the input.
    """
    x = mx.array(x)
    # MLX has a built-in sinc function that uses the normalized definition
    # sinc(x) = sin(pi*x)/(pi*x), which matches NumPy's behavior
    # However, let's check if it exists first
    if hasattr(mx, 'sinc'):
        return mx.sinc(x)

    # Fallback implementation
    y = mx.multiply(mx.array(mx.pi, dtype=x.dtype), x)
    # Use where to handle the x=0 case
    return mx.where(
        mx.equal(x, mx.array(0.0, dtype=x.dtype)),
        mx.array(1.0, dtype=x.dtype),
        mx.divide(mx.sin(y), y)
    )


def isscalar(element):
    """
    Returns True if the type of element is a scalar type.

    Parameters
    ----------
    element : any
        Input argument, can be of any type and shape.

    Returns
    -------
    val : bool
        True if element is a scalar type, False if it is not.
    """
    # Check for Python scalars
    if isinstance(element, (int, float, complex, bool)):
        return True

    # Check for MLX arrays with size 1 (0-d or 1-element arrays)
    if hasattr(element, 'shape'):
        return element.shape == () or (len(element.shape) == 1 and element.shape[0] == 1)

    return False


def hstack(tup):
    """
    Stack arrays in sequence horizontally (column wise).

    This is equivalent to concatenation along the second axis,
    except for 1-D arrays where it concatenates along the first axis.

    Parameters
    ----------
    tup : sequence of arrays
        The arrays must have the same shape along all but the second axis,
        except 1-D arrays which can be any length.

    Returns
    -------
    stacked : array
        The array formed by stacking the given arrays.
    """
    arrs = [mx.array(a) for a in tup]

    # If all arrays are 1-D, concatenate along axis 0
    if all(len(a.shape) == 1 for a in arrs):
        return mx.concatenate(arrs, axis=0)

    # Otherwise concatenate along axis 1
    # Ensure at least 2D for consistency
    arrs = [mx.atleast_2d(a) for a in arrs]
    return mx.concatenate(arrs, axis=1)


# Add these functions to mx namespace for convenience
mx.place = place
mx.extract = extract
mx.sinc = sinc
mx.isscalar = isscalar
mx.hstack = hstack
