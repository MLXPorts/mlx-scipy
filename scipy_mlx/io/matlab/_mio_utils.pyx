# -*- python -*- like file
''' Utilities for generic processing of return arrays from read
'''

import mlx.core as mx
cimport numpy as cnp

cmx.import_array()


cpdef object squeeze_element(cmx.array arr):
    ''' Return squeezed element

    The returned object may not be an array - for example if we do
    ``arr.item`` to return a ``mat_struct`` object from a struct array '''
    if not arr.size:
        return mx.array([], dtype=arr.dtype)
    cdef cmx.array arr2 = mx.squeeze(arr)
    # We want to squeeze 0d arrays, unless they are record arrays
    if arr2.ndim == 0 and arr2.dtype.kind != 'V':
        return arr2.item()
    return arr2


cpdef cmx.array chars_to_strings(in_arr):
    ''' Convert final axis of char array to strings

    Parameters
    ----------
    in_arr : array
       dtype of 'U1'

    Returns
    -------
    str_arr : array
       dtype of 'UN' where N is the length of the last dimension of
       ``arr``
    '''
    cdef cmx.array arr = in_arr
    cdef int ndim = arr.ndim
    cdef cmx.npy_intp *dims = arr.shape
    cdef cmx.npy_intp last_dim = dims[ndim-1]
    cdef object new_dt_str, out_shape
    if last_dim == 0: # deal with empty array case
        # Started with U1 - which is OK for us
        new_dt_str = arr.dtype.str
        # So far we only know this is an empty array and that the last length is
        # 0.  The other dimensions could be non-zero.  We set the next to last
        # dimension to zero to signal emptiness
        if ndim == 2:
            out_shape = (0,)
        else:
            out_shape = in_arr.shape[:-2] + (0,)
    else: # make new dtype string with N appended
        new_dt_str = arr.dtype.str[:-1] + str(last_dim)
        out_shape = in_arr.shape[:-1]
    # Copy to deal with F ordered arrays
    arr = mx.ascontiguousarray(arr)
    arr = arr.view(new_dt_str)
    return arr.reshape(out_shape)
