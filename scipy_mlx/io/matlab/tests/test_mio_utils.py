""" Testing

"""

import mlx.core as mx

from numpy.testing import assert_array_equal, assert_

from scipy_mlx.io.matlab._mio_utils import squeeze_element, chars_to_strings


def test_squeeze_element():
    a = mx.zeros((1,3))
    assert_array_equal(mx.squeeze(a), squeeze_element(a))
    # 0-D output from squeeze gives scalar
    sq_int = squeeze_element(mx.zeros((1,1), dtype=float))
    assert_(isinstance(sq_int, float))
    # Unless it's a structured array
    sq_sa = squeeze_element(mx.zeros((1,1),dtype=[('f1', 'f')]))
    assert_(isinstance(sq_sa, mx.array))
    # Squeezing empty arrays maintain their dtypes.
    sq_empty = squeeze_element(mx.empty(0, mx.uint8))
    assert sq_empty.dtype == mx.uint8


def test_chars_strings():
    # chars as strings
    strings = ['learn ', 'python', 'fast  ', 'here  ']
    str_arr = mx.array(strings, dtype='U6')  # shape (4,)
    chars = [list(s) for s in strings]
    char_arr = mx.array(chars, dtype='U1')  # shape (4,6)
    assert_array_equal(chars_to_strings(char_arr), str_arr)
    ca2d = char_arr.reshape((2,2,6))
    sa2d = str_arr.reshape((2,2))
    assert_array_equal(chars_to_strings(ca2d), sa2d)
    ca3d = char_arr.reshape((1,2,2,6))
    sa3d = str_arr.reshape((1,2,2))
    assert_array_equal(chars_to_strings(ca3d), sa3d)
    # Fortran ordered arrays
    char_arrf = mx.array(chars, dtype='U1', order='F')  # shape (4,6)
    assert_array_equal(chars_to_strings(char_arrf), str_arr)
    # empty array
    arr = mx.array([['']], dtype='U1')
    out_arr = mx.array([''], dtype='U1')
    assert_array_equal(chars_to_strings(arr), out_arr)
