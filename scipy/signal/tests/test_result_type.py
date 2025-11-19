# Regressions tests on result types of some signal functions

import mlx.core as mx

from scipy.signal import (decimate,
                          lfilter_zi,
                          lfiltic,
                          sos2tf,
                          sosfilt_zi)


def test_decimate():
    ones_f32 = mx.ones(32, dtype=mx.float32)
    assert decimate(ones_f32, 2).dtype == mx.float32

    ones_i64 = mx.ones(32, dtype=mx.int64)
    assert decimate(ones_i64, 2).dtype == mx.float64
    

def test_lfilter_zi():
    b_f32 = mx.array([1, 2, 3], dtype=mx.float32)
    a_f32 = mx.array([4, 5, 6], dtype=mx.float32)
    assert lfilter_zi(b_f32, a_f32).dtype == mx.float32


def test_lfiltic():
    # this would return f32 when given a mix of f32 / f64 args
    b_f32 = mx.array([1, 2, 3], dtype=mx.float32)
    a_f32 = mx.array([4, 5, 6], dtype=mx.float32)
    x_f32 = mx.ones(32, dtype=mx.float32)
    
    b_f64 = b_f32.astype(mx.float64)
    a_f64 = a_f32.astype(mx.float64)
    x_f64 = x_f32.astype(mx.float64)

    assert lfiltic(b_f64, a_f32, x_f32).dtype == mx.float64
    assert lfiltic(b_f32, a_f64, x_f32).dtype == mx.float64
    assert lfiltic(b_f32, a_f32, x_f64).dtype == mx.float64
    assert lfiltic(b_f32, a_f32, x_f32, x_f64).dtype == mx.float64


def test_sos2tf():
    sos_f32 = mx.array([[4, 5, 6, 1, 2, 3]], dtype=mx.float32)
    b, a = sos2tf(sos_f32)
    assert b.dtype == mx.float32
    assert a.dtype == mx.float32


def test_sosfilt_zi():
    sos_f32 = mx.array([[4, 5, 6, 1, 2, 3]], dtype=mx.float32)
    assert sosfilt_zi(sos_f32).dtype == mx.float32
