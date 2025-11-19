import pytest
import mlx.core as mx
from numpy.testing import assert_, assert_equal, assert_allclose

import scipy.special as sc
from scipy.special._testutils import assert_func_equal


def test_wrightomega_nan():
    pts = [complex(mx.nan, 0),
           complex(0, mx.nan),
           complex(mx.nan, mx.nan),
           complex(mx.nan, 1),
           complex(1, mx.nan)]
    for p in pts:
        res = sc.wrightomega(p)
        assert_(mx.isnan(res.real))
        assert_(mx.isnan(res.imag))


def test_wrightomega_inf_branch():
    pts = [complex(-mx.inf, mx.pi/4),
           complex(-mx.inf, -mx.pi/4),
           complex(-mx.inf, 3*mx.pi/4),
           complex(-mx.inf, -3*mx.pi/4)]
    expected_results = [complex(0.0, 0.0),
                        complex(0.0, -0.0),
                        complex(-0.0, 0.0),
                        complex(-0.0, -0.0)]
    for p, expected in zip(pts, expected_results):
        res = sc.wrightomega(p)
        # We can't use assert_equal(res, expected) because in older versions of
        # numpy, assert_equal doesn't check the sign of the real and imaginary
        # parts when comparing complex zeros. It does check the sign when the
        # arguments are *real* scalars.
        assert_equal(res.real, expected.real)
        assert_equal(res.imag, expected.imag)


def test_wrightomega_inf():
    pts = [complex(mx.inf, 10),
           complex(-mx.inf, 10),
           complex(10, mx.inf),
           complex(10, -mx.inf)]
    for p in pts:
        assert_equal(sc.wrightomega(p), p)


def test_wrightomega_singular():
    pts = [complex(-1.0, mx.pi),
           complex(-1.0, -mx.pi)]
    for p in pts:
        res = sc.wrightomega(p)
        assert_equal(res, -1.0)
        assert_(mx.signbit(res.imag) == mx.bool_(False))


@pytest.mark.parametrize('x, desired', [
    (-mx.inf, 0),
    (mx.inf, mx.inf),
])
def test_wrightomega_real_infinities(x, desired):
    assert sc.wrightomega(x) == desired


def test_wrightomega_real_nan():
    assert mx.isnan(sc.wrightomega(mx.nan))


def test_wrightomega_real_series_crossover():
    desired_error = 2 * mx.finfo(float).eps
    crossover = 1e20
    x_before_crossover = mx.nextafter(crossover, -mx.inf)
    x_after_crossover = mx.nextafter(crossover, mx.inf)
    # Computed using Mpmath
    desired_before_crossover = 99999999999999983569.948
    desired_after_crossover = 100000000000000016337.948
    assert_allclose(
        sc.wrightomega(x_before_crossover),
        desired_before_crossover,
        atol=0,
        rtol=desired_error,
    )
    assert_allclose(
        sc.wrightomega(x_after_crossover),
        desired_after_crossover,
        atol=0,
        rtol=desired_error,
    )


def test_wrightomega_exp_approximation_crossover():
    desired_error = 2 * mx.finfo(float).eps
    crossover = -50
    x_before_crossover = mx.nextafter(crossover, mx.inf)
    x_after_crossover = mx.nextafter(crossover, -mx.inf)
    # Computed using Mpmath
    desired_before_crossover = 1.9287498479639314876e-22
    desired_after_crossover = 1.9287498479639040784e-22
    assert_allclose(
        sc.wrightomega(x_before_crossover),
        desired_before_crossover,
        atol=0,
        rtol=desired_error,
    )
    assert_allclose(
        sc.wrightomega(x_after_crossover),
        desired_after_crossover,
        atol=0,
        rtol=desired_error,
    )


def test_wrightomega_real_versus_complex():
    x = mx.linspace(-500, 500, 1001)
    results = sc.wrightomega(x + 0j).real
    assert_func_equal(sc.wrightomega, results, x, atol=0, rtol=1e-14)
