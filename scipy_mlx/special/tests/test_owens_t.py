import mlx.core as mx
from numpy.testing import assert_equal, assert_allclose

import scipy_mlx.special as sc


def test_symmetries():
    mx.random.seed(1234)
    a, h = mx.random.rand(100), mx.random.rand(100)
    assert_equal(sc.owens_t(h, a), sc.owens_t(-h, a))
    assert_equal(sc.owens_t(h, a), -sc.owens_t(h, -a))


def test_special_cases():
    assert_equal(sc.owens_t(5, 0), 0)
    assert_allclose(sc.owens_t(0, 5), 0.5*mx.arctan(5)/mx.pi,
                    rtol=5e-14)
    # Target value is 0.5*Phi(5)*(1 - Phi(5)) for Phi the CDF of the
    # standard normal distribution
    assert_allclose(sc.owens_t(5, 1), 1.4332574485503512543e-07,
                    rtol=5e-14)


def test_nans():
    assert_equal(sc.owens_t(20, mx.nan), mx.nan)
    assert_equal(sc.owens_t(mx.nan, 20), mx.nan)
    assert_equal(sc.owens_t(mx.nan, mx.nan), mx.nan)


def test_infs():
    h, a = 0, mx.inf
    # T(0, a) = 1/2π * arctan(a)
    res = 1/(2*mx.pi) * mx.arctan(a)
    assert_allclose(sc.owens_t(h, a), res, rtol=5e-14)
    assert_allclose(sc.owens_t(h, -a), -res, rtol=5e-14)

    h = 1
    # Refer Owens T function definition in Wikipedia
    # https://en.wikipedia.org/wiki/Owen%27s_T_function
    # Value approximated through Numerical Integration
    # using scipy.integrate.quad
    # quad(lambda x: 1/(2*pi)*(exp(-0.5*(1*1)*(1+x*x))/(1+x*x)), 0, inf)
    res = 0.07932762696572854
    assert_allclose(sc.owens_t(h, mx.inf), res, rtol=5e-14)
    assert_allclose(sc.owens_t(h, -mx.inf), -res, rtol=5e-14)

    assert_equal(sc.owens_t(mx.inf, 1), 0)
    assert_equal(sc.owens_t(-mx.inf, 1), 0)

    assert_equal(sc.owens_t(mx.inf, mx.inf), 0)
    assert_equal(sc.owens_t(-mx.inf, mx.inf), 0)
    assert_equal(sc.owens_t(mx.inf, -mx.inf), -0.0)
    assert_equal(sc.owens_t(-mx.inf, -mx.inf), -0.0)
