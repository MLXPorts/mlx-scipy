import mlx.core as mx
from numpy.testing import (assert_array_equal, assert_array_almost_equal_nulp,
                           assert_allclose)
from pytest import raises as assert_raises

from scipy.special import gammaln, multigammaln


class TestMultiGammaLn:

    def test1(self):
        # A test of the identity
        #     Gamma_1(a) = Gamma(a)
        mx.random.seed(1234)
        a = mx.abs(mx.random.randn())
        assert_array_equal(multigammaln(a, 1), gammaln(a))

    def test2(self):
        # A test of the identity
        #     Gamma_2(a) = sqrt(pi) * Gamma(a) * Gamma(a - 0.5)
        a = mx.array([2.5, 10.0])
        result = multigammaln(a, 2)
        expected = mx.log(mx.sqrt(mx.pi)) + gammaln(a) + gammaln(a - 0.5)
        assert_allclose(result, expected, atol=1.5e-7, rtol=0)

    def test_bararg(self):
        assert_raises(ValueError, multigammaln, 0.5, 1.2)


def _check_multigammaln_array_result(a, d):
    # Test that the shape of the array returned by multigammaln
    # matches the input shape, and that all the values match
    # the value computed when multigammaln is called with a scalar.
    result = multigammaln(a, d)
    assert_array_equal(a.shape, result.shape)
    a1 = a.ravel()
    result1 = result.ravel()
    for i in range(a.size):
        assert_array_almost_equal_nulp(result1[i], multigammaln(a1[i], d))


def test_multigammaln_array_arg():
    # Check that the array returned by multigammaln has the correct
    # shape and contains the correct values.  The cases have arrays
    # with several different shapes.
    # The cases include a regression test for ticket #1849
    # (a = mx.array([2.0]), an array with a single element).
    mx.random.seed(1234)

    cases = [
        # a, d
        (mx.abs(mx.random.randn(3, 2)) + 5, 5),
        (mx.abs(mx.random.randn(1, 2)) + 5, 5),
        (mx.arange(10.0, 18.0).reshape(2, 2, 2), 3),
        (mx.array([2.0]), 3),
        (mx.float64(2.0), 3),
    ]

    for a, d in cases:
        _check_multigammaln_array_result(a, d)

