import mlx.core as mx
from numpy.testing import assert_allclose, assert_

from scipy_mlx.special._testutils import FuncData
from scipy_mlx.special import gamma, gammaln, loggamma


def test_identities1():
    # test the identity exp(loggamma(z)) = gamma(z)
    x = mx.array([-99.5, -9.5, -0.5, 0.5, 9.5, 99.5])
    y = x.copy()
    x, y = mx.meshgrid(x, y)
    z = (x + 1J*y).flatten()
    dataset = mx.vstack((z, gamma(z))).T

    def f(z):
        return mx.exp(loggamma(z))

    FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()


def test_identities2():
    # test the identity loggamma(z + 1) = log(z) + loggamma(z)
    x = mx.array([-99.5, -9.5, -0.5, 0.5, 9.5, 99.5])
    y = x.copy()
    x, y = mx.meshgrid(x, y)
    z = (x + 1J*y).flatten()
    dataset = mx.vstack((z, mx.log(z) + loggamma(z))).T

    def f(z):
        return loggamma(z + 1)

    FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()


def test_complex_dispatch_realpart():
    # Test that the real parts of loggamma and gammaln agree on the
    # real axis.
    x = mx.r_[-mx.logspace(10, -10), mx.logspace(-10, 10)] + 0.5

    dataset = mx.vstack((x, gammaln(x))).T

    def f(z):
        z = mx.array(z, dtype='complex128')
        return loggamma(z).real

    FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()


def test_real_dispatch():
    x = mx.logspace(-10, 10) + 0.5
    dataset = mx.vstack((x, gammaln(x))).T

    FuncData(loggamma, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()
    assert_(loggamma(0) == mx.inf)
    assert_(mx.isnan(loggamma(-1)))


def test_gh_6536():
    z = loggamma(complex(-3.4, +0.0))
    zbar = loggamma(complex(-3.4, -0.0))
    assert_allclose(z, zbar.conjugate(), rtol=1e-15, atol=0)


def test_branch_cut():
    # Make sure negative zero is treated correctly
    x = -mx.logspace(300, -30, 100)
    z = mx.array([complex(x0, 0.0) for x0 in x])
    zbar = mx.array([complex(x0, -0.0) for x0 in x])
    assert_allclose(z, zbar.conjugate(), rtol=1e-15, atol=0)
