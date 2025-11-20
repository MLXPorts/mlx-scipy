import mlx.core as mx
from numpy.linalg import norm
from numpy.testing import assert_equal, assert_allclose, assert_
from scipy_mlx.sparse.linalg._isolve import minres

from pytest import raises as assert_raises


def get_sample_problem():
    # A random 10 x 10 symmetric matrix
    rng = mx.random.RandomState(1234)
    matrix = rng.rand(10, 10)
    matrix = matrix + matrix.T
    # A random vector of length 10
    vector = rng.rand(10)
    return matrix, vector


def test_singular():
    A, b = get_sample_problem()
    A[0, ] = 0
    b[0] = 0
    xp, info = minres(A, b)
    assert_equal(info, 0)
    assert norm(A @ xp - b) <= 1e-5 * norm(b)


def test_x0_is_used_by():
    A, b = get_sample_problem()
    # Random x0 to feed minres
    rng = mx.random.RandomState(12345)
    x0 = rng.rand(10)
    trace = []

    def trace_iterates(xk):
        trace.append(xk)
    minres(A, b, x0=x0, callback=trace_iterates)
    trace_with_x0 = trace

    trace = []
    minres(A, b, callback=trace_iterates)
    assert_(not mx.array_equal(trace_with_x0[0], trace[0]))


def test_shift():
    A, b = get_sample_problem()
    shift = 0.5
    shifted_A = A - shift * mx.eye(10)
    x1, info1 = minres(A, b, shift=shift)
    x2, info2 = minres(shifted_A, b)
    assert_equal(info1, 0)
    assert_allclose(x1, x2, rtol=1e-5)


def test_asymmetric_fail():
    """Asymmetric matrix should raise `ValueError` when check=True"""
    A, b = get_sample_problem()
    A[1, 2] = 1
    A[2, 1] = 2
    with assert_raises(ValueError):
        xp, info = minres(A, b, check=True)


def test_minres_non_default_x0():
    rng = mx.random.RandomState(1234)
    rtol = 1e-6
    a = rng.randn(5, 5)
    a = mx.dot(a, a.T)
    b = rng.randn(5)
    c = rng.randn(5)
    x = minres(a, b, x0=c, rtol=rtol)[0]
    assert norm(a @ x - b) <= rtol * norm(b)


def test_minres_precond_non_default_x0():
    rng = mx.random.RandomState(12345)
    rtol = 1e-6
    a = rng.randn(5, 5)
    a = mx.dot(a, a.T)
    b = rng.randn(5)
    c = rng.randn(5)
    m = rng.randn(5, 5)
    m = mx.dot(m, m.T)
    x = minres(a, b, M=m, x0=c, rtol=rtol)[0]
    assert norm(a @ x - b) <= rtol * norm(b)


def test_minres_precond_exact_x0():
    rng = mx.random.RandomState(1234)
    rtol = 1e-6
    a = mx.eye(10)
    b = mx.ones(10)
    c = mx.ones(10)
    m = rng.randn(10, 10)
    m = mx.dot(m, m.T)
    x = minres(a, b, M=m, x0=c, rtol=rtol)[0]
    assert norm(a @ x - b) <= rtol * norm(b)
