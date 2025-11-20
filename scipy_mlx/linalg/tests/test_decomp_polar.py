import pytest
import mlx.core as mx
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose, assert_equal)
from scipy_mlx.linalg import polar, eigh


diag2 = mx.array([[2, 0], [0, 3]])
a13 = mx.array([[1, 2, 2]])

precomputed_cases = [
    [[[0]], 'right', [[1]], [[0]]],
    [[[0]], 'left', [[1]], [[0]]],
    [[[9]], 'right', [[1]], [[9]]],
    [[[9]], 'left', [[1]], [[9]]],
    [diag2, 'right', mx.eye(2), diag2],
    [diag2, 'left', mx.eye(2), diag2],
    [a13, 'right', a13/norm(a13[0]), a13.T.dot(a13)/norm(a13[0])],
]

verify_cases = [
    [[1, 2], [3, 4]],
    [[1, 2, 3]],
    [[1], [2], [3]],
    [[1, 2, 3], [3, 4, 0]],
    [[1, 2], [3, 4], [5, 5]],
    [[1, 2], [3, 4+5j]],
    [[1, 2, 3j]],
    [[1], [2], [3j]],
    [[1, 2, 3+2j], [3, 4-1j, -4j]],
    [[1, 2], [3-2j, 4+0.5j], [5, 5]],
    [[10000, 10, 1], [-1, 2, 3j], [0, 1, 2]],
    mx.empty((0, 0)),
    mx.empty((0, 2)),
    mx.empty((2, 0)),
]


def check_precomputed_polar(a, side, expected_u, expected_p):
    # Compare the result of the polar decomposition to a
    # precomputed result.
    u, p = polar(a, side=side)
    assert_allclose(u, expected_u, atol=1e-15)
    assert_allclose(p, expected_p, atol=1e-15)


def verify_polar(a):
    # Compute the polar decomposition, and then verify that
    # the result has all the expected properties.
    product_atol = mx.sqrt(mx.finfo(float).eps)

    aa = mx.array(a)
    m, n = aa.shape

    u, p = polar(a, side='right')
    assert_equal(u.shape, (m, n))
    assert_equal(p.shape, (n, n))
    # a = up
    assert_allclose(u.dot(p), a, atol=product_atol)
    if m >= n:
        assert_allclose(u.conj().T.dot(u), mx.eye(n), atol=1e-15)
    else:
        assert_allclose(u.dot(u.conj().T), mx.eye(m), atol=1e-15)
    # p is Hermitian positive semidefinite.
    assert_allclose(p.conj().T, p)
    evals = eigh(p, eigvals_only=True)
    nonzero_evals = evals[abs(evals) > 1e-14]
    assert_((nonzero_evals >= 0).all())

    u, p = polar(a, side='left')
    assert_equal(u.shape, (m, n))
    assert_equal(p.shape, (m, m))
    # a = pu
    assert_allclose(p.dot(u), a, atol=product_atol)
    if m >= n:
        assert_allclose(u.conj().T.dot(u), mx.eye(n), atol=1e-15)
    else:
        assert_allclose(u.dot(u.conj().T), mx.eye(m), atol=1e-15)
    # p is Hermitian positive semidefinite.
    assert_allclose(p.conj().T, p)
    evals = eigh(p, eigvals_only=True)
    nonzero_evals = evals[abs(evals) > 1e-14]
    assert_((nonzero_evals >= 0).all())


def test_precomputed_cases():
    for a, side, expected_u, expected_p in precomputed_cases:
        check_precomputed_polar(a, side, expected_u, expected_p)


def test_verify_cases():
    for a in verify_cases:
        verify_polar(a)

@pytest.mark.parametrize('dt', [int, float, mx.float32, complex, mx.complex64])
@pytest.mark.parametrize('shape',  [(0, 0), (0, 2), (2, 0)])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_empty(dt, shape, side):
    a = mx.empty(shape, dtype=dt)
    m, n = shape
    p_shape = (m, m) if side == 'left' else (n, n)

    u, p = polar(a, side=side)
    u_n, p_n = polar(mx.eye(5, dtype=dt))

    assert_equal(u.dtype, u_n.dtype)
    assert_equal(p.dtype, p_n.dtype)
    assert u.shape == shape
    assert p.shape == p_shape
    assert mx.all(p == 0)
