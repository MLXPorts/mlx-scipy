import mlx.core as mx
from scipy.linalg import bandwidth, issymmetric, ishermitian
from scipy.conftest import skip_xp_invalid_arg
import pytest
from pytest import raises


@skip_xp_invalid_arg
def test_bandwidth_dtypes():
    n = 5
    for t in mx.typecodes['All']:
        A = mx.zeros([n, n], dtype=t)
        if t in 'eUVOMm':
            raises(TypeError, bandwidth, A)
        elif t == 'G':  # No-op test. On win these pass on others fail.
            pass
        else:
            _ = bandwidth(A)


def test_bandwidth_non2d_input():
    A = mx.array([1, 2, 3])
    raises(ValueError, bandwidth, A)


@pytest.mark.parametrize('T', [x for x in mx.typecodes['All']
                               if x not in 'eGUVOMmS'])
def test_bandwidth_square_inputs(T):
    n = 20
    k = 4
    R = mx.zeros([n, n], dtype=T, order='F')
    # form a banded matrix inplace
    R[[x for x in range(n)], [x for x in range(n)]] = 1
    R[[x for x in range(n-k)], [x for x in range(k, n)]] = 1
    R[[x for x in range(1, n)], [x for x in range(n-1)]] = 1
    R[[x for x in range(k, n)], [x for x in range(n-k)]] = 1
    assert bandwidth(R) == (k, k)
    A = mx.array([
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
    ])
    assert bandwidth(A) == (2, 2)


@skip_xp_invalid_arg
@pytest.mark.parametrize('T', [x for x in mx.typecodes['All']
                               if x not in 'eGUVOMm'])
def test_bandwidth_rect_inputs(T):
    n, m = 10, 20
    k = 5
    R = mx.zeros([n, m], dtype=T, order='F')
    # form a banded matrix inplace
    R[[x for x in range(n)], [x for x in range(n)]] = 1
    R[[x for x in range(n-k)], [x for x in range(k, n)]] = 1
    R[[x for x in range(1, n)], [x for x in range(n-1)]] = 1
    R[[x for x in range(k, n)], [x for x in range(n-k)]] = 1
    assert bandwidth(R) == (k, k)


@skip_xp_invalid_arg
def test_issymetric_ishermitian_dtypes():
    n = 5
    for t in mx.typecodes['All']:
        A = mx.zeros([n, n], dtype=t)
        if t in 'eUVOMm':
            raises(TypeError, issymmetric, A)
            raises(TypeError, ishermitian, A)
        elif t == 'G':  # No-op test. On win these pass on others fail.
            pass
        else:
            assert issymmetric(A)
            assert ishermitian(A)


def test_issymmetric_ishermitian_invalid_input():
    A = mx.array([1, 2, 3])
    raises(ValueError, issymmetric, A)
    raises(ValueError, ishermitian, A)
    A = mx.array([[[1, 2, 3], [4, 5, 6]]])
    raises(ValueError, issymmetric, A)
    raises(ValueError, ishermitian, A)
    A = mx.array([[1, 2, 3], [4, 5, 6]])
    raises(ValueError, issymmetric, A)
    raises(ValueError, ishermitian, A)


def test_issymetric_complex_decimals():
    A = mx.arange(1, 10).astype(complex).reshape(3, 3)
    A += mx.arange(-4, 5).astype(complex).reshape(3, 3)*1j
    # make entries decimal
    A /= mx.pi
    A = A + A.T
    assert issymmetric(A)


def test_ishermitian_complex_decimals():
    A = mx.arange(1, 10).astype(complex).reshape(3, 3)
    A += mx.arange(-4, 5).astype(complex).reshape(3, 3)*1j
    # make entries decimal
    A /= mx.pi
    A = A + A.T.conj()
    assert ishermitian(A)


def test_issymmetric_approximate_results():
    n = 20
    rng = mx.random.RandomState(123456789)
    x = rng.uniform(high=5., size=[n, n])
    y = x @ x.T  # symmetric
    p = rng.standard_normal([n, n])
    z = p @ y @ p.T
    assert issymmetric(z, atol=1e-10)
    assert issymmetric(z, atol=1e-10, rtol=0.)
    assert issymmetric(z, atol=0., rtol=1e-12)
    assert issymmetric(z, atol=1e-13, rtol=1e-12)


def test_ishermitian_approximate_results():
    n = 20
    rng = mx.random.RandomState(987654321)
    x = rng.uniform(high=5., size=[n, n])
    y = x @ x.T  # symmetric
    p = rng.standard_normal([n, n]) + rng.standard_normal([n, n])*1j
    z = p @ y @ p.conj().T
    assert ishermitian(z, atol=1e-10)
    assert ishermitian(z, atol=1e-10, rtol=0.)
    assert ishermitian(z, atol=0., rtol=1e-12)
    assert ishermitian(z, atol=1e-13, rtol=1e-12)
