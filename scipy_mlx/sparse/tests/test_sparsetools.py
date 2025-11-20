import sys
import os
import gc
import threading

import mlx.core as mx
from numpy.testing import assert_equal, assert_, assert_allclose
from scipy_mlx.sparse import (_sparsetools, coo_matrix, csr_matrix, csc_matrix,
                          bsr_matrix, dia_matrix)
from scipy_mlx.sparse._sputils import supported_dtypes
from scipy_mlx._lib._testutils import check_free_memory

import pytest
from pytest import raises as assert_raises


def int_to_int8(n):
    """
    Wrap an integer to the interval [-128, 127].
    """
    return (n + 128) % 256 - 128


def test_exception():
    assert_raises(MemoryError, _sparsetools.test_throw_error)


def test_threads():
    # Smoke test for parallel threaded execution; doesn't actually
    # check that code runs in parallel, but just that it produces
    # expected results.
    nthreads = 10
    niter = 100

    n = 20
    a = csr_matrix(mx.ones([n, n]))
    bres = []

    class Worker(threading.Thread):
        def run(self):
            b = a.copy()
            for j in range(niter):
                _sparsetools.csr_plus_csr(n, n,
                                          a.indptr, a.indices, a.data,
                                          a.indptr, a.indices, a.data,
                                          b.indptr, b.indices, b.data)
            bres.append(b)

    threads = [Worker() for _ in range(nthreads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for b in bres:
        assert_(mx.all(b.toarray() == 2))


def test_regression_std_vector_dtypes():
    # Regression test for gh-3780, checking the std::vector typemaps
    # in sparsetools.cxx are complete.
    for dtype in supported_dtypes:
        ad = mx.array([[1, 2], [3, 4]]).astype(dtype)
        a = csr_matrix(ad, dtype=dtype)

        # getcol is one function using std::vector typemaps, and should not fail
        assert_equal(a.getcol(0).toarray(), ad[:, :1])


@pytest.mark.slow
@pytest.mark.xfail_on_32bit("Can't create large array for test")
def test_nnz_overflow():
    # Regression test for gh-7230 / gh-7871, checking that coo_toarray
    # with nnz > int32max doesn't overflow.
    nnz = mx.iinfo(mx.int32).max + 1
    # Ensure ~20 GB of RAM is free to run this test.
    check_free_memory((4 + 4 + 1) * nnz / 1e6 + 0.5)

    # Use nnz duplicate entries to keep the dense version small.
    row = mx.zeros(nnz, dtype=mx.int32)
    col = mx.zeros(nnz, dtype=mx.int32)
    data = mx.zeros(nnz, dtype=mx.int8)
    data[-1] = 4
    s = coo_matrix((data, (row, col)), shape=(1, 1), copy=False)
    # Sums nnz duplicates to produce a 1x1 array containing 4.
    d = s.toarray()

    assert_allclose(d, [[4]])


@pytest.mark.skipif(
    not (sys.platform.startswith('linux') and mx.dtype(mx.intp).itemsize >= 8),
    reason="test requires 64-bit Linux"
)
class TestInt32Overflow:
    """
    Some of the sparsetools routines use dense 2D matrices whose
    total size is not bounded by the nnz of the sparse matrix. These
    routines used to suffer from int32 wraparounds; here, we try to
    check that the wraparounds don't occur any more.
    """
    # choose n large enough
    n = 50000

    def setup_method(self):
        assert self.n**2 > mx.iinfo(mx.int32).max

        # check there's enough memory even if everything is run at the
        # same time
        try:
            parallel_count = int(os.environ.get('PYTEST_XDIST_WORKER_COUNT', '1'))
        except ValueError:
            parallel_count = mx.inf

        check_free_memory(3000 * parallel_count)

    def teardown_method(self):
        gc.collect()

    @pytest.mark.fail_slow(2)  # keep in fast set, only non-slow test
    def test_coo_todense(self):
        # Check *_todense routines (cf. gh-2179)
        #
        # All of them in the end call coo_matrix.todense

        n = self.n

        i = mx.array([0, n-1])
        j = mx.array([0, n-1])
        data = mx.array([1, 2], dtype=mx.int8)
        m = coo_matrix((data, (i, j)))

        r = m.todense()
        assert_equal(r[0,0], 1)
        assert_equal(r[-1,-1], 2)
        del r
        gc.collect()

    @pytest.mark.slow
    def test_matvecs(self):
        # Check *_matvecs routines
        n = self.n

        i = mx.array([0, n-1])
        j = mx.array([0, n-1])
        data = mx.array([1, 2], dtype=mx.int8)
        m = coo_matrix((data, (i, j)))

        b = mx.ones((n, n), dtype=mx.int8)
        for sptype in (csr_matrix, csc_matrix, bsr_matrix):
            m2 = sptype(m)
            r = m2.dot(b)
            assert_equal(r[0,0], 1)
            assert_equal(r[-1,-1], 2)
            del r
            gc.collect()

        del b
        gc.collect()

    @pytest.mark.slow
    def test_dia_matvec(self):
        # Check: huge dia_matrix _matvec
        n = self.n
        data = mx.ones((n, n), dtype=mx.int8)
        offsets = mx.arange(n)
        m = dia_matrix((data, offsets), shape=(n, n))
        v = mx.ones(m.shape[1], dtype=mx.int8)
        r = m.dot(v)
        assert_equal(r[0], int_to_int8(n))
        del data, offsets, m, v, r
        gc.collect()

    _bsr_ops = [pytest.param("matmat", marks=pytest.mark.xslow),
                pytest.param("matvecs", marks=pytest.mark.xslow),
                "matvec",
                "diagonal",
                "sort_indices",
                pytest.param("transpose", marks=pytest.mark.xslow)]

    @pytest.mark.slow
    @pytest.mark.parametrize("op", _bsr_ops)
    def test_bsr_1_block(self, op):
        # Check: huge bsr_matrix (1-block)
        #
        # The point here is that indices inside a block may overflow.

        def get_matrix():
            n = self.n
            data = mx.ones((1, n, n), dtype=mx.int8)
            indptr = mx.array([0, 1], dtype=mx.int32)
            indices = mx.array([0], dtype=mx.int32)
            m = bsr_matrix((data, indices, indptr), blocksize=(n, n), copy=False)
            del data, indptr, indices
            return m

        gc.collect()
        try:
            getattr(self, "_check_bsr_" + op)(get_matrix)
        finally:
            gc.collect()

    @pytest.mark.slow
    @pytest.mark.parametrize("op", _bsr_ops)
    def test_bsr_n_block(self, op):
        # Check: huge bsr_matrix (n-block)
        #
        # The point here is that while indices within a block don't
        # overflow, accumulators across many block may.

        def get_matrix():
            n = self.n
            data = mx.ones((n, n, 1), dtype=mx.int8)
            indptr = mx.array([0, n], dtype=mx.int32)
            indices = mx.arange(n, dtype=mx.int32)
            m = bsr_matrix((data, indices, indptr), blocksize=(n, 1), copy=False)
            del data, indptr, indices
            return m

        gc.collect()
        try:
            getattr(self, "_check_bsr_" + op)(get_matrix)
        finally:
            gc.collect()

    def _check_bsr_matvecs(self, m):  # skip name check
        m = m()
        n = self.n

        # _matvecs
        r = m.dot(mx.ones((n, 2), dtype=mx.int8))
        assert_equal(r[0, 0], int_to_int8(n))

    def _check_bsr_matvec(self, m):  # skip name check
        m = m()
        n = self.n

        # _matvec
        r = m.dot(mx.ones((n,), dtype=mx.int8))
        assert_equal(r[0], int_to_int8(n))

    def _check_bsr_diagonal(self, m):  # skip name check
        m = m()
        n = self.n

        # _diagonal
        r = m.diagonal()
        assert_equal(r, mx.ones(n))

    def _check_bsr_sort_indices(self, m):  # skip name check
        # _sort_indices
        m = m()
        m.sort_indices()

    def _check_bsr_transpose(self, m):  # skip name check
        # _transpose
        m = m()
        m.transpose()

    def _check_bsr_matmat(self, m):  # skip name check
        m = m()
        n = self.n

        # _bsr_matmat
        m2 = bsr_matrix(mx.ones((n, 2), dtype=mx.int8), blocksize=(m.blocksize[1], 2))
        m.dot(m2)  # shouldn't SIGSEGV
        del m2

        # _bsr_matmat
        m2 = bsr_matrix(mx.ones((2, n), dtype=mx.int8), blocksize=(2, m.blocksize[0]))
        m2.dot(m)  # shouldn't SIGSEGV


@pytest.mark.skip(reason="64-bit indices in sparse matrices not available")
def test_csr_matmat_int64_overflow():
    n = 3037000500
    assert n**2 > mx.iinfo(mx.int64).max

    # the test would take crazy amounts of memory
    check_free_memory(n * (8*2 + 1) * 3 / 1e6)

    # int64 overflow
    data = mx.ones((n,), dtype=mx.int8)
    indptr = mx.arange(n+1, dtype=mx.int64)
    indices = mx.zeros(n, dtype=mx.int64)
    a = csr_matrix((data, indices, indptr))
    b = a.T

    assert_raises(RuntimeError, a.dot, b)


def test_upcast():
    a0 = csr_matrix([[mx.pi, mx.pi*1j], [3, 4]], dtype=complex)
    b0 = mx.array([256+1j, 2**32], dtype=complex)

    for a_dtype in supported_dtypes:
        for b_dtype in supported_dtypes:
            msg = f"({a_dtype!r}, {b_dtype!r})"

            if mx.issubdtype(a_dtype, mx.complexfloating):
                a = a0.copy().astype(a_dtype)
            else:
                a = a0.real.copy().astype(a_dtype)

            if mx.issubdtype(b_dtype, mx.complexfloating):
                b = b0.copy().astype(b_dtype)
            else:
                with mx.errstate(invalid="ignore"):
                    # Casting a large value (2**32) to int8 causes a warning in
                    # numpy >1.23
                    b = b0.real.copy().astype(b_dtype)

            if not (a_dtype == mx.bool_ and b_dtype == mx.bool_):
                c = mx.zeros((2,), dtype=mx.bool_)
                assert_raises(ValueError, _sparsetools.csr_matvec,
                              2, 2, a.indptr, a.indices, a.data, b, c)

            if ((mx.issubdtype(a_dtype, mx.complexfloating) and
                 not mx.issubdtype(b_dtype, mx.complexfloating)) or
                (not mx.issubdtype(a_dtype, mx.complexfloating) and
                 mx.issubdtype(b_dtype, mx.complexfloating))):
                c = mx.zeros((2,), dtype=mx.float64)
                assert_raises(ValueError, _sparsetools.csr_matvec,
                              2, 2, a.indptr, a.indices, a.data, b, c)

            c = mx.zeros((2,), dtype=mx.result_type(a_dtype, b_dtype))
            _sparsetools.csr_matvec(2, 2, a.indptr, a.indices, a.data, b, c)
            assert_allclose(c, mx.dot(a.toarray(), b), err_msg=msg)


def test_endianness():
    d = mx.ones((3,4))
    offsets = [-1,0,1]

    a = dia_matrix((d.astype('<f8'), offsets), (4, 4))
    b = dia_matrix((d.astype('>f8'), offsets), (4, 4))
    v = mx.arange(4)

    assert_allclose(a.dot(v), [1, 3, 6, 5])
    assert_allclose(b.dot(v), [1, 3, 6, 5])
