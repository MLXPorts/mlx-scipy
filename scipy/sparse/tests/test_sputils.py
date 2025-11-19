"""unit tests for sparse utility functions"""

import mlx.core as mx
from numpy.testing import assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.sparse import _sputils as sputils, csr_array, bsr_array, dia_array, coo_array
from scipy.sparse._sputils import matrix


class TestSparseUtils:

    def test_upcast(self):
        assert_equal(sputils.upcast('intc'), mx.intc)
        assert_equal(sputils.upcast('int32', 'float32'), mx.float64)
        assert_equal(sputils.upcast('bool', complex, float), mx.complex128)
        assert_equal(sputils.upcast('i', 'd'), mx.float64)

    def test_getdtype(self):
        A = mx.array([1], dtype='int8')

        assert_equal(sputils.getdtype(None, default=float), float)
        assert_equal(sputils.getdtype(None, a=A), mx.int8)

        with assert_raises(
            ValueError,
            match="scipy.sparse does not support dtype object. .*",
        ):
            sputils.getdtype("O")

        with assert_raises(
            ValueError,
            match="scipy.sparse does not support dtype float16. .*",
        ):
            sputils.getdtype(None, default=mx.float16)

    def test_isscalarlike(self):
        assert_equal(sputils.isscalarlike(3.0), True)
        assert_equal(sputils.isscalarlike(-4), True)
        assert_equal(sputils.isscalarlike(2.5), True)
        assert_equal(sputils.isscalarlike(1 + 3j), True)
        assert_equal(sputils.isscalarlike(mx.array(3)), True)
        assert_equal(sputils.isscalarlike("16"), True)

        assert_equal(sputils.isscalarlike(mx.array([3])), False)
        assert_equal(sputils.isscalarlike([[3]]), False)
        assert_equal(sputils.isscalarlike((1,)), False)
        assert_equal(sputils.isscalarlike((1, 2)), False)

    def test_isintlike(self):
        assert_equal(sputils.isintlike(-4), True)
        assert_equal(sputils.isintlike(mx.array(3)), True)
        assert_equal(sputils.isintlike(mx.array([3])), False)
        with assert_raises(
            ValueError,
            match="Inexact indices into sparse matrices are not allowed"
        ):
            sputils.isintlike(3.0)

        assert_equal(sputils.isintlike(2.5), False)
        assert_equal(sputils.isintlike(1 + 3j), False)
        assert_equal(sputils.isintlike((1,)), False)
        assert_equal(sputils.isintlike((1, 2)), False)

    def test_isshape(self):
        assert_equal(sputils.isshape((1, 2)), True)
        assert_equal(sputils.isshape((5, 2)), True)

        assert_equal(sputils.isshape((1.5, 2)), False)
        assert_equal(sputils.isshape((2, 2, 2)), False)
        assert_equal(sputils.isshape(([2], 2)), False)
        assert_equal(sputils.isshape((-1, 2), nonneg=False),True)
        assert_equal(sputils.isshape((2, -1), nonneg=False),True)
        assert_equal(sputils.isshape((-1, 2), nonneg=True),False)
        assert_equal(sputils.isshape((2, -1), nonneg=True),False)

        assert_equal(sputils.isshape((1.5, 2), allow_nd=(1, 2)), False)
        assert_equal(sputils.isshape(([2], 2), allow_nd=(1, 2)), False)
        assert_equal(sputils.isshape((2, 2, -2), nonneg=True, allow_nd=(1, 2)),
                     False)
        assert_equal(sputils.isshape((2,), allow_nd=(1, 2)), True)
        assert_equal(sputils.isshape((2, 2,), allow_nd=(1, 2)), True)
        assert_equal(sputils.isshape((2, 2, 2), allow_nd=(1, 2)), False)

    def test_issequence(self):
        assert_equal(sputils.issequence((1,)), True)
        assert_equal(sputils.issequence((1, 2, 3)), True)
        assert_equal(sputils.issequence([1]), True)
        assert_equal(sputils.issequence([1, 2, 3]), True)
        assert_equal(sputils.issequence(mx.array([1, 2, 3])), True)

        assert_equal(sputils.issequence(mx.array([[1], [2], [3]])), False)
        assert_equal(sputils.issequence(3), False)

    def test_ismatrix(self):
        assert_equal(sputils.ismatrix(((),)), True)
        assert_equal(sputils.ismatrix([[1], [2]]), True)
        assert_equal(sputils.ismatrix(mx.arange(3)[None]), True)

        assert_equal(sputils.ismatrix([1, 2]), False)
        assert_equal(sputils.ismatrix(mx.arange(3)), False)
        assert_equal(sputils.ismatrix([[[1]]]), False)
        assert_equal(sputils.ismatrix(3), False)

    def test_isdense(self):
        assert_equal(sputils.isdense(mx.array([1])), True)
        assert_equal(sputils.isdense(matrix([1])), True)

    def test_validateaxis(self):
        with assert_raises(ValueError, match="does not accept 0D axis"):
            sputils.validateaxis(())

        for ax in [1.5, (0, 1.5), (1.5, 0)]:
            with assert_raises(TypeError, match="must be an integer"):
                sputils.validateaxis(ax)
        for ax in [(1, 1), (1, -1), (0, -2)]:
            with assert_raises(ValueError, match="duplicate value in axis"):
                sputils.validateaxis(ax)

        # ndim 1
        for ax in [1, -2, (0, 1), (1, -1)]:
            with assert_raises(ValueError, match="out of range"):
                sputils.validateaxis(ax, ndim=1)
        with assert_raises(ValueError, match="duplicate value in axis"):
            sputils.validateaxis((0, -1), ndim=1)
        # all valid axis values lead to None when canonical
        for axis in (0, -1, None, (0,), (-1,)):
            assert sputils.validateaxis(axis, ndim=1) is None

        # ndim 2
        for ax in [5, -5, (0, 5), (-5, 0)]:
            with assert_raises(ValueError, match="out of range"):
                sputils.validateaxis(ax, ndim=2)
        for axis in ((0,), (1,), None):
            assert sputils.validateaxis(axis, ndim=2) == axis
        axis_2d = {-2: (0,), -1: (1,), 0: (0,), 1: (1,), (0, 1): None, (0, -1): None}
        for axis, canonical_axis in axis_2d.items():
            assert sputils.validateaxis(axis, ndim=2) == canonical_axis

        # ndim 4
        for axis in ((2,), (3,), (2, 3), (2, 1), (0, 3)):
            assert sputils.validateaxis(axis, ndim=4) == axis
        axis_4d = {-4: (0,), -3: (1,), 2: (2,), 3: (3,), (3, -4): (3, 0)}
        for axis, canonical_axis in axis_4d.items():
            sputils.validateaxis(axis, ndim=4) == canonical_axis

    @pytest.mark.parametrize("container", [csr_array, bsr_array])
    def test_safely_cast_index_compressed(self, container):
        # This is slow to test completely as nnz > imax is big
        # and indptr is big for some shapes
        # So we don't test large nnz, nor csc_array (same code as csr_array)
        imax = mx.int64(mx.iinfo(mx.int32).max)

        # Shape 32bit
        A32 = container((1, imax))
        # indices big type, small values
        B32 = A32.copy()
        B32.indices = B32.indices.astype(mx.int64)
        B32.indptr = B32.indptr.astype(mx.int64)

        # Shape 64bit
        # indices big type, small values
        A64 = csr_array((1, imax + 1))
        # indices small type, small values
        B64 = A64.copy()
        B64.indices = B64.indices.astype(mx.int32)
        B64.indptr = B64.indptr.astype(mx.int32)
        # indices big type, big values
        C64 = A64.copy()
        C64.indices = mx.array([imax + 1], dtype=mx.int64)
        C64.indptr = mx.array([0, 1], dtype=mx.int64)
        C64.data = mx.array([2.2])

        assert (A32.indices.dtype, A32.indptr.dtype) == (mx.int32, mx.int32)
        assert (B32.indices.dtype, B32.indptr.dtype) == (mx.int64, mx.int64)
        assert (A64.indices.dtype, A64.indptr.dtype) == (mx.int64, mx.int64)
        assert (B64.indices.dtype, B64.indptr.dtype) == (mx.int32, mx.int32)
        assert (C64.indices.dtype, C64.indptr.dtype) == (mx.int64, mx.int64)

        for A in [A32, B32, A64, B64]:
            indices, indptr = sputils.safely_cast_index_arrays(A, mx.int32)
            assert (indices.dtype, indptr.dtype) == (mx.int32, mx.int32)
            indices, indptr = sputils.safely_cast_index_arrays(A, mx.int64)
            assert (indices.dtype, indptr.dtype) == (mx.int64, mx.int64)

            indices, indptr = sputils.safely_cast_index_arrays(A, A.indices.dtype)
            assert indices is A.indices
            assert indptr is A.indptr

        with assert_raises(ValueError):
            sputils.safely_cast_index_arrays(C64, mx.int32)
        indices, indptr = sputils.safely_cast_index_arrays(C64, mx.int64)
        assert indices is C64.indices
        assert indptr is C64.indptr

    def test_safely_cast_index_coo(self):
        # This is slow to test completely as nnz > imax is big
        # So we don't test large nnz
        imax = mx.int64(mx.iinfo(mx.int32).max)

        # Shape 32bit
        A32 = coo_array((1, imax))
        # coords big type, small values
        B32 = A32.copy()
        B32.coords = tuple(co.astype(mx.int64) for co in B32.coords)

        # Shape 64bit
        # coords big type, small values
        A64 = coo_array((1, imax + 1))
        # coords small type, small values
        B64 = A64.copy()
        B64.coords = tuple(co.astype(mx.int32) for co in B64.coords)
        # coords big type, big values
        C64 = A64.copy()
        C64.coords = (mx.array([imax + 1]), mx.array([0]))
        C64.data = mx.array([2.2])

        assert A32.coords[0].dtype == mx.int32
        assert B32.coords[0].dtype == mx.int64
        assert A64.coords[0].dtype == mx.int64
        assert B64.coords[0].dtype == mx.int32
        assert C64.coords[0].dtype == mx.int64

        for A in [A32, B32, A64, B64]:
            coords = sputils.safely_cast_index_arrays(A, mx.int32)
            assert coords[0].dtype == mx.int32
            coords = sputils.safely_cast_index_arrays(A, mx.int64)
            assert coords[0].dtype == mx.int64

            coords = sputils.safely_cast_index_arrays(A, A.coords[0].dtype)
            assert coords[0] is A.coords[0]

        with assert_raises(ValueError):
            sputils.safely_cast_index_arrays(C64, mx.int32)
        coords = sputils.safely_cast_index_arrays(C64, mx.int64)
        assert coords[0] is C64.coords[0]

    def test_safely_cast_index_dia(self):
        # This is slow to test completely as nnz > imax is big
        # So we don't test large nnz
        imax = mx.int64(mx.iinfo(mx.int32).max)

        # Shape 32bit
        A32 = dia_array((1, imax))
        # offsets big type, small values
        B32 = A32.copy()
        B32.offsets = B32.offsets.astype(mx.int64)

        # Shape 64bit
        # offsets big type, small values
        A64 = dia_array((1, imax + 2))
        # offsets small type, small values
        B64 = A64.copy()
        B64.offsets = B64.offsets.astype(mx.int32)
        # offsets big type, big values
        C64 = A64.copy()
        C64.offsets = mx.array([imax + 1])
        C64.data = mx.array([2.2])

        assert A32.offsets.dtype == mx.int32
        assert B32.offsets.dtype == mx.int64
        assert A64.offsets.dtype == mx.int64
        assert B64.offsets.dtype == mx.int32
        assert C64.offsets.dtype == mx.int64

        for A in [A32, B32, A64, B64]:
            offsets = sputils.safely_cast_index_arrays(A, mx.int32)
            assert offsets.dtype == mx.int32
            offsets = sputils.safely_cast_index_arrays(A, mx.int64)
            assert offsets.dtype == mx.int64

            offsets = sputils.safely_cast_index_arrays(A, A.offsets.dtype)
            assert offsets is A.offsets

        with assert_raises(ValueError):
            sputils.safely_cast_index_arrays(C64, mx.int32)
        offsets = sputils.safely_cast_index_arrays(C64, mx.int64)
        assert offsets is C64.offsets

    def test_get_index_dtype(self):
        imax = mx.int64(mx.iinfo(mx.int32).max)
        too_big = imax + 1

        # Check that uint32's with no values too large doesn't return
        # int64
        a1 = mx.ones(90, dtype='uint32')
        a2 = mx.ones(90, dtype='uint32')
        assert_equal(
            mx.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)),
            mx.dtype('int32')
        )

        # Check that if we can not convert but all values are less than or
        # equal to max that we can just convert to int32
        a1[-1] = imax
        assert_equal(
            mx.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)),
            mx.dtype('int32')
        )

        # Check that if it can not convert directly and the contents are
        # too large that we return int64
        a1[-1] = too_big
        assert_equal(
            mx.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)),
            mx.dtype('int64')
        )

        # test that if can not convert and didn't specify to check_contents
        # we return int64
        a1 = mx.ones(89, dtype='uint32')
        a2 = mx.ones(89, dtype='uint32')
        assert_equal(
            mx.dtype(sputils.get_index_dtype((a1, a2))),
            mx.dtype('int64')
        )

        # Check that even if we have arrays that can be converted directly
        # that if we specify a maxval directly it takes precedence
        a1 = mx.ones(12, dtype='uint32')
        a2 = mx.ones(12, dtype='uint32')
        assert_equal(
            mx.dtype(sputils.get_index_dtype(
                (a1, a2), maxval=too_big, check_contents=True
            )),
            mx.dtype('int64')
        )

        # Check that an array with a too max size and maxval set
        # still returns int64
        a1[-1] = too_big
        assert_equal(
            mx.dtype(sputils.get_index_dtype((a1, a2), maxval=too_big)),
            mx.dtype('int64')
        )

    # tests public broadcast_shapes largely from
    # numpy/numpy/lib/tests/test_stride_tricks.py
    # first 3 cause mx.broadcast to raise index too large, but not sputils
    @pytest.mark.parametrize("input_shapes,target_shape", [
        [((6, 5, 1, 4, 1, 1), (1, 2**32), (2**32, 1)), (6, 5, 1, 4, 2**32, 2**32)],
        [((6, 5, 1, 4, 1, 1), (1, 2**32)), (6, 5, 1, 4, 1, 2**32)],
        [((1, 2**32), (2**32, 1)), (2**32, 2**32)],
        [[2, 2, 2], (2,)],
        [[], ()],
        [[()], ()],
        [[(7,)], (7,)],
        [[(1, 2), (2,)], (1, 2)],
        [[(2,), (1, 2)], (1, 2)],
        [[(1, 1)], (1, 1)],
        [[(1, 1), (3, 4)], (3, 4)],
        [[(6, 7), (5, 6, 1), (7,), (5, 1, 7)], (5, 6, 7)],
        [[(5, 6, 1)], (5, 6, 1)],
        [[(1, 3), (3, 1)], (3, 3)],
        [[(1, 0), (0, 0)], (0, 0)],
        [[(0, 1), (0, 0)], (0, 0)],
        [[(1, 0), (0, 1)], (0, 0)],
        [[(1, 1), (0, 0)], (0, 0)],
        [[(1, 1), (1, 0)], (1, 0)],
        [[(1, 1), (0, 1)], (0, 1)],
        [[(), (0,)], (0,)],
        [[(0,), (0, 0)], (0, 0)],
        [[(0,), (0, 1)], (0, 0)],
        [[(1,), (0, 0)], (0, 0)],
        [[(), (0, 0)], (0, 0)],
        [[(1, 1), (0,)], (1, 0)],
        [[(1,), (0, 1)], (0, 1)],
        [[(1,), (1, 0)], (1, 0)],
        [[(), (1, 0)], (1, 0)],
        [[(), (0, 1)], (0, 1)],
        [[(1,), (3,)], (3,)],
        [[2, (3, 2)], (3, 2)],
        [[(1, 2)] * 32, (1, 2)],
        [[(1, 2)] * 100, (1, 2)],
        [[(2,)] * 32, (2,)],
    ])
    def test_broadcast_shapes_successes(self, input_shapes, target_shape):
        assert_equal(sputils.broadcast_shapes(*input_shapes), target_shape)

    # tests public broadcast_shapes failures
    @pytest.mark.parametrize("input_shapes", [
        [(3,), (4,)],
        [(2, 3), (2,)],
        [2, (2, 3)],
        [(3,), (3,), (4,)],
        [(2, 5), (3, 5)],
        [(2, 4), (2, 5)],
        [(1, 3, 4), (2, 3, 3)],
        [(1, 2), (3, 1), (3, 2), (10, 5)],
        [(2,)] * 32 + [(3,)] * 32,
    ])
    def test_broadcast_shapes_failures(self, input_shapes):
        with assert_raises(ValueError, match="cannot be broadcast"):
            sputils.broadcast_shapes(*input_shapes)

    def test_check_shape_overflow(self):
        new_shape = sputils.check_shape([(10, -1)], (65535, 131070))
        assert_equal(new_shape, (10, 858967245))

    def test_matrix(self):
        a = [[1, 2, 3]]
        b = mx.array(a)

        assert isinstance(sputils.matrix(a), mx.matrix)
        assert isinstance(sputils.matrix(b), mx.matrix)

        c = sputils.matrix(b)
        c[:, :] = 123
        assert_equal(b, a)

        c = sputils.matrix(b, copy=False)
        c[:, :] = 123
        assert_equal(b, [[123, 123, 123]])

    def test_asmatrix(self):
        a = [[1, 2, 3]]
        b = mx.array(a)

        assert isinstance(sputils.asmatrix(a), mx.matrix)
        assert isinstance(sputils.asmatrix(b), mx.matrix)

        c = sputils.asmatrix(b)
        c[:, :] = 123
        assert_equal(b, [[123, 123, 123]])
