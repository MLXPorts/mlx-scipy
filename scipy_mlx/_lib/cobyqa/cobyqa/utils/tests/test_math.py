import mlx.core as mx
import pytest

from ..math import get_arrays_tol, exact_1d_array, exact_2d_array


class TestGetArraysTol:

    def test_simple(self):
        tol = get_arrays_tol(mx.array([1, 2]), mx.array([3, 4, 5]))
        assert mx.isfinite(tol)
        assert tol < 1e3 * mx.finfo(mx.float64).eps

    def test_infinite(self):
        tol = get_arrays_tol(mx.array([1, 2]), mx.array([3, 4, float("inf")]))
        assert mx.isfinite(tol)
        assert tol < 1e3 * mx.finfo(mx.float64).eps

    def test_nan(self):
        tol = get_arrays_tol(mx.array([1, 2]), mx.array([3, 4, float("nan")]))
        assert mx.isfinite(tol)
        assert tol < 1e3 * mx.finfo(mx.float64).eps

    def test_exceptions(self):
        with pytest.raises(ValueError):
            get_arrays_tol()


class TestExact1DArray:

    def test_simple(self):
        x = exact_1d_array([1, 2], "Error")
        assert mx.array_equal(x, mx.array([1.0, 2.0]))

    def test_broadcast(self):
        x = exact_1d_array(1, "Error")
        assert mx.array_equal(x, mx.array([1.0]))

    def test_exceptions(self):
        with pytest.raises(ValueError):
            exact_1d_array([[1.0, 2.0], [3.0, 4.0]], "Error")


class TestExact2DArray:

    def test_simple(self):
        x = exact_2d_array([[1, 2], [3, 4]], "Error")
        assert mx.array_equal(x, mx.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_broadcast(self):
        x = exact_2d_array([1, 2], "Error")
        assert mx.array_equal(x, mx.array([[1.0, 2.0]]))

    def test_exceptions(self):
        with pytest.raises(ValueError):
            exact_2d_array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], "Error")
