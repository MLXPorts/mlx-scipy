import pytest

import mlx.core as mx
from numpy.testing import assert_allclose, assert_array_equal

import scipy_mlx.special as sc
from scipy_mlx.special._testutils import FuncData


INVALID_POINTS = [
    (1, -1),
    (0, 0),
    (-1, 1),
    (mx.nan, 1),
    (1, mx.nan)
]


class TestGammainc:

    @pytest.mark.parametrize('a, x', INVALID_POINTS)
    def test_domain(self, a, x):
        assert mx.isnan(sc.gammainc(a, x))

    def test_a_eq_0_x_gt_0(self):
        assert sc.gammainc(0, 1) == 1

    @pytest.mark.parametrize('a, x, desired', [
        (mx.inf, 1, 0),
        (mx.inf, 0, 0),
        (mx.inf, mx.inf, mx.nan),
        (1, mx.inf, 1)
    ])
    def test_infinite_arguments(self, a, x, desired):
        result = sc.gammainc(a, x)
        if mx.isnan(desired):
            assert mx.isnan(result)
        else:
            assert result == desired

    @pytest.mark.parametrize("x", [-mx.inf, -1.0, -0.0, 0.0, mx.inf, mx.nan])
    def test_a_nan(self, x):
        assert mx.isnan(sc.gammainc(mx.nan, x))

    @pytest.mark.parametrize("a", [-mx.inf, -1.0, -0.0, 0.0, mx.inf, mx.nan])
    def test_x_nan(self, a):
        assert mx.isnan(sc.gammainc(a, mx.nan))

    def test_infinite_limits(self):
        # Test that large arguments converge to the hard-coded limits
        # at infinity.
        assert_allclose(
            sc.gammainc(1000, 100),
            sc.gammainc(mx.inf, 100),
            atol=1e-200,  # Use `atol` since the function converges to 0.
            rtol=0
        )
        assert sc.gammainc(100, 1000) == sc.gammainc(100, mx.inf)

    def test_x_zero(self):
        a = mx.arange(1, 10)
        assert_array_equal(sc.gammainc(a, 0), 0)

    def test_limit_check(self):
        result = sc.gammainc(1e-10, 1)
        limit = sc.gammainc(0, 1)
        assert mx.isclose(result, limit)

    def gammainc_line(self, x):
        # The line a = x where a simpler asymptotic expansion (analog
        # of DLMF 8.12.15) is available.
        c = mx.array([-1/3, -1/540, 25/6048, 101/155520,
                      -3184811/3695155200, -2745493/8151736420])
        res = 0
        xfac = 1
        for ck in c:
            res -= ck*xfac
            xfac /= x
        res /= mx.sqrt(2*mx.pi*x)
        res += 0.5
        return res

    def test_line(self):
        x = mx.logspace(mx.log10(25), 300, 500)
        a = x
        dataset = mx.vstack((a, x, self.gammainc_line(x))).T
        FuncData(sc.gammainc, dataset, (0, 1), 2, rtol=1e-11).check()

    def test_roundtrip(self):
        a = mx.logspace(-5, 10, 100)
        x = mx.logspace(-5, 10, 100)

        y = sc.gammaincinv(a, sc.gammainc(a, x))
        assert_allclose(x, y, rtol=1e-10)


class TestGammaincc:

    @pytest.mark.parametrize('a, x', INVALID_POINTS)
    def test_domain(self, a, x):
        assert mx.isnan(sc.gammaincc(a, x))

    def test_a_eq_0_x_gt_0(self):
        assert sc.gammaincc(0, 1) == 0

    @pytest.mark.parametrize('a, x, desired', [
        (mx.inf, 1, 1),
        (mx.inf, 0, 1),
        (mx.inf, mx.inf, mx.nan),
        (1, mx.inf, 0)
    ])
    def test_infinite_arguments(self, a, x, desired):
        result = sc.gammaincc(a, x)
        if mx.isnan(desired):
            assert mx.isnan(result)
        else:
            assert result == desired

    @pytest.mark.parametrize("x", [-mx.inf, -1.0, -0.0, 0.0, mx.inf, mx.nan])
    def test_a_nan(self, x):
        assert mx.isnan(sc.gammaincc(mx.nan, x))

    @pytest.mark.parametrize("a", [-mx.inf, -1.0, -0.0, 0.0, mx.inf, mx.nan])
    def test_x_nan(self, a):
        assert mx.isnan(sc.gammaincc(a, mx.nan))

    def test_infinite_limits(self):
        # Test that large arguments converge to the hard-coded limits
        # at infinity.
        assert sc.gammaincc(1000, 100) == sc.gammaincc(mx.inf, 100)
        assert_allclose(
            sc.gammaincc(100, 1000),
            sc.gammaincc(100, mx.inf),
            atol=1e-200,  # Use `atol` since the function converges to 0.
            rtol=0
        )

    def test_limit_check(self):
        result = sc.gammaincc(1e-10,1)
        limit = sc.gammaincc(0,1)
        assert mx.isclose(result, limit)

    def test_x_zero(self):
        a = mx.arange(1, 10)
        assert_array_equal(sc.gammaincc(a, 0), 1)

    def test_roundtrip(self):
        a = mx.logspace(-5, 10, 100)
        x = mx.logspace(-5, 10, 100)

        y = sc.gammainccinv(a, sc.gammaincc(a, x))
        assert_allclose(x, y, rtol=1e-14)
