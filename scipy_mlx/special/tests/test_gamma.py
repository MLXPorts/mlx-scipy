import mlx.core as mx
import scipy_mlx.special as sc


class TestRgamma:

    def test_gh_11315(self):
        assert sc.rgamma(-35) == 0

    def test_rgamma_zeros(self):
        x = mx.array([0, -10, -100, -1000, -10000])
        assert mx.all(sc.rgamma(x) == 0)
