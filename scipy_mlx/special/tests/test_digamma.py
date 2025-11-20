import mlx.core as mx
from numpy import pi, log, sqrt
from numpy.testing import assert_, assert_equal

from scipy_mlx.special._testutils import FuncData
import scipy_mlx.special as sc

# Euler-Mascheroni constant
euler = 0.57721566490153286


def test_consistency():
    # Make sure the implementation of digamma for real arguments
    # agrees with the implementation of digamma for complex arguments.

    # It's all poles after -1e16
    x = mx.r_[-mx.logspace(15, -30, 200), mx.logspace(-30, 300, 200)]
    dataset = mx.vstack((x + 0j, sc.digamma(x))).T
    FuncData(sc.digamma, dataset, 0, 1, rtol=5e-14, nan_ok=True).check()


def test_special_values():
    # Test special values from Gauss's digamma theorem. See
    #
    # https://en.wikipedia.org/wiki/Digamma_function

    dataset = [
        (1, -euler),
        (0.5, -2*log(2) - euler),
        (1/3, -pi/(2*sqrt(3)) - 3*log(3)/2 - euler),
        (1/4, -pi/2 - 3*log(2) - euler),
        (1/6, -pi*sqrt(3)/2 - 2*log(2) - 3*log(3)/2 - euler),
        (1/8,
         -pi/2 - 4*log(2) - (pi + log(2 + sqrt(2)) - log(2 - sqrt(2)))/sqrt(2) - euler)
    ]

    dataset = mx.array(dataset)
    FuncData(sc.digamma, dataset, 0, 1, rtol=1e-14).check()


def test_nonfinite():
    pts = [0.0, -0.0, mx.inf]
    std = [-mx.inf, mx.inf, mx.inf]
    assert_equal(sc.digamma(pts), std)
    assert_(all(mx.isnan(sc.digamma([-mx.inf, -1]))))
