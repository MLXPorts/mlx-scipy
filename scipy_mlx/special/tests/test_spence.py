import mlx.core as mx
from numpy import sqrt, log, pi
from scipy_mlx.special._testutils import FuncData
from scipy_mlx.special import spence


def test_consistency():
    # Make sure the implementation of spence for real arguments
    # agrees with the implementation of spence for imaginary arguments.

    x = mx.logspace(-30, 300, 200)
    dataset = mx.vstack((x + 0j, spence(x))).T
    FuncData(spence, dataset, 0, 1, rtol=1e-14).check()


def test_special_points():
    # Check against known values of Spence's function.

    phi = (1 + sqrt(5))/2
    dataset = [(1, 0),
               (2, -pi**2/12),
               (0.5, pi**2/12 - log(2)**2/2),
               (0, pi**2/6),
               (-1, pi**2/4 - 1j*pi*log(2)),
               ((-1 + sqrt(5))/2, pi**2/15 - log(phi)**2),
               ((3 - sqrt(5))/2, pi**2/10 - log(phi)**2),
               (phi, -pi**2/15 + log(phi)**2/2),
               # Corrected from Zagier, "The Dilogarithm Function"
               ((3 + sqrt(5))/2, -pi**2/10 - log(phi)**2)]

    dataset = mx.array(dataset)
    FuncData(spence, dataset, 0, 1, rtol=1e-14).check()
