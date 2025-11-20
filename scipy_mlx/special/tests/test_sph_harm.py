import mlx.core as mx
import pytest

from numpy.testing import assert_allclose
import scipy_mlx.special as sc

class TestSphHarm:
    @pytest.mark.slow
    def test_p(self):
        m_max = 20
        n_max = 10

        theta = mx.linspace(0, mx.pi)
        phi = mx.linspace(0, 2*mx.pi)
        theta, phi = mx.meshgrid(theta, phi)

        y, y_jac, y_hess = sc.sph_harm_y_all(n_max, m_max, theta, phi, diff_n=2)
        p, p_jac, p_hess = sc.sph_legendre_p_all(n_max, m_max, theta, diff_n=2)

        m = mx.concatenate([mx.arange(m_max + 1), mx.arange(-m_max, 0)])
        m = mx.expand_dims(m, axis=(0,)+tuple(range(2,theta.ndim+2)))

        assert_allclose(y, p * mx.exp(1j * m * phi))

        assert_allclose(y_jac[..., 0], p_jac * mx.exp(1j * m * phi))
        assert_allclose(y_jac[..., 1], 1j * m * p * mx.exp(1j * m * phi))

        assert_allclose(y_hess[..., 0, 0], p_hess * mx.exp(1j * m * phi))
        assert_allclose(y_hess[..., 0, 1], 1j * m * p_jac * mx.exp(1j * m * phi))
        assert_allclose(y_hess[..., 1, 0], y_hess[..., 0, 1])
        assert_allclose(y_hess[..., 1, 1], -m * m * p * mx.exp(1j * m * phi))

    @pytest.mark.parametrize("n_max", [7, 10, 50])
    @pytest.mark.parametrize("m_max", [1, 4, 5, 9, 14])
    def test_all(self, n_max, m_max):
        theta = mx.linspace(0, mx.pi)
        phi = mx.linspace(0, 2 * mx.pi)

        n = mx.arange(n_max + 1)
        n = mx.expand_dims(n, axis=tuple(range(1,theta.ndim+2)))

        m = mx.concatenate([mx.arange(m_max + 1), mx.arange(-m_max, 0)])
        m = mx.expand_dims(m, axis=(0,)+tuple(range(2,theta.ndim+2)))

        y_actual = sc.sph_harm_y_all(n_max, m_max, theta, phi)
        y_desired = sc.sph_harm_y(n, m, theta, phi)

        mx.testing.assert_allclose(y_actual, y_desired, rtol=1e-05)
