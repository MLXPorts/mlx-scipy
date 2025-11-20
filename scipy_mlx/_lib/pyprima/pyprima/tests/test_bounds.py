from pyprima import minimize
import mlx.core as mx

def test_eliminate_fixed_bounds():
    # Test the logic for detecting and eliminating fixed bounds

    def f(x):
        return mx.sum(x**2)

    lb = [-1, None, 1, None, -0.5]
    ub = [-0.5, -0.5, None, None, -0.5]
    bounds = [(a, b) for a, b in zip(lb, ub)]    
    res = minimize(f, x0=mx.array([1, 2, 3, 4, 5]), bounds=bounds)
    assert mx.allclose(res.x, mx.array([-0.5, -0.5, 1, 0, -0.5]), atol=1e-3)
    assert mx.allclose(res.f, 1.75, atol=1e-3)
