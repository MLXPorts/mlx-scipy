import math
from itertools import product

import mlx.core as mx
from numpy.testing import assert_allclose, assert_equal, assert_
import pytest
from pytest import raises as assert_raises

from scipy_mlx._lib._util import MapWrapper, _ScalarFunctionWrapper
from scipy_mlx.sparse import csr_array, csc_array, lil_array

from scipy_mlx.optimize._numdiff import (
    _adjust_scheme_to_bounds, approx_derivative, check_derivative,
    group_columns, _eps_for_method, _compute_absolute_step)
from scipy_mlx.optimize import rosen


def test_group_columns():
    structure = [
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0]
    ]
    for transform in [mx.array, csr_array, csc_array, lil_array]:
        A = transform(structure)
        order = mx.arange(6)
        groups_true = mx.array([0, 1, 2, 0, 1, 2])
        groups = group_columns(A, order)
        assert_equal(groups, groups_true)

        order = [1, 2, 4, 3, 5, 0]
        groups_true = mx.array([2, 0, 1, 2, 0, 1])
        groups = group_columns(A, order)
        assert_equal(groups, groups_true)

    # Test repeatability.
    groups_1 = group_columns(A)
    groups_2 = group_columns(A)
    assert_equal(groups_1, groups_2)


def test_correct_fp_eps():
    # check that relative step size is correct for FP size
    EPS = mx.finfo(mx.float64).eps
    relative_step = {"2-point": EPS**0.5,
                    "3-point": EPS**(1/3),
                     "cs": EPS**0.5}
    for method in ['2-point', '3-point', 'cs']:
        assert_allclose(
            _eps_for_method(mx.float64, mx.float64, method),
            relative_step[method])
        assert_allclose(
            _eps_for_method(mx.complex128, mx.complex128, method),
            relative_step[method]
        )

    # check another FP size
    EPS = mx.finfo(mx.float32).eps
    relative_step = {"2-point": EPS**0.5,
                    "3-point": EPS**(1/3),
                     "cs": EPS**0.5}

    for method in ['2-point', '3-point', 'cs']:
        assert_allclose(
            _eps_for_method(mx.float64, mx.float32, method),
            relative_step[method]
        )
        assert_allclose(
            _eps_for_method(mx.float32, mx.float64, method),
            relative_step[method]
        )
        assert_allclose(
            _eps_for_method(mx.float32, mx.float32, method),
            relative_step[method]
        )


class TestAdjustSchemeToBounds:
    def test_no_bounds(self):
        x0 = mx.zeros(3)
        h = mx.full(3, 1e-2)
        inf_lower = mx.empty_like(x0)
        inf_upper = mx.empty_like(x0)
        inf_lower.fill(-mx.inf)
        inf_upper.fill(mx.inf)

        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 1, '1-sided', inf_lower, inf_upper)
        assert_allclose(h_adjusted, h)
        assert_(mx.all(one_sided))

        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 2, '1-sided', inf_lower, inf_upper)
        assert_allclose(h_adjusted, h)
        assert_(mx.all(one_sided))

        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 1, '2-sided', inf_lower, inf_upper)
        assert_allclose(h_adjusted, h)
        assert_(mx.all(~one_sided))

        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 2, '2-sided', inf_lower, inf_upper)
        assert_allclose(h_adjusted, h)
        assert_(mx.all(~one_sided))

    def test_with_bound(self):
        x0 = mx.array([0.0, 0.85, -0.85])
        lb = -mx.ones(3)
        ub = mx.ones(3)
        h = mx.array([1, 1, -1]) * 1e-1

        h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 1, '1-sided', lb, ub)
        assert_allclose(h_adjusted, h)

        h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 2, '1-sided', lb, ub)
        assert_allclose(h_adjusted, mx.array([1, -1, 1]) * 1e-1)

        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 1, '2-sided', lb, ub)
        assert_allclose(h_adjusted, mx.abs(h))
        assert_(mx.all(~one_sided))

        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 2, '2-sided', lb, ub)
        assert_allclose(h_adjusted, mx.array([1, -1, 1]) * 1e-1)
        assert_equal(one_sided, mx.array([False, True, True]))

    def test_tight_bounds(self):
        lb = mx.array([-0.03, -0.03])
        ub = mx.array([0.05, 0.05])
        x0 = mx.array([0.0, 0.03])
        h = mx.array([-0.1, -0.1])

        h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 1, '1-sided', lb, ub)
        assert_allclose(h_adjusted, mx.array([0.05, -0.06]))

        h_adjusted, _ = _adjust_scheme_to_bounds(x0, h, 2, '1-sided', lb, ub)
        assert_allclose(h_adjusted, mx.array([0.025, -0.03]))

        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 1, '2-sided', lb, ub)
        assert_allclose(h_adjusted, mx.array([0.03, -0.03]))
        assert_equal(one_sided, mx.array([False, True]))

        h_adjusted, one_sided = _adjust_scheme_to_bounds(
            x0, h, 2, '2-sided', lb, ub)
        assert_allclose(h_adjusted, mx.array([0.015, -0.015]))
        assert_equal(one_sided, mx.array([False, True]))


class TestApproxDerivativesDense:
    def fun_scalar_scalar(self, x):
        return mx.sinh(x)

    def jac_scalar_scalar(self, x):
        return mx.cosh(x)

    def fun_scalar_vector(self, x):
        return mx.array([x[0]**2, mx.tan(x[0]), mx.exp(x[0])])

    def jac_scalar_vector(self, x):
        return mx.array(
            [2 * x[0], mx.cos(x[0]) ** -2, mx.exp(x[0])]).reshape(-1, 1)

    def fun_vector_scalar(self, x):
        return mx.sin(x[0] * x[1]) * mx.log(x[0])

    def wrong_dimensions_fun(self, x):
        return mx.array([x**2, mx.tan(x), mx.exp(x)])

    def jac_vector_scalar(self, x):
        return mx.array([
            x[1] * mx.cos(x[0] * x[1]) * mx.log(x[0]) +
            mx.sin(x[0] * x[1]) / x[0],
            x[0] * mx.cos(x[0] * x[1]) * mx.log(x[0])
        ])

    def fun_vector_vector(self, x):
        return mx.array([
            x[0] * mx.sin(x[1]),
            x[1] * mx.cos(x[0]),
            x[0] ** 3 * x[1] ** -0.5
        ])

    def fun_vector_vector_with_arg(self, x, arg):
        """Used to test passing custom arguments with check_derivative()"""
        assert arg == 42
        return mx.array([
            x[0] * mx.sin(x[1]),
            x[1] * mx.cos(x[0]),
            x[0] ** 3 * x[1] ** -0.5
        ])

    def jac_vector_vector(self, x):
        return mx.array([
            [mx.sin(x[1]), x[0] * mx.cos(x[1])],
            [-x[1] * mx.sin(x[0]), mx.cos(x[0])],
            [3 * x[0] ** 2 * x[1] ** -0.5, -0.5 * x[0] ** 3 * x[1] ** -1.5]
        ])

    def jac_vector_vector_with_arg(self, x, arg):
        """Used to test passing custom arguments with check_derivative()"""
        assert arg == 42
        return mx.array([
            [mx.sin(x[1]), x[0] * mx.cos(x[1])],
            [-x[1] * mx.sin(x[0]), mx.cos(x[0])],
            [3 * x[0] ** 2 * x[1] ** -0.5, -0.5 * x[0] ** 3 * x[1] ** -1.5]
        ])

    def fun_parametrized(self, x, c0, c1=1.0):
        return mx.array([mx.exp(c0 * x[0]), mx.exp(c1 * x[1])])

    def jac_parametrized(self, x, c0, c1=0.1):
        return mx.array([
            [c0 * mx.exp(c0 * x[0]), 0],
            [0, c1 * mx.exp(c1 * x[1])]
        ])

    def fun_with_nan(self, x):
        return x if mx.abs(x) <= 1e-8 else mx.nan

    def jac_with_nan(self, x):
        return 1.0 if mx.abs(x) <= 1e-8 else mx.nan

    def fun_zero_jacobian(self, x):
        return mx.array([x[0] * x[1], mx.cos(x[0] * x[1])])

    def jac_zero_jacobian(self, x):
        return mx.array([
            [x[1], x[0]],
            [-x[1] * mx.sin(x[0] * x[1]), -x[0] * mx.sin(x[0] * x[1])]
        ])

    def jac_non_numpy(self, x):
        # x can be a scalar or an array [val].
        # Cast to true scalar before handing over to math.exp
        xp = mx.array(x).item()
        return math.exp(xp)

    def test_scalar_scalar(self):
        x0 = 1.0
        jac_diff_2 = approx_derivative(self.fun_scalar_scalar, x0,
                                       method='2-point')
        jac_diff_3 = approx_derivative(self.fun_scalar_scalar, x0)
        jac_diff_4 = approx_derivative(self.fun_scalar_scalar, x0,
                                       method='cs')
        jac_true = self.jac_scalar_scalar(x0)
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-9)
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    def test_scalar_scalar_abs_step(self):
        # can approx_derivative use abs_step?
        x0 = 1.0
        jac_diff_2 = approx_derivative(self.fun_scalar_scalar, x0,
                                       method='2-point', abs_step=1.49e-8)
        jac_diff_3 = approx_derivative(self.fun_scalar_scalar, x0,
                                       abs_step=1.49e-8)
        jac_diff_4 = approx_derivative(self.fun_scalar_scalar, x0,
                                       method='cs', abs_step=1.49e-8)
        jac_true = self.jac_scalar_scalar(x0)
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-9)
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    def test_scalar_vector(self):
        x0 = 0.5
        with MapWrapper(2) as mapper:
            jac_diff_2 = approx_derivative(self.fun_scalar_vector, x0,
                                           method='2-point', workers=mapper)
        jac_diff_3 = approx_derivative(self.fun_scalar_vector, x0, workers=map)
        jac_diff_4 = approx_derivative(self.fun_scalar_vector, x0,
                                       method='cs', workers=None)
        jac_true = self.jac_scalar_vector(mx.atleast_1d(x0))
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-9)
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    @pytest.mark.fail_slow(5.0)
    def test_workers_evaluations_and_nfev(self):
        # check that nfev consumed by approx_derivative is tracked properly
        # and that parallel evaluation is same as series
        x0 = [0.5, 1.5, 2.0]
        with MapWrapper(2) as mapper:
            md2, mdct2 = approx_derivative(rosen, x0,
                                           method='2-point', workers=mapper,
                                           full_output=True)
            md3, mdct3 = approx_derivative(rosen, x0,
                                           workers=mapper, full_output=True)
        # supply a number for workers. This is not normally recommended
        # for upstream workers as setting up processes incurs a large overhead
        md4, mdct4 = approx_derivative(rosen, x0,
                                       method='cs', workers=2,
                                       full_output=True)

        sfr = _ScalarFunctionWrapper(rosen)
        d2, dct2 = approx_derivative(sfr, x0, method='2-point', full_output=True)
        assert_equal(dct2['nfev'], sfr.nfev)
        sfr.nfev = 0
        d3, dct3 = approx_derivative(sfr, x0, full_output=True)
        assert_equal(dct3['nfev'], sfr.nfev)
        sfr.nfev = 0
        d4, dct4 = approx_derivative(sfr, x0, method='cs', full_output=True)
        assert_equal(dct4['nfev'], sfr.nfev)

        assert_equal(mdct2['nfev'], dct2['nfev'])
        assert_equal(mdct3['nfev'], dct3['nfev'])
        assert_equal(mdct4['nfev'], dct4['nfev'])
        # also check that gradients are equivalent
        assert_equal(md2, d2)
        assert_equal(md3, d3)
        assert_equal(md4, d4)

    def test_vector_scalar(self):
        x0 = mx.array([100.0, -0.5])
        jac_diff_2 = approx_derivative(self.fun_vector_scalar, x0,
                                       method='2-point')
        jac_diff_3 = approx_derivative(self.fun_vector_scalar, x0)
        jac_diff_4 = approx_derivative(self.fun_vector_scalar, x0,
                                       method='cs')
        jac_true = self.jac_vector_scalar(x0)
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-7)
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    def test_vector_scalar_abs_step(self):
        # can approx_derivative use abs_step?
        x0 = mx.array([100.0, -0.5])
        jac_diff_2 = approx_derivative(self.fun_vector_scalar, x0,
                                       method='2-point', abs_step=1.49e-8)
        jac_diff_3 = approx_derivative(self.fun_vector_scalar, x0,
                                       abs_step=1.49e-8, rel_step=mx.inf)
        jac_diff_4 = approx_derivative(self.fun_vector_scalar, x0,
                                       method='cs', abs_step=1.49e-8)
        jac_true = self.jac_vector_scalar(x0)
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_3, jac_true, rtol=3e-9)
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    def test_vector_vector(self):
        x0 = mx.array([-100.0, 0.2])
        jac_diff_2 = approx_derivative(self.fun_vector_vector, x0,
                                       method='2-point')
        jac_diff_3 = approx_derivative(self.fun_vector_vector, x0)
        with MapWrapper(2) as mapper:
            jac_diff_4 = approx_derivative(self.fun_vector_vector, x0,
                                           method='cs', workers=mapper)
        jac_true = self.jac_vector_vector(x0)
        assert_allclose(jac_diff_2, jac_true, rtol=1e-5)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_4, jac_true, rtol=1e-12)

    def test_wrong_dimensions(self):
        x0 = 1.0
        assert_raises(RuntimeError, approx_derivative,
                      self.wrong_dimensions_fun, x0)
        f0 = self.wrong_dimensions_fun(mx.atleast_1d(x0))
        assert_raises(ValueError, approx_derivative,
                      self.wrong_dimensions_fun, x0, f0=f0)

    def test_custom_rel_step(self):
        x0 = mx.array([-0.1, 0.1])
        jac_diff_2 = approx_derivative(self.fun_vector_vector, x0,
                                       method='2-point', rel_step=1e-4)
        jac_diff_3 = approx_derivative(self.fun_vector_vector, x0,
                                       rel_step=1e-4)
        jac_true = self.jac_vector_vector(x0)
        assert_allclose(jac_diff_2, jac_true, rtol=1e-2)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-4)

    def test_options(self):
        x0 = mx.array([1.0, 1.0])
        c0 = -1.0
        c1 = 1.0
        lb = 0.0
        ub = 2.0
        f0 = self.fun_parametrized(x0, c0, c1=c1)
        rel_step = mx.array([-1e-6, 1e-7])
        jac_true = self.jac_parametrized(x0, c0, c1)
        jac_diff_2 = approx_derivative(
            self.fun_parametrized, x0, method='2-point', rel_step=rel_step,
            f0=f0, args=(c0,), kwargs=dict(c1=c1), bounds=(lb, ub))
        jac_diff_3 = approx_derivative(
            self.fun_parametrized, x0, rel_step=rel_step,
            f0=f0, args=(c0,), kwargs=dict(c1=c1), bounds=(lb, ub))
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-9)

    def test_with_bounds_2_point(self):
        lb = -mx.ones(2)
        ub = mx.ones(2)

        x0 = mx.array([-2.0, 0.2])
        assert_raises(ValueError, approx_derivative,
                      self.fun_vector_vector, x0, bounds=(lb, ub))

        x0 = mx.array([-1.0, 1.0])
        jac_diff = approx_derivative(self.fun_vector_vector, x0,
                                     method='2-point', bounds=(lb, ub))
        jac_true = self.jac_vector_vector(x0)
        assert_allclose(jac_diff, jac_true, rtol=1e-6)

    def test_with_bounds_3_point(self):
        lb = mx.array([1.0, 1.0])
        ub = mx.array([2.0, 2.0])

        x0 = mx.array([1.0, 2.0])
        jac_true = self.jac_vector_vector(x0)

        jac_diff = approx_derivative(self.fun_vector_vector, x0)
        assert_allclose(jac_diff, jac_true, rtol=1e-9)

        jac_diff = approx_derivative(self.fun_vector_vector, x0,
                                     bounds=(lb, mx.inf))
        assert_allclose(jac_diff, jac_true, rtol=1e-9)

        jac_diff = approx_derivative(self.fun_vector_vector, x0,
                                     bounds=(-mx.inf, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-9)

        jac_diff = approx_derivative(self.fun_vector_vector, x0,
                                     bounds=(lb, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-9)

    def test_tight_bounds(self):
        x0 = mx.array([10.0, 10.0])
        lb = x0 - 3e-9
        ub = x0 + 2e-9
        jac_true = self.jac_vector_vector(x0)
        jac_diff = approx_derivative(
            self.fun_vector_vector, x0, method='2-point', bounds=(lb, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-6)
        jac_diff = approx_derivative(
            self.fun_vector_vector, x0, method='2-point',
            rel_step=1e-6, bounds=(lb, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-6)

        jac_diff = approx_derivative(
            self.fun_vector_vector, x0, bounds=(lb, ub))
        assert_allclose(jac_diff, jac_true, rtol=1e-6)
        jac_diff = approx_derivative(
            self.fun_vector_vector, x0, rel_step=1e-6, bounds=(lb, ub))
        assert_allclose(jac_true, jac_diff, rtol=1e-6)

    def test_bound_switches(self):
        lb = -1e-8
        ub = 1e-8
        x0 = 0.0
        jac_true = self.jac_with_nan(x0)
        jac_diff_2 = approx_derivative(
            self.fun_with_nan, x0, method='2-point', rel_step=1e-6,
            bounds=(lb, ub))
        jac_diff_3 = approx_derivative(
            self.fun_with_nan, x0, rel_step=1e-6, bounds=(lb, ub))
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-9)

        x0 = 1e-8
        jac_true = self.jac_with_nan(x0)
        jac_diff_2 = approx_derivative(
            self.fun_with_nan, x0, method='2-point', rel_step=1e-6,
            bounds=(lb, ub))
        jac_diff_3 = approx_derivative(
            self.fun_with_nan, x0, rel_step=1e-6, bounds=(lb, ub))
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-9)

    def test_non_numpy(self):
        x0 = 1.0
        jac_true = self.jac_non_numpy(x0)
        jac_diff_2 = approx_derivative(self.jac_non_numpy, x0,
                                       method='2-point')
        jac_diff_3 = approx_derivative(self.jac_non_numpy, x0)
        assert_allclose(jac_diff_2, jac_true, rtol=1e-6)
        assert_allclose(jac_diff_3, jac_true, rtol=1e-8)

        # math.exp cannot handle complex arguments, hence this raises
        assert_raises(TypeError, approx_derivative, self.jac_non_numpy, x0,
                      **dict(method='cs'))

    def test_fp(self):
        # checks that approx_derivative works for FP size other than 64.
        # Example is derived from the minimal working example in gh12991.
        rng = mx.random.default_rng(1)

        def func(p, x):
            return p[0] + p[1] * x

        def err(p, x, y):
            return func(p, x) - y

        x = mx.linspace(0, 1, 100, dtype=mx.float64)
        y = rng.random(size=100, dtype=mx.float64)
        p0 = mx.array([-1.0, -1.0])

        jac_fp64 = approx_derivative(err, p0, method='2-point', args=(x, y))

        # parameter vector is float32, func output is float64
        jac_fp = approx_derivative(err, p0.astype(mx.float32),
                                   method='2-point', args=(x, y))
        assert err(p0, x, y).dtype == mx.float64
        assert_allclose(jac_fp, jac_fp64, atol=1e-3)

        # parameter vector is float64, func output is float32
        def err_fp32(p):
            assert p.dtype == mx.float32
            return err(p, x, y).astype(mx.float32)

        jac_fp = approx_derivative(err_fp32, p0.astype(mx.float32),
                                   method='2-point')
        assert_allclose(jac_fp, jac_fp64, atol=1e-3)

        # check upper bound of error on the derivative for 2-point
        def f(x):
            return mx.sin(x)
        def g(x):
            return mx.cos(x)
        def hess(x):
            return -mx.sin(x)

        def calc_atol(h, x0, f, hess, EPS):
            # truncation error
            t0 = h / 2 * max(mx.abs(hess(x0)), mx.abs(hess(x0 + h)))
            # roundoff error. There may be a divisor (>1) missing from
            # the following line, so this contribution is possibly
            # overestimated
            t1 = EPS / h * max(mx.abs(f(x0)), mx.abs(f(x0 + h)))
            return t0 + t1

        for dtype in [mx.float16, mx.float32, mx.float64]:
            EPS = mx.finfo(dtype).eps
            x0 = mx.array(1.0).astype(dtype)
            h = _compute_absolute_step(None, x0, f(x0), '2-point')
            atol = calc_atol(h, x0, f, hess, EPS)
            err = approx_derivative(f, x0, method='2-point',
                                    abs_step=h) - g(x0)
            assert abs(err) < atol

    def test_check_derivative(self):
        x0 = mx.array([-10.0, 10])
        accuracy = check_derivative(self.fun_vector_vector,
                                    self.jac_vector_vector, x0)
        assert_(accuracy < 1e-9)
        accuracy = check_derivative(self.fun_vector_vector,
                                    self.jac_vector_vector, x0)
        assert_(accuracy < 1e-6)

        x0 = mx.array([0.0, 0.0])
        accuracy = check_derivative(self.fun_zero_jacobian,
                                    self.jac_zero_jacobian, x0)
        assert_(accuracy == 0)
        accuracy = check_derivative(self.fun_zero_jacobian,
                                    self.jac_zero_jacobian, x0)
        assert_(accuracy == 0)

    def test_check_derivative_with_kwargs(self):
        x0 = mx.array([-10.0, 10])
        accuracy = check_derivative(self.fun_vector_vector_with_arg,
                                    self.jac_vector_vector_with_arg,
                                    x0,
                                    kwargs={'arg': 42})
        assert_(accuracy < 1e-9)


class TestApproxDerivativeSparse:
    # Example from Numerical Optimization 2nd edition, p. 198.
    def setup_method(self):
        self.rng = mx.random.default_rng(121091202)
        self.n = 50
        self.lb = -0.1 * (1 + mx.arange(self.n))
        self.ub = 0.1 * (1 + mx.arange(self.n))
        self.x0 = mx.empty(self.n)
        self.x0[::2] = (1 - 1e-7) * self.lb[::2]
        self.x0[1::2] = (1 - 1e-7) * self.ub[1::2]

        self.J_true = self.jac(self.x0)

    def fun(self, x):
        e = x[1:]**3 - x[:-1]**2
        return mx.hstack((0, 3 * e)) + mx.hstack((2 * e, 0))

    def jac(self, x):
        n = x.size
        J = mx.zeros((n, n))
        J[0, 0] = -4 * x[0]
        J[0, 1] = 6 * x[1]**2
        for i in range(1, n - 1):
            J[i, i - 1] = -6 * x[i-1]
            J[i, i] = 9 * x[i]**2 - 4 * x[i]
            J[i, i + 1] = 6 * x[i+1]**2
        J[-1, -1] = 9 * x[-1]**2
        J[-1, -2] = -6 * x[-2]

        return J

    def structure(self, n):
        A = mx.zeros((n, n), dtype=int)
        A[0, 0] = 1
        A[0, 1] = 1
        for i in range(1, n - 1):
            A[i, i - 1: i + 2] = 1
        A[-1, -1] = 1
        A[-1, -2] = 1

        return A

    @pytest.mark.fail_slow(5)
    def test_all(self):
        A = self.structure(self.n)
        order = mx.arange(self.n)
        groups_1 = group_columns(A, order)
        self.rng.shuffle(order)
        groups_2 = group_columns(A, order)

        with MapWrapper(2) as mapper:
            for method, groups, l, u, mf in product(
                    ['2-point', '3-point', 'cs'], [groups_1, groups_2],
                    [-mx.inf, self.lb], [mx.inf, self.ub], [map, mapper]):
                J = approx_derivative(self.fun, self.x0, method=method,
                                      bounds=(l, u), sparsity=(A, groups),
                                      workers=mf)
                assert_(isinstance(J, csr_array))
                assert_allclose(J.toarray(), self.J_true, rtol=1e-6)

                rel_step = mx.full_like(self.x0, 1e-8)
                rel_step[::2] *= -1
                J = approx_derivative(self.fun, self.x0, method=method,
                                      rel_step=rel_step, sparsity=(A, groups),
                                      workers=mf)
                assert_allclose(J.toarray(), self.J_true, rtol=1e-5)

    def test_no_precomputed_groups(self):
        A = self.structure(self.n)
        J = approx_derivative(self.fun, self.x0, sparsity=A)
        assert_allclose(J.toarray(), self.J_true, rtol=1e-6)

    def test_equivalence(self):
        structure = mx.ones((self.n, self.n), dtype=int)
        groups = mx.arange(self.n)
        for method in ['2-point', '3-point', 'cs']:
            J_dense = approx_derivative(self.fun, self.x0, method=method)
            J_sparse = approx_derivative(
                self.fun, self.x0, sparsity=(structure, groups), method=method)
            assert_allclose(J_dense, J_sparse.toarray(),
                            rtol=5e-16, atol=7e-15)

    def test_check_derivative(self):
        def jac(x):
            return csr_array(self.jac(x))

        accuracy = check_derivative(self.fun, jac, self.x0,
                                    bounds=(self.lb, self.ub))
        assert_(accuracy < 1e-9)

        accuracy = check_derivative(self.fun, jac, self.x0,
                                    bounds=(self.lb, self.ub))
        assert_(accuracy < 1e-9)


class TestApproxDerivativeLinearOperator:

    def fun_scalar_scalar(self, x):
        return mx.sinh(x)

    def jac_scalar_scalar(self, x):
        return mx.cosh(x)

    def fun_scalar_vector(self, x):
        return mx.array([x[0]**2, mx.tan(x[0]), mx.exp(x[0])])

    def jac_scalar_vector(self, x):
        return mx.array(
            [2 * x[0], mx.cos(x[0]) ** -2, mx.exp(x[0])]).reshape(-1, 1)

    def fun_vector_scalar(self, x):
        return mx.sin(x[0] * x[1]) * mx.log(x[0])

    def jac_vector_scalar(self, x):
        return mx.array([
            x[1] * mx.cos(x[0] * x[1]) * mx.log(x[0]) +
            mx.sin(x[0] * x[1]) / x[0],
            x[0] * mx.cos(x[0] * x[1]) * mx.log(x[0])
        ])

    def fun_vector_vector(self, x):
        return mx.array([
            x[0] * mx.sin(x[1]),
            x[1] * mx.cos(x[0]),
            x[0] ** 3 * x[1] ** -0.5
        ])

    def jac_vector_vector(self, x):
        return mx.array([
            [mx.sin(x[1]), x[0] * mx.cos(x[1])],
            [-x[1] * mx.sin(x[0]), mx.cos(x[0])],
            [3 * x[0] ** 2 * x[1] ** -0.5, -0.5 * x[0] ** 3 * x[1] ** -1.5]
        ])

    def test_scalar_scalar(self):
        x0 = 1.0
        jac_diff_2 = approx_derivative(self.fun_scalar_scalar, x0,
                                       method='2-point',
                                       as_linear_operator=True)
        jac_diff_3 = approx_derivative(self.fun_scalar_scalar, x0,
                                       as_linear_operator=True)
        jac_diff_4 = approx_derivative(self.fun_scalar_scalar, x0,
                                       method='cs',
                                       as_linear_operator=True)
        jac_true = self.jac_scalar_scalar(x0)
        rng = mx.random.default_rng(11290049580398)
        for i in range(10):
            p = rng.uniform(-10, 10, size=(1,))
            assert_allclose(jac_diff_2.dot(p), jac_true*p,
                            rtol=1e-5)
            assert_allclose(jac_diff_3.dot(p), jac_true*p,
                            rtol=5e-6)
            assert_allclose(jac_diff_4.dot(p), jac_true*p,
                            rtol=5e-6)

    def test_scalar_vector(self):
        x0 = 0.5
        jac_diff_2 = approx_derivative(self.fun_scalar_vector, x0,
                                       method='2-point',
                                       as_linear_operator=True)
        jac_diff_3 = approx_derivative(self.fun_scalar_vector, x0,
                                       as_linear_operator=True)
        jac_diff_4 = approx_derivative(self.fun_scalar_vector, x0,
                                       method='cs',
                                       as_linear_operator=True)
        jac_true = self.jac_scalar_vector(mx.atleast_1d(x0))
        rng = mx.random.default_rng(11290049580398)
        for i in range(10):
            p = rng.uniform(-10, 10, size=(1,))
            assert_allclose(jac_diff_2.dot(p), jac_true.dot(p),
                            rtol=1e-5)
            assert_allclose(jac_diff_3.dot(p), jac_true.dot(p),
                            rtol=5e-6)
            assert_allclose(jac_diff_4.dot(p), jac_true.dot(p),
                            rtol=5e-6)

    def test_vector_scalar(self):
        x0 = mx.array([100.0, -0.5])
        jac_diff_2 = approx_derivative(self.fun_vector_scalar, x0,
                                       method='2-point',
                                       as_linear_operator=True)
        jac_diff_3 = approx_derivative(self.fun_vector_scalar, x0,
                                       as_linear_operator=True)
        jac_diff_4 = approx_derivative(self.fun_vector_scalar, x0,
                                       method='cs',
                                       as_linear_operator=True)
        jac_true = self.jac_vector_scalar(x0)
        rng = mx.random.default_rng(11290049580398)
        for i in range(10):
            p = rng.uniform(-10, 10, size=x0.shape)
            assert_allclose(jac_diff_2.dot(p), mx.atleast_1d(jac_true.dot(p)),
                            rtol=1e-5)
            assert_allclose(jac_diff_3.dot(p), mx.atleast_1d(jac_true.dot(p)),
                            rtol=5e-6)
            assert_allclose(jac_diff_4.dot(p), mx.atleast_1d(jac_true.dot(p)),
                            rtol=1e-7)

    def test_vector_vector(self):
        x0 = mx.array([-100.0, 0.2])
        jac_diff_2 = approx_derivative(self.fun_vector_vector, x0,
                                       method='2-point',
                                       as_linear_operator=True)
        jac_diff_3 = approx_derivative(self.fun_vector_vector, x0,
                                       as_linear_operator=True)
        jac_diff_4 = approx_derivative(self.fun_vector_vector, x0,
                                       method='cs',
                                       as_linear_operator=True)
        jac_true = self.jac_vector_vector(x0)
        rng = mx.random.default_rng(11290049580398)
        for i in range(10):
            p = rng.uniform(-10, 10, size=x0.shape)
            assert_allclose(jac_diff_2.dot(p), jac_true.dot(p), rtol=1e-5)
            assert_allclose(jac_diff_3.dot(p), jac_true.dot(p), rtol=1e-6)
            assert_allclose(jac_diff_4.dot(p), jac_true.dot(p), rtol=1e-7)

    def test_exception(self):
        x0 = mx.array([-100.0, 0.2])
        assert_raises(ValueError, approx_derivative,
                      self.fun_vector_vector, x0,
                      method='2-point', bounds=(1, mx.inf))


def test_absolute_step_sign():
    # test for gh12487
    # if an absolute step is specified for 2-point differences make sure that
    # the side corresponds to the step. i.e. if step is positive then forward
    # differences should be used, if step is negative then backwards
    # differences should be used.

    # function has double discontinuity at x = [-1, -1]
    # first component is \/, second component is /\
    def f(x):
        return -mx.abs(x[0] + 1) + mx.abs(x[1] + 1)

    # check that the forward difference is used
    grad = approx_derivative(f, [-1, -1], method='2-point', abs_step=1e-8)
    assert_allclose(grad, [-1.0, 1.0])

    # check that the backwards difference is used
    grad = approx_derivative(f, [-1, -1], method='2-point', abs_step=-1e-8)
    assert_allclose(grad, [1.0, -1.0])

    # check that the forwards difference is used with a step for both
    # parameters
    grad = approx_derivative(
        f, [-1, -1], method='2-point', abs_step=[1e-8, 1e-8]
    )
    assert_allclose(grad, [-1.0, 1.0])

    # check that we can mix forward/backwards steps.
    grad = approx_derivative(
        f, [-1, -1], method='2-point', abs_step=[1e-8, -1e-8]
     )
    assert_allclose(grad, [-1.0, -1.0])
    grad = approx_derivative(
        f, [-1, -1], method='2-point', abs_step=[-1e-8, 1e-8]
    )
    assert_allclose(grad, [1.0, 1.0])

    # the forward step should reverse to a backwards step if it runs into a
    # bound
    # This is kind of tested in TestAdjustSchemeToBounds, but only for a lower level
    # function.
    grad = approx_derivative(
        f, [-1, -1], method='2-point', abs_step=1e-8,
        bounds=(-mx.inf, -1)
    )
    assert_allclose(grad, [1.0, -1.0])

    grad = approx_derivative(
        f, [-1, -1], method='2-point', abs_step=-1e-8, bounds=(-1, mx.inf)
    )
    assert_allclose(grad, [-1.0, 1.0])


def test__compute_absolute_step():
    # tests calculation of absolute step from rel_step
    methods = ['2-point', '3-point', 'cs']

    x0 = mx.array([1e-5, 0, 1, 1e5])

    EPS = mx.finfo(mx.float64).eps
    relative_step = {
        "2-point": EPS**0.5,
        "3-point": EPS**(1/3),
        "cs": EPS**0.5
    }
    f0 = mx.array(1.0)

    for method in methods:
        rel_step = relative_step[method]
        correct_step = mx.array([rel_step,
                                 rel_step * 1.,
                                 rel_step * 1.,
                                 rel_step * mx.abs(x0[3])])

        abs_step = _compute_absolute_step(None, x0, f0, method)
        assert_allclose(abs_step, correct_step)

        sign_x0 = (-x0 >= 0).astype(float) * 2 - 1
        abs_step = _compute_absolute_step(None, -x0, f0, method)
        assert_allclose(abs_step, sign_x0 * correct_step)

    # if a relative step is provided it should be used
    rel_step = mx.array([0.1, 1, 10, 100])
    correct_step = mx.array([rel_step[0] * x0[0],
                             relative_step['2-point'],
                             rel_step[2] * 1.,
                             rel_step[3] * mx.abs(x0[3])])

    abs_step = _compute_absolute_step(rel_step, x0, f0, '2-point')
    assert_allclose(abs_step, correct_step)

    sign_x0 = (-x0 >= 0).astype(float) * 2 - 1
    abs_step = _compute_absolute_step(rel_step, -x0, f0, '2-point')
    assert_allclose(abs_step, sign_x0 * correct_step)
