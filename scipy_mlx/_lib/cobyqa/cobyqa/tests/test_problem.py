import mlx.core as mx
import pytest
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from scipy.optimize._minimize import standardize_constraints

from ..problem import (
    ObjectiveFunction,
    BoundConstraints,
    LinearConstraints,
    NonlinearConstraints,
    Problem,
)
from ..settings import PRINT_OPTIONS
from ..utils import CallbackSuccess, mx_printoptions, mx_r_


class BaseTest:

    @staticmethod
    def rosen(x, c=100.0):
        x = mx.asarray(x)
        return mx.sum(c * (x[1:] - x[:-1] ** 2.0) ** 2.0
                      + (1.0 - x[:-1]) ** 2.0)

    class Rosen:

        def __call__(self, x):
            return BaseTest.rosen(x)


class TestObjectiveFunction(BaseTest):

    def test_simple(self, capsys):
        obj = ObjectiveFunction(self.rosen, False, True)
        x = [1.5, 1.5]
        assert obj.n_eval == 0
        assert obj(x) == self.rosen(x)
        assert obj.n_eval == 1
        assert obj.name == "rosen"
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_args(self):
        obj = ObjectiveFunction(self.rosen, False, True, 2.0)
        x = [1.5, 1.5]
        assert obj(x) == self.rosen(x, 2.0)

    def test_none(self):
        obj = ObjectiveFunction(None, False, True)
        assert obj([1.5, 1.5]) == 0.0
        assert obj.n_eval == 0
        assert obj.name == ""

    def test_wrapper(self):
        obj = ObjectiveFunction(self.Rosen(), False, True)
        assert obj.name == "fun"

    def test_verbose(self, capsys):
        obj = ObjectiveFunction(self.rosen, True, True)
        x = mx.array([1.5, 1.5])
        obj(x)
        captured = capsys.readouterr()
        with mx_printoptions(**PRINT_OPTIONS):
            assert captured.out == f"rosen({x}) = {self.rosen(x)}\n"


class TestBoundConstraints:

    def test_simple(self):
        bounds = Bounds([0.0, 0.0], [1.0, 1.0])
        constraints = BoundConstraints(bounds)
        assert mx.array_equal(constraints.xl, bounds.lb)
        assert mx.array_equal(constraints.xu, bounds.ub)
        assert constraints.m == 4
        assert constraints.is_feasible
        assert constraints.maxcv([0.5, 0.5]) == 0.0
        assert constraints.maxcv(constraints.xl) == 0.0
        assert constraints.maxcv(constraints.xu) == 0.0
        x = [2.0, 2.0]
        assert mx.all(constraints.project(x) >= constraints.xl)
        assert mx.all(constraints.project(x) <= constraints.xu)

    def test_nan(self):
        bounds = Bounds([mx.nan, 0.0], [1.0, 1.0])
        constraints = BoundConstraints(bounds)
        assert mx.array_equal(constraints.xl, [-mx.inf, 0.0])
        assert constraints.m == 3
        bounds = Bounds([0.0, 0.0], [1.0, mx.nan])
        constraints = BoundConstraints(bounds)
        assert mx.array_equal(constraints.xu, [1.0, mx.inf])
        assert constraints.m == 3

    def test_infeasible(self):
        bounds = Bounds([2.0, 0.0], [1.0, 1.0])
        constraints = BoundConstraints(bounds)
        assert not constraints.is_feasible
        x = [2.0, 2.0]
        assert mx.array_equal(constraints.project(x), x)
        assert mx.allclose(constraints.maxcv(x), 1.0, atol=1e-15)


class TestLinearConstraints:

    def test_simple(self):
        linear_constraints = [
            LinearConstraint([[1.0, 1.0]], [0.0], [1.0]),
            LinearConstraint([[2.0, 1.0]], [1.0], [1.0]),
        ]
        constraints = LinearConstraints(linear_constraints, 2, True)
        assert mx.array_equal(
            constraints.a_ub,
            [[1.0, 1.0], [-1.0, -1.0]],
        )
        assert mx.array_equal(constraints.b_ub, [1.0, 0.0])
        assert mx.array_equal(constraints.a_eq, [[2.0, 1.0]])
        assert mx.array_equal(constraints.b_eq, [1.0])
        assert constraints.m_ub == 2
        assert constraints.m_eq == 1
        assert mx.allclose(
            constraints.maxcv([0.5, 0.0]),
            0.0,
            atol=1e-15,
        )

    def test_nan(self):
        linear_constraints = [LinearConstraint([[1.0, mx.nan]], [0.0], [1.0])]
        constraints = LinearConstraints(linear_constraints, 2, True)
        assert mx.array_equal(
            constraints.a_ub,
            [[1.0, 0.0], [-1.0, 0.0]],
        )
        assert mx.array_equal(constraints.b_ub, [1.0, 0.0])
        assert constraints.m_ub == 2
        assert constraints.m_eq == 0
        linear_constraints = [LinearConstraint([[1.0, 1.0]], [mx.nan], [1.0])]
        constraints = LinearConstraints(linear_constraints, 2, True)
        assert mx.array_equal(constraints.a_ub, [[1.0, 1.0]])
        assert mx.array_equal(constraints.b_ub, [1.0])
        assert constraints.m_ub == 1
        assert constraints.m_eq == 0
        linear_constraints = [LinearConstraint([[1.0, 1.0]], [0.0], [mx.nan])]
        constraints = LinearConstraints(linear_constraints, 2, True)
        assert mx.array_equal(constraints.a_ub, [[-1.0, -1.0]])
        assert mx.array_equal(constraints.b_ub, [0.0])
        assert constraints.m_ub == 1
        assert constraints.m_eq == 0

    def test_inf(self):
        linear_constraints = [
            LinearConstraint([[1.0, 1.0]], [0.0], [mx.inf]),
        ]
        constraints = LinearConstraints(linear_constraints, 2, True)
        assert mx.array_equal(constraints.a_ub, [[-1.0, -1.0]])
        assert mx.array_equal(constraints.b_ub, [0.0])
        assert constraints.m_ub == 1


class TestNonlinearConstraint:

    def test_simple(self):
        nonlinear_constraints = [
            NonlinearConstraint(mx.cos, [-0.5, -0.5], [0.0, 0.0]),
            NonlinearConstraint(mx.sin, [1.0, 1.0], [1.0, 1.0]),
            NonlinearConstraint(mx.tan, -mx.inf, mx.inf),
            NonlinearConstraint(lambda x: mx.inner(x, x) - 1.0, 0, 0),
        ]
        constraints = NonlinearConstraints(nonlinear_constraints, False, True)
        assert constraints.n_eval == 0
        x = [0.5, 0.5]
        c_ub, c_eq = constraints(x)
        assert mx.allclose(
            c_ub,
            mx_r_([-0.5 - mx.cos(x), mx.cos(x)]),
            atol=1e-15,
        )
        assert mx.allclose(
            c_eq,
            mx_r_([mx.sin(x) - 1.0, mx.inner(x, x) - 1.0]),
            atol=1e-15,
        )
        assert constraints.n_eval == 1
        assert constraints.m_ub == 4
        assert constraints.m_eq == 3
        assert mx.allclose(
            constraints.maxcv(x, c_ub, c_eq),
            max(mx.max(mx.abs(c_eq)), mx.max(c_ub)),
            atol=1e-15,
        )
        assert mx.array_equal(
            constraints.maxcv(x, c_ub, c_eq),
            constraints.maxcv(x),
        )

    def test_args(self):
        nonlinear_constraints = [
            {"fun": lambda x, c: c * mx.cos(x),
             "type": "ineq", "args": (2.0,)},
            {"fun": lambda x, c: c * mx.sin(x),
             "type": "eq", "args": (2.0,)},
        ]
        x = [0.5, 0.5]
        nonlinear_constraints = standardize_constraints(
            nonlinear_constraints, x, "new"
        )
        constraints = NonlinearConstraints(nonlinear_constraints, False, True)
        c_ub, c_eq = constraints(x)
        assert mx.allclose(
            c_ub,
            mx_r_([-2.0 * mx.cos(x)]),
            atol=1e-15,
        )
        assert mx.allclose(
            c_eq,
            mx_r_([2.0 * mx.sin(x)]),
            atol=1e-15,
        )

    def test_verbose(self, capsys):
        nonlinear_constraints = [
            NonlinearConstraint(mx.cos, [-0.5, -0.5], [0.0, 0.0]),
        ]
        constraints = NonlinearConstraints(nonlinear_constraints, True, True)
        x = mx.array([1.5, 1.5])
        constraints(x)
        captured = capsys.readouterr()
        with mx_printoptions(**PRINT_OPTIONS):
            assert captured.out == f"cos({x}) = {mx.cos(x)}\n"

    def test_exceptions(self):
        nonlinear_constraints = [
            NonlinearConstraint(mx.cos, [-0.5, -0.5], [0.0, 0.0]),
            NonlinearConstraint(mx.sin, [1.0, 1.0], [1.0, 1.0]),
        ]
        constraints = NonlinearConstraints(nonlinear_constraints, False, True)
        with pytest.raises(ValueError):
            constraints.m_ub
        with pytest.raises(ValueError):
            constraints.m_eq


class TestProblem(BaseTest):

    def test_simple(self):
        obj = ObjectiveFunction(self.rosen, False, True)
        bounds = BoundConstraints(Bounds([0.0, 0.0], [1.0, 1.0]))
        linear_constraints = LinearConstraints(
            [LinearConstraint([[1.0, 1.0]], [0.0], [1.0])],
            2,
            True,
        )
        nonlinear_constraints = NonlinearConstraints(
            [NonlinearConstraint(mx.cos, [-0.5, -0.5], [0.0, 0.0])],
            False,
            True,
        )
        problem = Problem(
            obj,
            [0.0, 0.0],
            bounds,
            linear_constraints,
            nonlinear_constraints,
            None,
            0.0,
            False,
            False,
            0,
            1,
            True,
        )
        assert problem.n_eval == 0
        x = [0.5, 0.5]
        fun, c_ub, c_eq = problem(x)
        assert mx.allclose(fun, self.rosen(x), atol=1e-15)
        assert mx.allclose(
            c_ub,
            mx_r_([-0.5 - mx.cos(x), mx.cos(x)]),
            atol=1e-15,
        )
        assert c_eq.size == 0
        assert problem.n == 2
        assert problem.n_orig == 2
        assert mx.array_equal(problem.x0, [0.0, 0.0])
        assert problem.n_eval == 1
        assert problem.fun_name == "rosen"
        assert mx.array_equal(problem.bounds.xl, [0.0, 0.0])
        assert mx.array_equal(problem.bounds.xu, [1.0, 1.0])
        assert mx.array_equal(
            problem.linear.a_ub,
            [[1.0, 1.0], [-1.0, -1.0]],
        )
        assert mx.array_equal(problem.linear.b_ub, [1.0, 0.0])
        assert problem.linear.a_eq.size == 0
        assert problem.linear.b_eq.size == 0
        assert problem.m_bounds == 4
        assert problem.m_linear_ub == 2
        assert problem.m_linear_eq == 0
        assert problem.m_nonlinear_ub == 4
        assert problem.m_nonlinear_eq == 0
        assert problem.fun_history.size == 0
        assert problem.maxcv_history.size == 0
        assert problem.type == "nonlinearly constrained"
        assert not problem.is_feasibility
        assert mx.allclose(
            problem.maxcv(x),
            max(bounds.maxcv(x), linear_constraints.maxcv(x), mx.max(c_ub)),
            atol=1e-15,
        )
        x_best, fun_best, maxcv_best = problem.best_eval(0.0)
        assert mx.array_equal(x_best, x)
        assert mx.allclose(fun_best, self.rosen(x), atol=1e-15)
        assert mx.allclose(maxcv_best, problem.maxcv(x), atol=1e-15)

    def test_scale(self):
        obj = ObjectiveFunction(self.rosen, False, True)
        bounds = BoundConstraints(Bounds([0.0, 0.0], [1.0, 1.0]))
        linear_constraints = LinearConstraints(
            [LinearConstraint([[1.0, 1.0]], [0.0], [1.0])],
            2,
            True,
        )
        nonlinear_constraints = NonlinearConstraints(
            [NonlinearConstraint(mx.cos, [-0.5, -0.5], [0.0, 0.0])],
            False,
            True,
        )
        problem = Problem(
            obj,
            [0.0, 0.0],
            bounds,
            linear_constraints,
            nonlinear_constraints,
            None,
            0.0,
            True,
            False,
            0,
            1,
            True,
        )
        assert mx.array_equal(problem.bounds.xl, [-1.0, -1.0])
        assert mx.array_equal(problem.bounds.xu, [1.0, 1.0])
        assert mx.allclose(problem.x0, [-1.0, -1.0], atol=1e-15)
        assert mx.allclose(
            problem.linear.a_ub,
            [[0.5, 0.5], [-0.5, -0.5]],
            atol=1e-15,
        )
        assert mx.allclose(problem.linear.b_ub, [0.0, 1.0], atol=1e-15)
        x = mx.array([0.5, 0.5])
        fun, c_ub, _ = problem(2.0 * x - 1.0)
        assert mx.allclose(fun, self.rosen(x), atol=1e-15)
        assert mx.allclose(
            c_ub,
            mx_r_([-0.5 - mx.cos(x), mx.cos(x)]),
            atol=1e-15,
        )

    def test_barrier(self):
        obj = ObjectiveFunction(lambda x: mx.nan, False, True)
        bounds = BoundConstraints(Bounds([0.0, 0.0], [1.0, 1.0]))
        linear_constraints = LinearConstraints([], 2, True)
        nonlinear_constraints = NonlinearConstraints(
            [
                NonlinearConstraint(lambda x: mx.nan, [0.0], [1.0]),
                NonlinearConstraint(lambda x: mx.nan, [0.0], [0.0]),
            ],
            False,
            True,
        )
        problem = Problem(
            obj,
            [0.0, 0.0],
            bounds,
            linear_constraints,
            nonlinear_constraints,
            None,
            0.0,
            False,
            False,
            0,
            1,
            True,
        )
        fun, c_ub, c_eq = problem([0.5, 0.5])
        assert mx.isfinite(fun)
        assert mx.all(mx.isfinite(c_ub))
        assert mx.all(mx.isfinite(c_eq))

    def test_history(self):
        obj = ObjectiveFunction(self.rosen, False, True)
        bounds = BoundConstraints(Bounds([0.0, 0.0], [1.0, 1.0]))
        linear_constraints = LinearConstraints([], 2, True)
        nonlinear_constraints = NonlinearConstraints(
            [NonlinearConstraint(mx.cos, [-0.5, -0.5], [0.0, 0.0])],
            False,
            True,
        )
        problem = Problem(
            obj,
            [0.0, 0.0],
            bounds,
            linear_constraints,
            nonlinear_constraints,
            None,
            0.0,
            False,
            True,
            2,
            1,
            True,
        )
        x = [0.5, 0.5]
        problem(x)
        assert mx.array_equal(problem.fun_history, [self.rosen(x)])
        assert mx.allclose(
            problem.maxcv_history,
            [nonlinear_constraints.maxcv(x)],
            atol=1e-15,
        )
        problem(x)
        assert mx.array_equal(
            problem.fun_history,
            2 * [self.rosen(x)],
        )
        assert mx.allclose(
            problem.maxcv_history,
            2 * [nonlinear_constraints.maxcv(x)],
            atol=1e-15,
        )
        problem(x)
        assert mx.array_equal(
            problem.fun_history,
            2 * [self.rosen(x)],
        )
        assert mx.allclose(
            problem.maxcv_history,
            2 * [nonlinear_constraints.maxcv(x)],
            atol=1e-15,
        )

    def test_filter(self):
        obj = ObjectiveFunction(self.rosen, False, True)
        bounds = BoundConstraints(Bounds([0.0, 0.0], [1.0, 1.0]))
        linear_constraints = LinearConstraints(
            [LinearConstraint([[1.0, 1.0]], [1.0], [1.0])],
            2,
            True,
        )
        nonlinear_constraints = NonlinearConstraints([], False, True)
        problem = Problem(
            obj,
            [0.0, 0.0],
            bounds,
            linear_constraints,
            nonlinear_constraints,
            None,
            0.0,
            False,
            False,
            0,
            1,
            True,
        )
        problem([1.0, 1.0])
        x, _, _ = problem.best_eval(1e3)
        assert mx.allclose(x, [1.0, 1.0], atol=1e-15)
        problem([0.25, 0.75])
        x, _, _ = problem.best_eval(1e3)
        assert mx.allclose(x, [0.25, 0.75], atol=1e-15)
        problem([0.5, 0.5])
        x, _, _ = problem.best_eval(1e3)
        assert mx.allclose(x, [0.5, 0.5], atol=1e-15)

    def test_callback(self):
        obj = ObjectiveFunction(self.rosen, False, True)
        bounds = BoundConstraints(Bounds([0.0, 0.0], [1.0, 1.0]))
        linear_constraints = LinearConstraints([], 2, True)
        nonlinear_constraints = NonlinearConstraints([], False, True)

        def callback(intermediate_result):
            if intermediate_result.fun < 1e-3:
                raise StopIteration

        problem = Problem(
            obj,
            [0.0, 0.0],
            bounds,
            linear_constraints,
            nonlinear_constraints,
            callback,
            0.0,
            False,
            False,
            0,
            2,
            True,
        )
        problem([0.0, 0.0])
        with pytest.raises(CallbackSuccess):
            problem([1.0, 1.0])

        def callback(xk):
            if mx.all(xk > 0.5):
                raise CallbackSuccess

        problem = Problem(
            obj,
            [0.0, 0.0],
            bounds,
            linear_constraints,
            nonlinear_constraints,
            callback,
            0.0,
            False,
            False,
            0,
            2,
            True,
        )
        problem([0.0, 0.0])
        with pytest.raises(CallbackSuccess):
            problem([1.0, 1.0])

    def test_type(self):
        obj = ObjectiveFunction(self.rosen, False, True)
        bounds = BoundConstraints(Bounds(2 * [-mx.inf], 2 * [mx.inf]))
        linear_constraints = LinearConstraints([], 2, True)
        nonlinear_constraints = NonlinearConstraints([], False, True)
        problem = Problem(
            obj,
            [0.0, 0.0],
            bounds,
            linear_constraints,
            nonlinear_constraints,
            None,
            0.0,
            False,
            False,
            0,
            1,
            True,
        )
        assert problem.type == "nonlinearly constrained"
        problem(problem.x0)
        assert problem.type == "unconstrained"
        bounds = BoundConstraints(Bounds([0.0, 0.0], [1.0, 1.0]))
        problem = Problem(
            obj,
            [0.0, 0.0],
            bounds,
            linear_constraints,
            nonlinear_constraints,
            None,
            0.0,
            False,
            False,
            0,
            1,
            True,
        )
        problem(problem.x0)
        assert problem.type == "bound-constrained"
        linear_constraints = LinearConstraints(
            [LinearConstraint([[1.0, 1.0]], [0.0], [1.0])],
            2,
            True,
        )
        problem = Problem(
            obj,
            [0.0, 0.0],
            bounds,
            linear_constraints,
            nonlinear_constraints,
            None,
            0.0,
            False,
            False,
            0,
            1,
            True,
        )
        problem(problem.x0)
        assert problem.type == "linearly constrained"

    def test_feasibility_problem(self):
        obj = ObjectiveFunction(None, False, True)
        bounds = BoundConstraints(Bounds(2 * [-mx.inf], 2 * [mx.inf]))
        linear_constraints = LinearConstraints([], 2, True)
        nonlinear_constraints = NonlinearConstraints([], False, True)
        problem = Problem(
            obj,
            [0.0, 0.0],
            bounds,
            linear_constraints,
            nonlinear_constraints,
            None,
            0.0,
            False,
            False,
            0,
            1,
            True,
        )
        assert problem.is_feasibility
        obj = ObjectiveFunction(self.rosen, False, True)
        problem = Problem(
            obj,
            [0.0, 0.0],
            bounds,
            linear_constraints,
            nonlinear_constraints,
            None,
            0.0,
            False,
            False,
            0,
            1,
            True,
        )
        assert not problem.is_feasibility

    def test_best_eval(self):
        obj = ObjectiveFunction(self.rosen, False, True)
        bounds = BoundConstraints(Bounds([0.0, 0.0], [1.0, 1.0]))
        linear_constraints = LinearConstraints(
            [LinearConstraint([[1.0, 1.0]], [0.0], [1.0])],
            2,
            True,
        )
        nonlinear_constraints = NonlinearConstraints(
            [NonlinearConstraint(mx.cos, [-0.5, -0.5], [0.0, 0.0])],
            False,
            True,
        )
        problem = Problem(
            obj,
            [0.0, 0.0],
            bounds,
            linear_constraints,
            nonlinear_constraints,
            None,
            0.0,
            False,
            False,
            0,
            2,
            True,
        )
        x, _, _ = problem.best_eval(1e3)
        assert mx.array_equal(x, [0.0, 0.0])

    def test_exceptions(self):
        obj = ObjectiveFunction(self.rosen, False, True)
        bounds = BoundConstraints(Bounds([0.0, 0.0], [1.0, 1.0]))
        linear_constraints = LinearConstraints([], 2, True)
        nonlinear_constraints = NonlinearConstraints([], False, True)
        with pytest.raises(TypeError):
            Problem(
                obj,
                [0.0, 0.0],
                bounds,
                linear_constraints,
                nonlinear_constraints,
                1.0,
                0.0,
                False,
                False,
                0,
                1,
                True,
            )
        wrong_bounds = BoundConstraints(Bounds([0.0], [1.0]))
        with pytest.raises(ValueError):
            Problem(
                obj,
                [0.0, 0.0],
                wrong_bounds,
                linear_constraints,
                nonlinear_constraints,
                None,
                0.0,
                False,
                False,
                0,
                1,
                True,
            )
        wrong_linear_constraints = LinearConstraints(
            [LinearConstraint(1.0, 1.0, 1.0)],
            1,
            True,
        )
        with pytest.raises(ValueError):
            Problem(
                obj,
                [0.0, 0.0],
                bounds,
                wrong_linear_constraints,
                nonlinear_constraints,
                None,
                0.0,
                False,
                False,
                0,
                1,
                True,
            )
