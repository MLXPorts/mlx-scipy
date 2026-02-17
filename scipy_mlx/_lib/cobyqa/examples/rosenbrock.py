#!/usr/bin/env python3
"""
Minimize the Rosenbrock function subject to simple bounds and randomly
generated linear inequality and equality constraints.
"""
import random

import mlx.core as mx
from cobyqa import minimize
from scipy.optimize import Bounds, LinearConstraint, rosen


if __name__ == "__main__":
    random.seed(0)
    n, m_linear_ub, m_linear_eq = 10, 3, 2

    # Generate an initial guess satisfying the bound constraints.
    bounds = Bounds(-2.048 * mx.ones(n), 2.048 * mx.ones(n))
    x0 = mx.clip(mx.array([random.uniform(-3.0, 3.0) for _ in range(n)]), bounds.lb, bounds.ub)

    # Generate feasible linear inequality and equality constraints.
    x_rand = mx.array([random.uniform(float(bounds.lb[i]), float(bounds.ub[i])) for i in range(n)])
    aub = mx.random.normal((m_linear_ub, n), loc=0.0, scale=1.0)
    bub = mx.matmul(aub, x_rand) + mx.random.uniform(0.0, 1.0, (m_linear_ub,))
    aeq = mx.random.normal((m_linear_eq, n), loc=0.0, scale=1.0)
    beq = mx.matmul(aeq, x_rand)
    constraints = [
        LinearConstraint(aub, -float("inf"), bub),
        LinearConstraint(aeq, beq, beq),
    ]
    res = minimize(rosen, x0, bounds=bounds, constraints=constraints)
    print(res)
