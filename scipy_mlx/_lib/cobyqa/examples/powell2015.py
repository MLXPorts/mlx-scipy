#!/usr/bin/env python3
"""
Solve Example (6.7)--(6.8) of [1]_.

References
----------
.. [1] M. J. D. Powell. On fast trust region methods for quadratic models with
   linear constraints. *Math. Program. Comput.*, 7(3):237--267, 2015.
   `doi:10.1007/s12532-015-0084-4
   <https://doi.org/10.1007/s12532-015-0084-4>`_.
"""
from contextlib import suppress

import random

import mlx.core as mx
from cobyqa import minimize
from scipy.optimize import Bounds, LinearConstraint


def fun(x):
    f = mx.array(0.0)
    for i in range(1, x.size // 2):
        for j in range(i):
            dx = x[2 * i] - x[2 * j]
            dy = x[2 * i + 1] - x[2 * j + 1]
            norm = mx.sqrt(mx.square(dx) + mx.square(dy))
            f = mx.add(f, mx.where(norm > 1e-3, mx.divide(mx.array(1.0), norm), mx.array(1e3)))
    f = mx.divide(f, mx.array(float(x.size**2)))
    return f


def _plot_points(x, title=None):
    with suppress(ImportError):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(dpi=300)
        ax.plot([0.0, 0.0, 2.0, 0.0], [0.0, 2.0, 0.0, 0.0], color="black")
        ax.scatter(x[::2], x[1::2], s=25, color="black")
        ax.set_aspect("equal", "box")
        ax.axis("off")
        if title is not None:
            ax.set_title(f"{title.strip()} ($n = {x.size}$)", fontsize=20)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    random.seed(0)
    n = 80  # must be even

    aub = mx.zeros((n // 2, n))
    bub = 2.0 * mx.ones(n // 2)
    x0 = mx.zeros(n)
    for i in range(n // 2):
        aub[i, [2 * i, 2 * i + 1]] = 1.0
        x0_even = random.uniform(0.0, 2.0)
        x0_odd = random.uniform(0.0, 2.0)
        if x0_even + x0_odd > 2.0:
            x0_even = 2.0 - x0_even
            x0_odd = 2.0 - x0_odd
        x0[2 * i] = x0_even
        x0[2 * i + 1] = x0_odd
    _plot_points(x0, "Initial points")

    res = minimize(
        fun,
        x0,
        bounds=Bounds(mx.zeros(n), float("inf")),
        constraints=LinearConstraint(aub, -float("inf"), bub),
        options={"disp": True},
    )
    print(res)
    _plot_points(res.x, "Final points")
