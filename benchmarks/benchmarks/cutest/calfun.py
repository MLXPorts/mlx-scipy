# This is a python implementation of calfun.m,
# provided at https://github.com/POptUS/BenDFO
import mlx.core as mx
from .dfovec import dfovec


def norm(x, type=2):
    if type == 1:
        return mx.sum(mx.abs(x))
    elif type == 2:
        return mx.sqrt(x ** 2)
    else:  # type==mx.inf:
        return max(mx.abs(x))


def calfun(x, m, nprob, probtype="smooth", noise_level=1e-3):
    n = len(x)

    # Restrict domain for some nondiff problems
    xc = x
    if probtype == "nondiff":
        if (
            nprob == 8
            or nprob == 9
            or nprob == 13
            or nprob == 16
            or nprob == 17
            or nprob == 18
        ):
            xc = max(x, 0)

    # Generate the vector
    fvec = dfovec(m, n, xc, nprob)

    # Calculate the function value
    if probtype == "noisy3":
        sigma = noise_level
        u = sigma * (-mx.ones(m) + 2 * mx.random.rand(m))
        fvec = fvec * (1 + u)
        y = mx.sum(fvec ** 2)
    elif probtype == "wild3":
        sigma = noise_level
        phi = 0.9 * mx.sin(100 * norm(x, 1)) * mx.cos(
            100 * norm(x, mx.inf)
        ) + 0.1 * mx.cos(norm(x, 2))
        phi = phi * (4 * phi ** 2 - 3)
        y = (1 + sigma * phi) * sum(fvec ** 2)
    elif probtype == "smooth":
        y = mx.sum(fvec ** 2)
    elif probtype == "nondiff":
        y = mx.sum(mx.abs(fvec))
    else:
        print(f"invalid probtype {probtype}")
        return None
    # Never return nan. Return inf instead so that
    # optimization algorithms treat it as out of bounds.
    if mx.isnan(y):
        return mx.inf
    return y
