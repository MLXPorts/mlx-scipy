"""
MLX stub for SciPy's compiled special-function ufuncs.

Upstream SciPy provides these as compiled ufuncs. In this repository we aim
for an MLX-first Python implementation; however, a full port of SciPy's special
functions is out of scope for this conversion pass.

This module therefore provides placeholders for the subset of symbols imported
at import-time by other `scipy_mlx.special` modules. Calling these functions
will raise `NotImplementedError`.
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(f"scipy_mlx.special.{name} is not yet implemented for MLX.")
    _fn.__name__ = name
    return _fn


# Public-ish ufuncs imported by scipy_mlx.special._basic
mathieu_a = _unimplemented("mathieu_a")
mathieu_b = _unimplemented("mathieu_b")
iv = _unimplemented("iv")
jv = _unimplemented("jv")
gamma = _unimplemented("gamma")
rgamma = _unimplemented("rgamma")
psi = _unimplemented("psi")
hyp2f1 = _unimplemented("hyp2f1")
loggamma = _unimplemented("loggamma")
beta = _unimplemented("beta")
betaln = _unimplemented("betaln")


def _lanczos_gammaln(z):
    """Lanczos approximation of log(Gamma(z)) for z >= 0.5 (real inputs).

    This is a lightweight MLX implementation used to unblock portions of
    `scipy_mlx.stats` and `scipy_mlx.special` that require `gammaln`/`betaln`.
    It is not a full replacement for SciPy's compiled special functions.
    """
    z = mx.array(z)

    # Coefficients for g=7, n=9 (Numerical Recipes / Wikipedia).
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    g = mx.array(7.0)

    z1 = mx.subtract(z, mx.array(1.0))
    x = mx.array(p[0])
    for i, coeff in enumerate(p[1:], start=1):
        x = mx.add(x, mx.divide(mx.array(coeff), mx.add(z1, mx.array(float(i)))))

    t = mx.add(z1, mx.add(g, mx.array(0.5)))
    log_sqrt_2pi = mx.array(0.91893853320467274178)  # 0.5*log(2*pi)
    return mx.add(
        mx.add(
            log_sqrt_2pi,
            mx.subtract(mx.multiply(mx.add(z1, mx.array(0.5)), mx.log(t)), t),
        ),
        mx.log(x),
    )


def gammaln(x):
    """Natural logarithm of the absolute value of the Gamma function.

    Notes
    -----
    This is an MLX approximation intended for broad compatibility with SciPy's
    API surface. Accuracy and edge-case behavior will differ from upstream for
    negative/complex inputs.
    """
    x = mx.array(x)

    # Reflection for x < 0.5: gammaln(x) = log(pi) - log(|sin(pi*x)|) - gammaln(1-x)
    mask = mx.less(x, mx.array(0.5))
    one_minus_x = mx.subtract(mx.array(1.0), x)

    main = _lanczos_gammaln(x)
    refl = mx.subtract(
        mx.subtract(mx.log(mx.array(float(mx.pi))), mx.log(mx.abs(mx.sin(mx.multiply(mx.array(float(mx.pi)), x))))),
        _lanczos_gammaln(one_minus_x),
    )

    return mx.where(mask, refl, main)


def betaln(a, b):
    a = mx.array(a)
    b = mx.array(b)
    return mx.subtract(mx.add(gammaln(a), gammaln(b)), gammaln(mx.add(a, b)))


def ndtr(x):
    """Standard normal cumulative distribution function."""
    x = mx.array(x)
    sqrt2 = mx.sqrt(mx.array(2.0))
    return mx.multiply(mx.array(0.5), mx.add(mx.array(1.0), mx.erf(mx.divide(x, sqrt2))))


def ndtri(p):
    """Inverse of `ndtr` (standard normal percentile function)."""
    p = mx.array(p)
    sqrt2 = mx.sqrt(mx.array(2.0))
    out = mx.multiply(sqrt2, mx.erfinv(mx.subtract(mx.multiply(mx.array(2.0), p), mx.array(1.0))))

    out = mx.where(mx.equal(p, mx.array(0.0)), mx.array(float("-inf")), out)
    out = mx.where(mx.equal(p, mx.array(1.0)), mx.array(float("inf")), out)
    invalid = mx.logical_or(mx.less(p, mx.array(0.0)), mx.greater(p, mx.array(1.0)))
    out = mx.where(invalid, mx.array(float("nan")), out)
    return out


def erf(x):
    return mx.erf(mx.array(x))


def erfinv(x):
    return mx.erfinv(mx.array(x))


def rel_entr(x, y):
    x = mx.array(x)
    y = mx.array(y)
    zero = mx.equal(x, mx.array(0))
    pos = mx.logical_and(mx.greater(x, 0), mx.greater(y, 0))
    inf = mx.array(float("inf"))
    out = mx.where(
        pos,
        mx.multiply(x, mx.subtract(mx.log(x), mx.log(y))),
        inf,
    )
    out = mx.where(
        mx.logical_and(zero, mx.greater_equal(y, mx.array(0))),
        mx.zeros_like(out),
        out,
    )
    return out


def entr(x):
    x = mx.array(x)
    zero = mx.equal(x, mx.array(0))
    pos = mx.greater(x, mx.array(0))
    neg = mx.less(x, mx.array(0))
    out = mx.where(pos, mx.multiply(mx.array(-1.0), mx.multiply(x, mx.log(x))), mx.array(float("-inf")))
    out = mx.where(zero, mx.zeros_like(out), out)
    out = mx.where(neg, mx.array(float("-inf")), out)
    return out


def xlogy(x, y):
    x = mx.array(x)
    y = mx.array(y)
    out = mx.multiply(x, mx.log(y))
    return mx.where(mx.equal(x, mx.array(0)), mx.zeros_like(out), out)


def xlog1py(x, y):
    x = mx.array(x)
    y = mx.array(y)
    out = mx.multiply(x, mx.log1p(y))
    return mx.where(mx.equal(x, mx.array(0)), mx.zeros_like(out), out)


hankel1 = _unimplemented("hankel1")
hankel2 = _unimplemented("hankel2")
yv = _unimplemented("yv")
kv = _unimplemented("kv")
poch = _unimplemented("poch")
binom = _unimplemented("binom")
_stirling2_inexact = _unimplemented("_stirling2_inexact")

# Internal helpers imported by other pure-python wrappers
_lambertw = _unimplemented("_lambertw")
_ellip_harm = _unimplemented("_ellip_harm")
_spherical_jn = _unimplemented("_spherical_jn")
_spherical_yn = _unimplemented("_spherical_yn")
_spherical_in = _unimplemented("_spherical_in")
_spherical_kn = _unimplemented("_spherical_kn")


__all__ = [
    "mathieu_a",
    "mathieu_b",
    "iv",
    "jv",
    "gamma",
    "rgamma",
    "psi",
    "hyp2f1",
    "gammaln",
    "loggamma",
    "beta",
    "betaln",
    "ndtr",
    "ndtri",
    "erf",
    "erfinv",
    "rel_entr",
    "entr",
    "xlogy",
    "xlog1py",
    "hankel1",
    "hankel2",
    "yv",
    "kv",
    "poch",
    "binom",
    "_stirling2_inexact",
    "_lambertw",
    "_ellip_harm",
    "_spherical_jn",
    "_spherical_yn",
    "_spherical_in",
    "_spherical_kn",
]
