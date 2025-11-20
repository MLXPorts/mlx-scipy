"""
MLX implementation of cosine distribution functions.

Replaces C implementation in _cosine.c with pure MLX code.

The cosine distribution CDF is:
    p = (pi + x + sin(x)) / (2*pi)  for -pi <= x <= pi

The inverse CDF (percent point function) is the inverse of this function.
"""

import mlx.core as mx


# M_PI64 is the 64-bit floating point representation of π
# This exact value is important for numerical precision
# 0x1.921fb54442d18p+1 in C hex float notation = 3.141592653589793
M_PI64 = mx.array(3.141592653589793, dtype=mx.float32)

# Correction term: float64(π - M_PI64) = 1.2246467991473532e-16
PI_CORRECTION = mx.array(1.2246467991473532e-16, dtype=mx.float32)


def _polevl(x, coeffs):
    """
    Evaluate polynomial using Horner's method.

    Matches cephes_polevl_wrap behavior: coefficients are in DECREASING
    degree order (highest degree first).

    For coeffs = [c_n, c_{n-1}, ..., c_1, c_0], evaluates:
    c_n*x^n + c_{n-1}*x^{n-1} + ... + c_1*x + c_0

    Parameters
    ----------
    x : array
        Input value(s)
    coeffs : list of float
        Polynomial coefficients in DECREASING degree order (highest first)

    Returns
    -------
    result : array
        Polynomial evaluated at x
    """
    # Convert coefficients to MLX array with explicit dtype
    # Coefficients are already in correct order for Horner's method
    c = mx.array(coeffs, dtype=x.dtype)

    # Horner's method: p(x) = c[0] + x*(c[1] + x*(c[2] + ...))
    # Use Python loop with explicit MLX array indexing to preserve dtype
    result = c[0:1].squeeze()  # Extract first element as MLX array
    for i in range(1, len(c)):
        coeff = c[i:i+1].squeeze()  # Extract as MLX array, not Python scalar
        result = mx.add(mx.multiply(result, x), coeff)

    return result


def _cosine_cdf_pade_approx_at_neg_pi(x):
    """
    Compute CDF of cosine distribution for x near -π using Pade approximant.

    This avoids loss of precision that occurs in the standard formula
    when x is close to -π.

    Parameters
    ----------
    x : array
        Input values close to but not less than -π

    Returns
    -------
    result : array
        CDF values
    """
    # Numerator coefficients (only non-zero terms in h^3 * P(h^2))
    numer_coeffs = [
        -3.8360369451359084e-08,
        1.0235408442872927e-05,
        -0.0007883197097740538,
        0.026525823848649224
    ]

    # Denominator coefficients
    denom_coeffs = [
        1.6955280904096042e-11,
        6.498171564823105e-09,
        1.4162345851873058e-06,
        0.00020944197182753272,
        0.020281047093125535,
        1.0
    ]

    # Compute h = x + π with high precision
    h = mx.add(mx.add(x, M_PI64), PI_CORRECTION)
    h2 = mx.multiply(h, h)
    h3 = mx.multiply(h2, h)

    # Evaluate polynomials
    numer = mx.multiply(h3, _polevl(h2, numer_coeffs))
    denom = _polevl(h2, denom_coeffs)

    return mx.divide(numer, denom)


def cosine_cdf(x):
    """
    Cumulative distribution function of the cosine distribution.

    The CDF is:
        0                           for x < -π
        (π + x + sin(x))/(2π)      for -π <= x <= π
        1                           for x > π

    Parameters
    ----------
    x : array
        Input values

    Returns
    -------
    result : array
        CDF values at x
    """
    x = mx.array(x, dtype=mx.float32)

    # Initialize result array
    result = mx.zeros_like(x)

    # Case 1: x >= π → return 1
    mask_high = mx.greater_equal(x, M_PI64)
    result = mx.where(mask_high, mx.array(1.0, dtype=mx.float32), result)

    # Case 2: x < -π → return 0
    mask_low = mx.less(x, mx.negative(M_PI64))
    result = mx.where(mask_low, mx.array(0.0, dtype=mx.float32), result)

    # Case 3: x close to -π (x < -1.6) → use Pade approximation
    threshold = mx.array(-1.6, dtype=mx.float32)
    mask_pade = mx.logical_and(
        mx.greater_equal(x, mx.negative(M_PI64)),
        mx.less(x, threshold)
    )
    pade_vals = _cosine_cdf_pade_approx_at_neg_pi(x)
    result = mx.where(mask_pade, pade_vals, result)

    # Case 4: -1.6 <= x < π → use standard formula
    mask_standard = mx.logical_and(
        mx.greater_equal(x, threshold),
        mx.less(x, M_PI64)
    )
    standard_vals = mx.add(
        mx.array(0.5, dtype=mx.float32),
        mx.divide(
            mx.add(x, mx.sin(x)),
            mx.multiply(mx.array(2.0, dtype=mx.float32), M_PI64)
        )
    )
    result = mx.where(mask_standard, standard_vals, result)

    return result


def _p2(t):
    """
    Numerator polynomial for inverse CDF Pade approximation.

    Evaluates polynomial in t (where t = y^2 for input y).
    The result must be multiplied by y to get the full numerator.
    """
    coeffs = [
        -6.8448463845552725e-09,
        3.4900934227012284e-06,
        -0.00030539712907115167,
        0.009350454384541677,
        -0.11602142940208726,
        0.5
    ]
    return _polevl(t, coeffs)


def _q2(t):
    """
    Denominator polynomial for inverse CDF Pade approximation.

    Evaluates polynomial in t (where t = y^2 for input y).
    """
    coeffs = [
        -5.579679571562129e-08,
        1.3728570152788793e-05,
        -0.0008916919927321117,
        0.022927496105281435,
        -0.25287619213750784,
        1.0
    ]
    return _polevl(t, coeffs)


def _poly_approx(s):
    """
    Asymptotic expansion of inverse function at p=0.

    This is related to the inverse Kepler equation with eccentricity e=1.
    Includes terms up to s^13.

    Parameters
    ----------
    s : array
        Input values

    Returns
    -------
    result : array
        Polynomial approximation
    """
    coeffs = [
        1.1911667949082915e-08,
        1.683039183039183e-07,
        2.4930426716141005e-06,  # 43.0/17248000
        3.968253968253968e-05,   # 1.0/25200
        0.0007142857142857143,   # 1.0/1400
        0.016666666666666666,    # 1.0/60
        1.0
    ]

    s2 = mx.multiply(s, s)
    poly_val = _polevl(s2, coeffs)
    return mx.multiply(s, poly_val)


def cosine_invcdf(p):
    """
    Inverse CDF (percent point function) of the cosine distribution.

    Returns x such that cosine_cdf(x) = p.

    Parameters
    ----------
    p : array
        Probability values in [0, 1]

    Returns
    -------
    result : array
        Quantile values
    """
    p = mx.array(p, dtype=mx.float32)

    # Handle invalid inputs
    invalid = mx.logical_or(mx.less(p, mx.array(0.0, dtype=mx.float32)),
                            mx.greater(p, mx.array(1.0, dtype=mx.float32)))

    # Initialize result
    result = mx.where(invalid, mx.array(mx.nan, dtype=mx.float32), mx.array(0.0, dtype=mx.float32))

    # Handle boundary cases (only for valid inputs)
    very_small = mx.logical_and(
        mx.logical_not(invalid),
        mx.less_equal(p, mx.array(1e-48, dtype=mx.float32))
    )
    result = mx.where(very_small, mx.negative(M_PI64), result)

    is_one = mx.equal(p, mx.array(1.0, dtype=mx.float32))
    result = mx.where(is_one, M_PI64, result)

    # Skip computation for boundary/invalid cases
    valid_mask = mx.logical_and(
        mx.logical_not(mx.logical_or(invalid, mx.logical_or(very_small, is_one))),
        mx.array(True, dtype=mx.bool_)
    )

    # Determine sign based on p > 0.5
    sgn = mx.where(mx.greater(p, mx.array(0.5, dtype=mx.float32)),
                   mx.array(-1.0, dtype=mx.float32),
                   mx.array(1.0, dtype=mx.float32))

    # Reflect p if > 0.5
    p_work = mx.where(mx.greater(p, mx.array(0.5, dtype=mx.float32)),
                      mx.subtract(mx.array(1.0, dtype=mx.float32), p),
                      p)

    # Case 1: p < 0.0925 → use asymptotic expansion
    use_asymp = mx.less(p_work, mx.array(0.0925, dtype=mx.float32))
    twelve_pi_p = mx.multiply(mx.multiply(mx.array(12.0, dtype=mx.float32), M_PI64), p_work)
    cbrt_val = mx.power(twelve_pi_p, mx.array(1.0/3.0, dtype=mx.float32))
    x_asymp = mx.subtract(_poly_approx(cbrt_val), M_PI64)

    # Case 2: p >= 0.0925 → use Pade approximation
    y = mx.multiply(M_PI64, mx.subtract(mx.multiply(mx.array(2.0, dtype=mx.float32), p_work),
                                        mx.array(1.0, dtype=mx.float32)))
    y2 = mx.multiply(y, y)
    x_pade = mx.divide(mx.multiply(y, _p2(y2)), _q2(y2))

    # Choose between asymptotic and Pade
    x = mx.where(use_asymp, x_asymp, x_pade)

    # Halley refinement for 0.0018 < p < 0.42
    needs_halley = mx.logical_and(
        mx.greater(p_work, mx.array(0.0018, dtype=mx.float32)),
        mx.less(p_work, mx.array(0.42, dtype=mx.float32))
    )

    # Halley's method iteration:
    # f(x) = pi + x + sin(x) - 2*pi*p
    # f'(x) = 1 + cos(x)
    # f''(x) = -sin(x)
    # x_new = x - 2*f*f'/(2*f'^2 - f*f'')
    two_pi_p = mx.multiply(mx.multiply(mx.array(2.0, dtype=mx.float32), M_PI64), p_work)
    f0 = mx.subtract(mx.add(mx.add(M_PI64, x), mx.sin(x)), two_pi_p)
    f1 = mx.add(mx.array(1.0, dtype=mx.float32), mx.cos(x))
    f2 = mx.negative(mx.sin(x))

    halley_correction = mx.divide(
        mx.multiply(mx.multiply(mx.array(2.0, dtype=mx.float32), f0), f1),
        mx.subtract(mx.multiply(mx.multiply(mx.array(2.0, dtype=mx.float32), f1), f1),
                    mx.multiply(f0, f2))
    )
    x_halley = mx.subtract(x, halley_correction)

    # Apply Halley correction where needed
    x = mx.where(needs_halley, x_halley, x)

    # Apply sign
    x = mx.multiply(sgn, x)

    # Update result for valid cases
    result = mx.where(valid_mask, x, result)

    return result


# Expose these functions similar to how the C code exports them
__all__ = ['cosine_cdf', 'cosine_invcdf']
