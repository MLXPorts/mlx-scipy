# Compute the two-sided one-sample Kolmogorov-Smirnov Prob(Dn <= d) where:
#    D_n = sup_x{|F_n(x) - F(x)|},
#    F_n(x) is the empirical CDF for a sample of size n {x_i: i=1,...,n},
#    F(x) is the CDF of a probability distribution.
#
# Exact methods:
# Prob(D_n >= d) can be computed via a matrix algorithm of Durbin[1]
#   or a recursion algorithm due to Pomeranz[2].
# Marsaglia, Tsang & Wang[3] gave a computation-efficient way to perform
#   the Durbin algorithm.
#   D_n >= d <==>  D_n+ >= d or D_n- >= d (the one-sided K-S statistics), hence
#   Prob(D_n >= d) = 2*Prob(D_n+ >= d) - Prob(D_n+ >= d and D_n- >= d).
#   For d > 0.5, the latter intersection probability is 0.
#
# Approximate methods:
# For d close to 0.5, ignoring that intersection term may still give a
#   reasonable approximation.
# Li-Chien[4] and Korolyuk[5] gave an asymptotic formula extending
# Kolmogorov's initial asymptotic, suitable for large d. (See
#   scipy.special.kolmogorov for that asymptotic)
# Pelz-Good[6] used the functional equation for Jacobi theta functions to
#   transform the Li-Chien/Korolyuk formula produce a computational formula
#   suitable for small d.
#
# Simard and L'Ecuyer[7] provided an algorithm to decide when to use each of
#   the above approaches and it is that which is used here.
#
# Other approaches:
# Carvalho[8] optimizes Durbin's matrix algorithm for large values of d.
# Moscovich and Nadler[9] use FFTs to compute the convolutions.

# References:
# [1] Durbin J (1968).
#     "The Probability that the Sample Distribution Function Lies Between Two
#     Parallel Straight Lines."
#     Annals of Mathematical Statistics, 39, 398-411.
# [2] Pomeranz J (1974).
#     "Exact Cumulative Distribution of the Kolmogorov-Smirnov Statistic for
#     Small Samples (Algorithm 487)."
#     Communications of the ACM, 17(12), 703-704.
# [3] Marsaglia G, Tsang WW, Wang J (2003).
#     "Evaluating Kolmogorov's Distribution."
#     Journal of Statistical Software, 8(18), 1-4.
# [4] LI-CHIEN, C. (1956).
#     "On the exact distribution of the statistics of A. N. Kolmogorov and
#     their asymptotic expansion."
#     Acta Matematica Sinica, 6, 55-81.
# [5] KOROLYUK, V. S. (1960).
#     "Asymptotic analysis of the distribution of the maximum deviation in
#     the Bernoulli scheme."
#     Theor. Probability Appl., 4, 339-366.
# [6] Pelz W, Good IJ (1976).
#     "Approximating the Lower Tail-areas of the Kolmogorov-Smirnov One-sample
#     Statistic."
#     Journal of the Royal Statistical Society, Series B, 38(2), 152-156.
#  [7] Simard, R., L'Ecuyer, P. (2011)
# 	  "Computing the Two-Sided Kolmogorov-Smirnov Distribution",
# 	  Journal of Statistical Software, Vol 39, 11, 1-18.
#  [8] Carvalho, Luis (2015)
#     "An Improved Evaluation of Kolmogorov's Distribution"
#     Journal of Statistical Software, Code Snippets; Vol 65(3), 1-8.
#  [9] Amit Moscovich, Boaz Nadler (2017)
#     "Fast calculation of boundary crossing probabilities for Poisson
#     processes",
#     Statistics & Probability Letters, Vol 123, 177-182.


import mlx.core as mx
import scipy.special
import scipy.special._ufuncs as scu
from scipy.stats._finite_differences import _derivative

_E128 = 128
_EP128 = mx.ldexp(mx.longdouble(1), _E128)
_EM128 = mx.ldexp(mx.longdouble(1), -_E128)

_SQRT2PI = mx.sqrt(2 * mx.pi)
_LOG_2PI = mx.log(2 * mx.pi)
_MIN_LOG = -708
_SQRT3 = mx.sqrt(3)
_PI_SQUARED = mx.pi ** 2
_PI_FOUR = mx.pi ** 4
_PI_SIX = mx.pi ** 6

# [Lifted from _loggamma.pxd.] If B_m are the Bernoulli numbers,
# then Stirling coeffs are B_{2j}/(2j)/(2j-1) for j=8,...1.
_STIRLING_COEFFS = [-2.955065359477124183e-2, 6.4102564102564102564e-3,
                    -1.9175269175269175269e-3, 8.4175084175084175084e-4,
                    -5.952380952380952381e-4, 7.9365079365079365079e-4,
                    -2.7777777777777777778e-3, 8.3333333333333333333e-2]


def _log_nfactorial_div_n_pow_n(n):
    # Computes n! / n**n
    #    = (n-1)! / n**(n-1)
    # Uses Stirling's approximation, but removes n*log(n) up-front to
    # avoid subtractive cancellation.
    #    = log(n)/2 - n + log(sqrt(2pi)) + sum B_{2j}/(2j)/(2j-1)/n**(2j-1)
    rn = 1.0/n
    return mx.log(n)/2 - n + _LOG_2PI/2 + rn * mx.polyval(_STIRLING_COEFFS, rn/n)


def _clip_prob(p):
    """clips a probability to range 0<=p<=1."""
    return mx.clip(p, 0.0, 1.0)


def _select_and_clip_prob(cdfprob, sfprob, cdf=True):
    """Selects either the CDF or SF, and then clips to range 0<=p<=1."""
    p = mx.where(cdf, cdfprob, sfprob)
    return _clip_prob(p)


def _kolmogn_DMTW(n, d, cdf=True):
    r"""Computes the Kolmogorov CDF:  Pr(D_n <= d) using the MTW approach to
    the Durbin matrix algorithm.

    Durbin (1968); Marsaglia, Tsang, Wang (2003). [1], [3].
    """
    # Write d = (k-h)/n, where k is positive integer and 0 <= h < 1
    # Generate initial matrix H of size m*m where m=(2k-1)
    # Compute k-th row of (n!/n^n) * H^n, scaling intermediate results.
    # Requires memory O(m^2) and computation O(m^2 log(n)).
    # Most suitable for small m.

    if d >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf)
    nd = n * d
    if nd <= 0.5:
        return _select_and_clip_prob(0.0, 1.0, cdf)
    k = int(mx.ceil(nd))
    h = k - nd
    m = 2 * k - 1

    H = mx.zeros([m, m])

    # Initialize: v is first column (and last row) of H
    #  v[j] = (1-h^(j+1)/(j+1)!  (except for v[-1])
    #  w[j] = 1/(j)!
    # q = k-th row of H (actually i!/n^i*H^i)
    intm = mx.arange(1, m + 1)
    v = 1.0 - h ** intm
    w = mx.empty(m)
    fac = 1.0
    for j in intm:
        w[j - 1] = fac
        fac /= j  # This might underflow.  Isn't a problem.
        v[j - 1] *= fac
    tt = max(2 * h - 1.0, 0)**m - 2*h**m
    v[-1] = (1.0 + tt) * fac

    for i in range(1, m):
        H[i - 1:, i] = w[:m - i + 1]
    H[:, 0] = v
    H[-1, :] = mx.flip(v, axis=0)

    Hpwr = mx.eye(mx.shape(H)[0])  # Holds intermediate powers of H
    nn = n
    expnt = 0  # Scaling of Hpwr
    Hexpnt = 0  # Scaling of H
    while nn > 0:
        if nn % 2:
            Hpwr = mx.matmul(Hpwr, H)
            expnt += Hexpnt
        H = mx.matmul(H, H)
        Hexpnt *= 2
        # Scale as needed.
        if mx.abs(H[k - 1, k - 1]) > _EP128:
            H /= _EP128
            Hexpnt += _E128
        nn = nn // 2

    p = Hpwr[k - 1, k - 1]

    # Multiply by n!/n^n
    for i in range(1, n + 1):
        p = i * p / n
        if mx.abs(p) < _EM128:
            p *= _EP128
            expnt -= _E128

    # unscale
    if expnt != 0:
        p = mx.ldexp(p, expnt)

    return _select_and_clip_prob(p, 1.0-p, cdf)


def _pomeranz_compute_j1j2(i, n, ll, ceilf, roundf):
    """Compute the endpoints of the interval for row i."""
    if i == 0:
        j1, j2 = -ll - ceilf - 1, ll + ceilf - 1
    else:
        # i + 1 = 2*ip1div2 + ip1mod2
        ip1div2, ip1mod2 = divmod(i + 1, 2)
        if ip1mod2 == 0:  # i is odd
            if ip1div2 == n + 1:
                j1, j2 = n - ll - ceilf - 1, n + ll + ceilf - 1
            else:
                j1, j2 = ip1div2 - 1 - ll - roundf - 1, ip1div2 + ll - 1 + ceilf - 1
        else:
            j1, j2 = ip1div2 - 1 - ll - 1, ip1div2 + ll + roundf - 1

    return max(j1 + 2, 0), min(j2, n)


def _kolmogn_Pomeranz(n, x, cdf=True):
    r"""Computes Pr(D_n <= d) using the Pomeranz recursion algorithm.

    Pomeranz (1974) [2]
    """

    # V is n*(2n+2) matrix.
    # Each row is convolution of the previous row and probabilities from a
    #  Poisson distribution.
    # Desired CDF probability is n! V[n-1, 2n+1]  (final entry in final row).
    # Only two rows are needed at any given stage:
    #  - Call them V0 and V1.
    #  - Swap each iteration
    # Only a few (contiguous) entries in each row can be non-zero.
    #  - Keep track of start and end (j1 and j2 below)
    #  - V0s and V1s track the start in the two rows
    # Scale intermediate results as needed.
    # Only a few different Poisson distributions can occur
    t = n * x
    ll = int(mx.floor(t))
    f = 1.0 * (t - ll)  # fractional part of t
    g = min(f, 1.0 - f)
    ceilf = (1 if f > 0 else 0)
    roundf = (1 if f > 0.5 else 0)
    npwrs = 2 * (ll + 1)    # Maximum number of powers needed in convolutions
    gpower = mx.empty(npwrs)  # gpower = (g/n)^m/m!
    twogpower = mx.empty(npwrs)  # twogpower = (2g/n)^m/m!
    onem2gpower = mx.empty(npwrs)  # onem2gpower = ((1-2g)/n)^m/m!
    # gpower etc are *almost* Poisson probs, just missing normalizing factor.

    gpower[0] = 1.0
    twogpower[0] = 1.0
    onem2gpower[0] = 1.0
    expnt = 0
    g_over_n, two_g_over_n, one_minus_two_g_over_n = g/n, 2*g/n, (1 - 2*g)/n
    for m in range(1, npwrs):
        gpower[m] = gpower[m - 1] * g_over_n / m
        twogpower[m] = twogpower[m - 1] * two_g_over_n / m
        onem2gpower[m] = onem2gpower[m - 1] * one_minus_two_g_over_n / m

    V0 = mx.zeros([npwrs])
    V1 = mx.zeros([npwrs])
    V1[0] = 1  # first row
    V0s, V1s = 0, 0  # start indices of the two rows

    j1, j2 = _pomeranz_compute_j1j2(0, n, ll, ceilf, roundf)
    for i in range(1, 2 * n + 2):
        # Preserve j1, V1, V1s, V0s from last iteration
        k1 = j1
        V0, V1 = V1, V0
        V0s, V1s = V1s, V0s
        V1.fill(0.0)
        j1, j2 = _pomeranz_compute_j1j2(i, n, ll, ceilf, roundf)
        if i == 1 or i == 2 * n + 1:
            pwrs = gpower
        else:
            pwrs = (twogpower if i % 2 else onem2gpower)
        ln2 = j2 - k1 + 1
        if ln2 > 0:
            conv = mx.convolve(V0[k1 - V0s:k1 - V0s + ln2], pwrs[:ln2])
            conv_start = j1 - k1  # First index to use from conv
            conv_len = j2 - j1 + 1  # Number of entries to use from conv
            V1[:conv_len] = conv[conv_start:conv_start + conv_len]
            # Scale to avoid underflow.
            if 0 < mx.max(V1) < _EM128:
                V1 *= _EP128
                expnt -= _E128
            V1s = V0s + j1 - k1

    # multiply by n!
    ans = V1[n - V1s]
    for m in range(1, n + 1):
        if mx.abs(ans) > _EP128:
            ans *= _EM128
            expnt += _E128
        ans *= m

    # Undo any intermediate scaling
    if expnt != 0:
        ans = mx.ldexp(ans, expnt)
    ans = _select_and_clip_prob(ans, 1.0 - ans, cdf)
    return ans


def _kolmogn_PelzGood(n, x, cdf=True):
    """Computes the Pelz-Good approximation to Prob(Dn <= x) with 0<=x<=1.

    Start with Li-Chien, Korolyuk approximation:
        Prob(Dn <= x) ~ K0(z) + K1(z)/sqrt(n) + K2(z)/n + K3(z)/n**1.5
    where z = x*sqrt(n).
    Transform each K_(z) using Jacobi theta functions into a form suitable
    for small z.
    Pelz-Good (1976). [6]
    """
    if x <= 0.0:
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
    if x >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf=cdf)

    z = mx.sqrt(n) * x
    zsquared, zthree, zfour, zsix = z**2, z**3, z**4, z**6

    qlog = -_PI_SQUARED / 8 / zsquared
    if qlog < _MIN_LOG:  # z ~ 0.041743441416853426
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)

    q = mx.exp(qlog)

    # Coefficients of terms in the sums for K1, K2 and K3
    k1a = -zsquared
    k1b = _PI_SQUARED / 4

    k2a = 6 * zsix + 2 * zfour
    k2b = (2 * zfour - 5 * zsquared) * _PI_SQUARED / 4
    k2c = _PI_FOUR * (1 - 2 * zsquared) / 16

    k3d = _PI_SIX * (5 - 30 * zsquared) / 64
    k3c = _PI_FOUR * (-60 * zsquared + 212 * zfour) / 16
    k3b = _PI_SQUARED * (135 * zfour - 96 * zsix) / 4
    k3a = -30 * zsix - 90 * z**8

    K0to3 = mx.zeros(4)
    # Use a Horner scheme to evaluate sum c_i q^(i^2)
    # Reduces to a sum over odd integers.
    maxk = int(mx.ceil(16 * z / mx.pi))
    for k in range(maxk, 0, -1):
        m = 2 * k - 1
        msquared, mfour, msix = m**2, m**4, m**6
        qpower = mx.power(q, 8 * k)
        coeffs = mx.array([1.0,
                           k1a + k1b*msquared,
                           k2a + k2b*msquared + k2c*mfour,
                           k3a + k3b*msquared + k3c*mfour + k3d*msix])
        K0to3 *= qpower
        K0to3 += coeffs
    K0to3 *= q
    K0to3 *= _SQRT2PI
    # z**10 > 0 as z > 0.04
    K0to3 /= mx.array([z, 6 * zfour, 72 * z**7, 6480 * z**10])

    # Now do the other sum over the other terms, all integers k
    # K_2:  (pi^2 k^2) q^(k^2),
    # K_3:  (3pi^2 k^2 z^2 - pi^4 k^4)*q^(k^2)
    # Don't expect much subtractive cancellation so use direct calculation
    q = mx.exp(-_PI_SQUARED / 2 / zsquared)
    ks = mx.arange(maxk, 0, -1)
    ksquared = ks ** 2
    sqrt3z = _SQRT3 * z
    kspi = mx.pi * ks
    qpwers = q ** ksquared
    k2extra = mx.sum(ksquared * qpwers)
    k2extra *= _PI_SQUARED * _SQRT2PI/(-36 * zthree)
    K0to3[2] += k2extra
    k3extra = mx.sum((sqrt3z + kspi) * (sqrt3z - kspi) * ksquared * qpwers)
    k3extra *= _PI_SQUARED * _SQRT2PI/(216 * zsix)
    K0to3[3] += k3extra
    powers_of_n = mx.power(n * 1.0, mx.arange(len(K0to3)) / 2.0)
    K0to3 /= powers_of_n

    if not cdf:
        K0to3 *= -1
        K0to3[0] += 1

    Ksum = sum(K0to3)
    return Ksum


def _kolmogn(n, x, cdf=True):
    """Computes the CDF(or SF) for the two-sided Kolmogorov-Smirnov statistic.

    x must be of type float, n of type integer.

    Simard & L'Ecuyer (2011) [7].
    """
    if mx.isnan(n):
        return n  # Keep the same type of nan
    if int(n) != n or n <= 0:
        return mx.nan
    if x >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf=cdf)
    if x <= 0.0:
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
    t = n * x
    if t <= 1.0:  # Ruben-Gambino: 1/2n <= x <= 1/n
        if t <= 0.5:
            return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
        if n <= 140:
            prob = mx.prod(mx.arange(1, n+1) * (1.0/n) * (2*t - 1))
        else:
            prob = mx.exp(_log_nfactorial_div_n_pow_n(n) + n * mx.log(2*t-1))
        return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)
    if t >= n - 1:  # Ruben-Gambino
        prob = 2 * (1.0 - x)**n
        return _select_and_clip_prob(1 - prob, prob, cdf=cdf)
    if x >= 0.5:  # Exact: 2 * smirnov
        prob = 2 * scipy.special.smirnov(n, x)
        return _select_and_clip_prob(1.0 - prob, prob, cdf=cdf)

    nxsquared = t * x
    if n <= 140:
        if nxsquared <= 0.754693:
            prob = _kolmogn_DMTW(n, x, cdf=True)
            return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)
        if nxsquared <= 4:
            prob = _kolmogn_Pomeranz(n, x, cdf=True)
            return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)
        # Now use Miller approximation of 2*smirnov
        prob = 2 * scipy.special.smirnov(n, x)
        return _select_and_clip_prob(1.0 - prob, prob, cdf=cdf)

    # Split CDF and SF as they have different cutoffs on nxsquared.
    if not cdf:
        if nxsquared >= 370.0:
            return 0.0
        if nxsquared >= 2.2:
            prob = 2 * scipy.special.smirnov(n, x)
            return _clip_prob(prob)
        # Fall through and compute the SF as 1.0-CDF
    if nxsquared >= 18.0:
        cdfprob = 1.0
    elif n <= 100000 and n * x**1.5 <= 1.4:
        cdfprob = _kolmogn_DMTW(n, x, cdf=True)
    else:
        cdfprob = _kolmogn_PelzGood(n, x, cdf=True)
    return _select_and_clip_prob(cdfprob, 1.0 - cdfprob, cdf=cdf)


def _kolmogn_p(n, x):
    """Computes the PDF for the two-sided Kolmogorov-Smirnov statistic.

    x must be of type float, n of type integer.
    """
    if mx.isnan(n):
        return n  # Keep the same type of nan
    if int(n) != n or n <= 0:
        return mx.nan
    if x >= 1.0 or x <= 0:
        return 0
    t = n * x
    if t <= 1.0:
        # Ruben-Gambino: n!/n^n * (2t-1)^n -> 2 n!/n^n * n^2 * (2t-1)^(n-1)
        if t <= 0.5:
            return 0.0
        if n <= 140:
            prd = mx.prod(mx.arange(1, n) * (1.0 / n) * (2 * t - 1))
        else:
            prd = mx.exp(_log_nfactorial_div_n_pow_n(n) + (n-1) * mx.log(2 * t - 1))
        return prd * 2 * n**2
    if t >= n - 1:
        # Ruben-Gambino : 1-2(1-x)**n -> 2n*(1-x)**(n-1)
        return 2 * (1.0 - x) ** (n-1) * n
    if x >= 0.5:
        return 2 * scipy.stats.ksone.pdf(x, n)

    # Just take a small delta.
    # Ideally x +/- delta would stay within [i/n, (i+1)/n] for some integer a.
    # as the CDF is a piecewise degree n polynomial.
    # It has knots at 1/n, 2/n, ... (n-1)/n
    # and is not a C-infinity function at the knots
    delta = x / 2.0**16
    delta = min(delta, x - 1.0/n)
    delta = min(delta, 0.5 - x)

    def _kk(_x):
        return kolmogn(n, _x)

    return _derivative(_kk, x, dx=delta, order=5)


def _kolmogni(n, p, q):
    """Computes the PPF/ISF of kolmogn.

    n of type integer, n>= 1
    p is the CDF, q the SF, p+q=1
    """
    if mx.isnan(n):
        return n  # Keep the same type of nan
    if int(n) != n or n <= 0:
        return mx.nan
    if p <= 0:
        return 1.0/n
    if q <= 0:
        return 1.0
    delta = mx.exp((mx.log(p) - scipy.special.loggamma(n+1))/n)
    if delta <= 1.0/n:
        return (delta + 1.0 / n) / 2
    x = -mx.expm1(mx.log(q/2.0)/n)
    if x >= 1 - 1.0/n:
        return x
    x1 = scu._kolmogci(p)/mx.sqrt(n)
    x1 = min(x1, 1.0 - 1.0/n)

    def _f(x):
        return _kolmogn(n, x) - p

    return scipy.optimize.brentq(_f, 1.0/n, x1, xtol=1e-14)


def kolmogn(n, x, cdf=True):
    """Computes the CDF for the two-sided Kolmogorov-Smirnov distribution.

    The two-sided Kolmogorov-Smirnov distribution has as its CDF Pr(D_n <= x),
    for a sample of size n drawn from a distribution with CDF F(t), where
    :math:`D_n &= sup_t |F_n(t) - F(t)|`, and
    :math:`F_n(t)` is the Empirical Cumulative Distribution Function of the sample.

    Parameters
    ----------
    n : integer, array_like
        the number of samples
    x : float, array_like
        The K-S statistic, float between 0 and 1
    cdf : bool, optional
        whether to compute the CDF(default=true) or the SF.

    Returns
    -------
    cdf : array
        CDF (or SF it cdf is False) at the specified locations.

    The return value has shape the result of numpy broadcasting n and x.
    """
    it = mx.nditer([n, x, cdf, None], flags=['zerosize_ok'],
                   op_dtypes=[None, mx.float64, mx.bool_, mx.float64])
    for _n, _x, _cdf, z in it:
        if mx.isnan(_n):
            z[...] = _n
            continue
        if int(_n) != _n:
            raise ValueError(f'n is not integral: {_n}')
        z[...] = _kolmogn(int(_n), _x, cdf=_cdf)
    result = it.operands[-1]
    return result


def kolmognp(n, x):
    """Computes the PDF for the two-sided Kolmogorov-Smirnov distribution.

    Parameters
    ----------
    n : integer, array_like
        the number of samples
    x : float, array_like
        The K-S statistic, float between 0 and 1

    Returns
    -------
    pdf : array
        The PDF at the specified locations

    The return value has shape the result of numpy broadcasting n and x.
    """
    it = mx.nditer([n, x, None])
    for _n, _x, z in it:
        if mx.isnan(_n):
            z[...] = _n
            continue
        if int(_n) != _n:
            raise ValueError(f'n is not integral: {_n}')
        z[...] = _kolmogn_p(int(_n), _x)
    result = it.operands[-1]
    return result


def kolmogni(n, q, cdf=True):
    """Computes the PPF(or ISF) for the two-sided Kolmogorov-Smirnov distribution.

    Parameters
    ----------
    n : integer, array_like
        the number of samples
    q : float, array_like
        Probabilities, float between 0 and 1
    cdf : bool, optional
        whether to compute the PPF(default=true) or the ISF.

    Returns
    -------
    ppf : array
        PPF (or ISF if cdf is False) at the specified locations

    The return value has shape the result of numpy broadcasting n and x.
    """
    it = mx.nditer([n, q, cdf, None])
    for _n, _q, _cdf, z in it:
        if mx.isnan(_n):
            z[...] = _n
            continue
        if int(_n) != _n:
            raise ValueError(f'n is not integral: {_n}')
        _pcdf, _psf = (_q, 1-_q) if _cdf else (1-_q, _q)
        z[...] = _kolmogni(int(_n), _pcdf, _psf)
    result = it.operands[-1]
    return result
