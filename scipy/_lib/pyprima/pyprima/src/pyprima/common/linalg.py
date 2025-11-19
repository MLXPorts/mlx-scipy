'''
This module provides some basic linear algebra procedures.

Translated from Zaikun Zhang's modern-Fortran reference implementation in PRIMA.

Dedicated to late Professor M. J. D. Powell FRS (1936--2015).

Python translation by Nickolai Belakovski.
'''

import mlx.core as mx
from .consts import DEBUGGING, EPS, REALMAX, REALMIN
from .present import present


# We use naive implementations of matrix multiplication and other routines for two
# reasons:
# 1. When Fortran is compiled in debug mode, and Python is using these routines, we
#    can get bit for bit identical results as compared to Fortran. This is helpful
#    for comparing the two implementations. It will be particularly helpful when porting
#    the other implementations like LINCOA, etc.
# 2. On some problems this algorithm is very sensitive to errors in finite precision
#    arithmetic. Switching to naive implementation will slow down the algorithm, but
#    may be more stable.
USE_NAIVE_MATH = False


def inprod(x, y):
    if not USE_NAIVE_MATH:
        return mx.dot(x, y)
    result = 0
    for i in range(len(x)):
        result += x[i] * y[i]
    return result


def matprod12(x, y):
    result = mx.zeros(y.shape[1])
    for i in range(y.shape[1]):
        result[i] = inprod(x, y[:, i])
    return result


def matprod21(x, y):
    result = mx.zeros(x.shape[0])
    for i in range(x.shape[1]):
        result += x[:, i] * y[i]
    return result


def matprod22(x, y):
    result = mx.zeros((x.shape[0], y.shape[1]))
    for i in range(y.shape[1]):
        for j in range(x.shape[1]):
            result[:, j] += x[:, i] * y[i, j]
    return result


def matprod(x, y):
    if not USE_NAIVE_MATH:
        return x@y
    if len(x.shape) == 1 and len(y.shape) == 1:
        return inprod(x, y)
    elif len(x.shape) == 1 and len(y.shape) == 2:
        return matprod12(x, y)
    elif len(x.shape) == 2 and len(y.shape) == 1:
        return matprod21(x, y)
    elif len(x.shape) == 2 and len(y.shape) == 2:
        return matprod22(x, y)
    else:
        raise ValueError(f'Invalid shapes for x and y: {x.shape} and {y.shape}')


def outprod(x, y):
    if not USE_NAIVE_MATH:
        return mx.outer(x, y)
    result = mx.zeros((len(x), len(y)))
    for i in range(len(x)):
            result[:, i] = x * y[i]
    return result


def lsqr(A, b, Q, Rdiag):
    if not USE_NAIVE_MATH:
        return mx.linalg.lstsq(A, b, rcond=None)[0]

    m = A.shape[0]
    n = A.shape[1]

    rank = min(m, n)

    x = mx.zeros(n)
    y = b.copy()

    for i in range(rank - 1, -1, -1):
        yq = inprod(y, Q[:, i])
        yqa = inprod(mx.abs(y), mx.abs(Q[:, i]))
        if isminor(yq, yqa):
            x[i] = 0
        else:
            x[i] = yq / Rdiag[i]
            y = y - x[i] * A[:, i]
    return x


def hypot(x1, x2):
    if not USE_NAIVE_MATH:
        return mx.hypot(x1, x2)
    if not mx.isfinite(x1):
        r = abs(x1)
    elif not mx.isfinite(x2):
        r = abs(x2)
    else:
        y = abs(mx.array([x1, x2]))
        y = mx.array([min(y), max(y)])
        if y[0] > mx.sqrt(REALMIN) and y[1] < mx.sqrt(REALMAX/2.1):
            r = mx.sqrt(sum(y*y))
        elif y[1] > 0:
            r = y[1] * mx.sqrt((y[0]/y[1])*(y[0]/y[1]) + 1)
        else:
            r = 0
    return r


def norm(x):
    if not USE_NAIVE_MATH:
        return mx.linalg.norm(x)
    # NOTE: Avoid mx.pow! And exponentiation in general!
    # It appears that in Fortran, x*x and x**2 are the same, but in Python they are not!
    # Try it with x = 5 - 1e-15
    result = mx.sqrt(sum([xi*xi for xi in x]))
    return result


def istril(A, tol=0):
    return primasum(abs(A) - mx.tril(abs(A))) <= tol

def istriu(A, tol=0):
    return primasum(abs(A) - mx.triu(abs(A))) <= tol


def inv(A):
    if not USE_NAIVE_MATH:
        return mx.linalg.inv(A)
    A = A.copy()
    n = A.shape[0]
    if istril(A):
        # This case is invoked in COBYLA.
        R = A.T
        B = mx.zeros((n, n))
        for i in range(n):
            B[i, i] = 1 / R[i, i]
            B[:i, i] = -matprod(B[:i, :i], R[:i, i]) / R[i, i]
        return B.T
    elif istriu(A):
        B = mx.zeros((n, n))
        for i in range(n):
            B[i, i] = 1 / A[i, i]
            B[:i, i] = -matprod(B[:i, :i], A[:i, i]) / A[i, i]
    else:
        # This is NOT the best algorithm for the inverse, but since the QR subroutine is available ...
        Q, R, P = qr(A)
        R = R.T
        B = mx.zeros((n, n))
        for i in range(n - 1, -1, -1):
            B[:, i] = (Q[:, i] - matprod(B[:, i + 1:n], R[i + 1:n, i])) / R[i, i]
        InvP = mx.zeros(n, dtype=int)
        InvP[P] = mx.linspace(0, n-1, n)
        B = B[:, InvP].T
    return B


def qr(A):
    m = A.shape[0]
    n = A.shape[1]

    Q = mx.eye(m)
    T = A.T
    P = mx.linspace(0, n-1, n, dtype=int)

    for j in range(n):
        k = mx.argmax(primasum(primapow2(T[j:n+1, j:m+1]), axis=1), axis=0)
        if k > 0 and k <= n - j - 1:
            k += j
            P[j], P[k] = P[k], P[j]
            T[[j, k], :] = T[[k, j], :]
        for i in range(m-1, j, -1):
            G = planerot(T[j, [j, i]]).T
            T[j, [j, i]] = mx.append(hypot(T[j, j], T[j, i]), 0)
            T[j + 1:n + 1, [j, i]] = matprod(T[j + 1:n + 1, [j, i]], G)
            Q[:, [j, i]] = matprod(Q[:, [j, i]], G)

    R = T.T

    return Q, R, P


def primasum(x, axis=None):
    '''
    According to its documentation, mx.sum will sometimes do partial pairwise summation.
    For our purposes, when comparing, we want don't want to do anything fancy, and we
    just want to add things up one at a time.
    '''
    if not USE_NAIVE_MATH:
        return mx.sum(x, axis=axis)
    if axis is None:
        if x.ndim == 2:
            # Sum columns first, then sum the result
            return sum(primasum(x, axis=0))
        else:
            return sum(x)
    elif axis == 0:
        result = mx.zeros(x.shape[1])
        for i in range(x.shape[1]):
            result[i] = sum(x[:, i])
        return result
    elif axis == 1:
        result = mx.zeros(x.shape[0])
        for i in range(x.shape[0]):
            result[i] = sum(x[i, :])
        return result


def primapow2(x):
    '''
    Believe it or now, x**2 is not always the same as x*x in Python. In Fortran they
    appear to be identical. Here's a quick one-line to find an example on your system
    (well, two liner after importing numpy):
    list(filter(lambda x: x[1], [(x:=mx.random.random(), x**2 - x*x != 0) for _ in range(10000)]))
    '''
    return x*x


def planerot(x):
    '''
    As in MATLAB, planerot(x) returns a 2x2 Givens matrix G for x in R2 so that Y=G@x has Y[1] = 0.
    Roughly speaking, G = mx.array([[x[0]/R, x[1]/R], [-x[1]/R, x[0]/R]]), where R = mx.linalg.norm(x).
    0. We need to take care of the possibilities of R=0, Inf, NaN, and over/underflow.
    1. The G defined above is continuous with respect to X except at 0. Following this definition,
    G = mx.array([[mx.sign(x[0]), 0], [0, mx.sign(x[0])]]) if x[1] == 0,
    G = mx.array([[0, mx.sign(x[1])], [mx.sign(x[1]), 0]]) if x[0] == 0
    Yet some implementations ignore the signs, leading to discontinuity and numerical instability.
    2. Difference from MATLAB: if x contains NaN of consists of only Inf, MATLAB returns a NaN matrix,
    but we return an identity matrix or a matrix of +/-mx.sqrt(2). We intend to keep G always orthogonal.
    '''

    # Preconditions
    if DEBUGGING:
        assert len(x) == 2, "x must be a 2-vector"

    # ==================
    # Calculation starts
    # ==================

    # Define C = X(1) / R and S = X(2) / R with R = HYPOT(X(1), X(2)). Handle Inf/NaN, over/underflow.
    if (any(mx.isnan(x))):
        # In this case, MATLAB sets G to NaN(2, 2). We refrain from doing so to keep G orthogonal.
        c = 1
        s = 0
    elif (all(mx.isinf(x))):
        # In this case, MATLAB sets G to NaN(2, 2). We refrain from doing so to keep G orthogonal.
        c = 1 / mx.sqrt(2) * mx.sign(x[0])
        s = 1 / mx.sqrt(2) * mx.sign(x[1])
    elif (abs(x[0]) <= 0 and abs(x[1]) <= 0): # X(1) == 0 == X(2).
        c = 1
        s = 0
    elif (abs(x[1]) <= EPS * abs(x[0])):
        # N.B.:
        # 0. With <= instead of <, this case covers X(1) == 0 == X(2), which is treated above separately
        # to avoid the confusing SIGN(., 0) (see 1).
        # 1. SIGN(A, 0) = ABS(A) in Fortran but sign(0) = 0 in MATLAB, Python, Julia, and R#
        # 2. Taking SIGN(X(1)) into account ensures the continuity of G with respect to X except at 0.
        c = mx.sign(x[0])
        s = 0
    elif (abs(x[0]) <= EPS * abs(x[1])):
        # N.B.: SIGN(A, X) = ABS(A) * sign of X /= A * sign of X # Therefore, it is WRONG to define G
        # as SIGN(RESHAPE([ZERO, -ONE, ONE, ZERO], [2, 2]), X(2)). This mistake was committed on
        # 20211206 and took a whole day to debug! NEVER use SIGN on arrays unless you are really sure.
        c = 0
        s = mx.sign(x[1])
    else:
        # Here is the normal case. It implements the Givens rotation in a stable & continuous way as in:
        # Bindel, D., Demmel, J., Kahan, W., and Marques, O. (2002). On computing Givens rotations
        # reliably and efficiently. ACM Transactions on Mathematical Software (TOMS), 28(2), 206-238.
        # N.B.: 1. Modern compilers compute SQRT(REALMIN) and SQRT(REALMAX/2.1) at compilation time.
        # 2. The direct calculation without involving T and U seems to work better; use it if possible.
        if (all(mx.logical_and(mx.sqrt(REALMIN) < mx.abs(x), mx.abs(x) < mx.sqrt(REALMAX / 2.1)))):
            # Do NOT use HYPOTENUSE here; the best implementation for one may be suboptimal for the other
            r = norm(x)
            c = x[0] / r
            s = x[1] / r
        elif (abs(x[0]) > abs(x[1])):
            t = x[1] / x[0]
            u = max(1, abs(t), mx.sqrt(1 + t*t))  # MAXVAL: precaution against rounding error.
            u *= mx.sign(x[0]) ##MATLAB: u = sign(x(1))*sqrt(1 + t**2)
            c = 1 / u
            s = t / u
        else:
            t = x[0] / x[1]
            u = max([1, abs(t), mx.sqrt(1 + t*t)])  # MAXVAL: precaution against rounding error.
            u *= mx.sign(x[1]) ##MATLAB: u = sign(x(2))*sqrt(1 + t**2)
            c = t / u
            s = 1 / u

    G = mx.array([[c, s], [-s, c]]) #  MATLAB: G = [c, s; -s, c]

    #====================#
    #  Calculation ends  #
    #====================#

    # Postconditions
    if DEBUGGING:
        assert G.shape == (2,2)
        assert mx.all(mx.isfinite(G))
        assert abs(G[0, 0] - G[1, 1]) + abs(G[0, 1] + G[1, 0]) <= 0
        tol = mx.maximum(1.0E-10, mx.minimum(1.0E-1, 1.0E6 * EPS))
        assert isorth(G, tol)
        if all(mx.logical_and(mx.isfinite(x), mx.abs(x) < mx.sqrt(REALMAX / 2.1))):
            r = mx.linalg.norm(x)
            assert max(abs(G@x - [r, 0])) <= max(tol, tol * r), 'G @ X = [||X||, 0]'

    return G


def isminor(x, ref):
    '''
    This function tests whether x is minor compared to ref. It is used by Powell, e.g., in COBYLA.
    In precise arithmetic, isminor(x, ref) is true if and only if x == 0; in floating point
    arithmetic, isminor(x, ref) is true if x is 0 or its nonzero value can be attributed to
    computer rounding errors according to ref.
    Larger sensitivity means the function is more strict/precise, the value 0.1 being due to Powell.

    For example:
    isminor(1e-20, 1e300) -> True, because in floating point arithmetic 1e-20 cannot be added to
    1e300 without being rounded to 1e300.
    isminor(1e300, 1e-20) -> False, because in floating point arithmetic adding 1e300 to 1e-20
    dominates the latter number.
    isminor(3, 4) -> False, because 3 can be added to 4 without being rounded off
    '''

    sensitivity = 0.1
    refa = abs(ref) + sensitivity * abs(x)
    refb = abs(ref) + 2 * sensitivity * abs(x)
    return mx.logical_or(abs(ref) >= refa, refa >= refb)


def isinv(A, B, tol=None):
    '''
    This procedure tests whether A = B^{-1} up to the tolerance TOL.
    '''

    # Sizes
    n = mx.size(A, 0)

    # Preconditions
    if DEBUGGING:
        assert mx.size(A, 0) == mx.size(A, 1)
        assert mx.size(B, 0) == mx.size(B, 1)
        assert mx.size(A, 0) == mx.size(B, 0)
        if present(tol):
            assert tol >= 0

    #====================#
    # Calculation starts #
    #====================#

    tol = tol if present(tol) else mx.minimum(1e-3, 1e2 * EPS * mx.maximum(mx.size(A, 0), mx.size(A, 1)))
    tol = mx.max([tol, tol * mx.max(abs(A)), tol * mx.max(abs(B))])
    is_inv = ((abs(matprod(A, B)) - mx.eye(n)) <= tol).all() or ((abs(matprod(B, A) - mx.eye(n))) <= tol).all()

    #===================#
    #  Calculation ends #
    #===================#
    return is_inv


def isorth(A, tol=None):
    '''
    This function tests whether the matrix A has orthonormal columns up to the tolerance TOL.
    '''

    # Preconditions
    if DEBUGGING:
        if present(tol):
            assert tol >= 0

    #====================#
    # Calculation starts #
    #====================#

    num_vars = mx.size(A, 1)

    if num_vars > mx.size(A, 0):
        is_orth = False
    elif (mx.isnan(primasum(abs(A)))):
        is_orth = False
    else:
        if present(tol):
            is_orth = (abs(matprod(A.T, A) - mx.eye(num_vars)) <= mx.maximum(tol, tol * mx.max(abs(A)))).all()
        else:
            is_orth = (abs(matprod(A.T, A) - mx.eye(num_vars)) <= 0).all()

    #====================#
    #  Calculation ends  #
    #====================#
    return is_orth


def get_arrays_tol(*arrays):
    """
    Get a relative tolerance for a set of arrays. Borrowed from COBYQA

    Parameters
    ----------
    *arrays: tuple
        Set of `mx.array` to get the tolerance for.

    Returns
    -------
    float
        Relative tolerance for the set of arrays.

    Raises
    ------
    ValueError
        If no array is provided.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided.")
    size = max(array.size for array in arrays)
    weight = max(
        mx.max(mx.abs(array[mx.isfinite(array)]), initial=1.0)
        for array in arrays
    )
    return 10.0 * EPS * max(size, 1.0) * weight
