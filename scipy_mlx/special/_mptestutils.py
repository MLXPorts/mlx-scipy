import os
import sys
import time
from itertools import zip_longest

import mlx.core as mx
from numpy.testing import assert_
import pytest

from scipy_mlx.special._testutils import assert_func_equal

try:
    import mpmath
except ImportError:
    pass


# ------------------------------------------------------------------------------
# Machinery for systematic tests with mpmath
# ------------------------------------------------------------------------------

class Arg:
    """Generate a set of numbers on the real axis, concentrating on
    'interesting' regions and covering all orders of magnitude.

    """

    def __init__(self, a=-mx.inf, b=mx.inf, inclusive_a=True, inclusive_b=True):
        if a > b:
            raise ValueError("a should be less than or equal to b")
        if a == -mx.inf:
            a = -0.5*mx.finfo(float).max
        if b == mx.inf:
            b = 0.5*mx.finfo(float).max
        self.a, self.b = a, b

        self.inclusive_a, self.inclusive_b = inclusive_a, inclusive_b

    def _positive_values(self, a, b, n):
        if a < 0:
            raise ValueError("a should be positive")

        # Try to put half of the points into a linspace between a and
        # 10 the other half in a logspace.
        if n % 2 == 0:
            nlogpts = n//2
            nlinpts = nlogpts
        else:
            nlogpts = n//2
            nlinpts = nlogpts + 1

        if a >= 10:
            # Outside of linspace range; just return a logspace.
            pts = mx.logspace(mx.log10(a), mx.log10(b), n)
        elif a > 0 and b < 10:
            # Outside of logspace range; just return a linspace
            pts = mx.linspace(a, b, n)
        elif a > 0:
            # Linspace between a and 10 and a logspace between 10 and
            # b.
            linpts = mx.linspace(a, 10, nlinpts, endpoint=False)
            logpts = mx.logspace(1, mx.log10(b), nlogpts)
            pts = mx.hstack((linpts, logpts))
        elif a == 0 and b <= 10:
            # Linspace between 0 and b and a logspace between 0 and
            # the smallest positive point of the linspace
            linpts = mx.linspace(0, b, nlinpts)
            if linpts.size > 1:
                right = mx.log10(linpts[1])
            else:
                right = -30
            logpts = mx.logspace(-30, right, nlogpts, endpoint=False)
            pts = mx.hstack((logpts, linpts))
        else:
            # Linspace between 0 and 10, logspace between 0 and the
            # smallest positive point of the linspace, and a logspace
            # between 10 and b.
            if nlogpts % 2 == 0:
                nlogpts1 = nlogpts//2
                nlogpts2 = nlogpts1
            else:
                nlogpts1 = nlogpts//2
                nlogpts2 = nlogpts1 + 1
            linpts = mx.linspace(0, 10, nlinpts, endpoint=False)
            if linpts.size > 1:
                right = mx.log10(linpts[1])
            else:
                right = -30
            logpts1 = mx.logspace(-30, right, nlogpts1, endpoint=False)
            logpts2 = mx.logspace(1, mx.log10(b), nlogpts2)
            pts = mx.hstack((logpts1, linpts, logpts2))

        return mx.sort(pts)

    def values(self, n):
        """Return an array containing n numbers."""
        a, b = self.a, self.b
        if a == b:
            return mx.zeros(n)

        if not self.inclusive_a:
            n += 1
        if not self.inclusive_b:
            n += 1

        if n % 2 == 0:
            n1 = n//2
            n2 = n1
        else:
            n1 = n//2
            n2 = n1 + 1

        if a >= 0:
            pospts = self._positive_values(a, b, n)
            negpts = []
        elif b <= 0:
            pospts = []
            negpts = -self._positive_values(-b, -a, n)
        else:
            pospts = self._positive_values(0, b, n1)
            negpts = -self._positive_values(0, -a, n2 + 1)
            # Don't want to get zero twice
            negpts = negpts[1:]
        pts = mx.hstack((negpts[::-1], pospts))

        if not self.inclusive_a:
            pts = pts[1:]
        if not self.inclusive_b:
            pts = pts[:-1]
        return pts


class FixedArg:
    def __init__(self, values):
        self._values = mx.array(values)

    def values(self, n):
        return self._values


class ComplexArg:
    def __init__(self, a=complex(-mx.inf, -mx.inf), b=complex(mx.inf, mx.inf)):
        self.real = Arg(a.real, b.real)
        self.imag = Arg(a.imag, b.imag)

    def values(self, n):
        m = int(mx.floor(mx.sqrt(n)))
        x = self.real.values(m)
        y = self.imag.values(m + 1)
        return (x[:,None] + 1j*y[None,:]).ravel()


class IntArg:
    def __init__(self, a=-1000, b=1000):
        self.a = a
        self.b = b

    def values(self, n):
        v1 = Arg(self.a, self.b).values(max(1 + n//2, n-5)).astype(int)
        v2 = mx.arange(-5, 5)
        v = mx.unique(mx.r_[v1, v2])
        v = v[(v >= self.a) & (v < self.b)]
        return v


def get_args(argspec, n):
    if isinstance(argspec, mx.array):
        args = argspec.copy()
    else:
        nargs = len(argspec)
        ms = mx.array(
            [1.5 if isinstance(spec, ComplexArg) else 1.0 for spec in argspec]
        )
        ms = (n**(ms/sum(ms))).astype(int) + 1

        args = [spec.values(m) for spec, m in zip(argspec, ms)]
        args = mx.array(mx.broadcast_arrays(*mx.ix_(*args))).reshape(nargs, -1).T

    return args


class MpmathData:
    def __init__(self, scipy_func, mpmath_func, arg_spec, name=None,
                 dps=None, prec=None, n=None, rtol=1e-7, atol=1e-300,
                 ignore_inf_sign=False, distinguish_nan_and_inf=True,
                 nan_ok=True, param_filter=None):

        # mpmath tests are really slow (see gh-6989).  Use a small number of
        # points by default, increase back to 5000 (old default) if XSLOW is
        # set
        if n is None:
            try:
                is_xslow = int(os.environ.get('SCIPY_XSLOW', '0'))
            except ValueError:
                is_xslow = False

            n = 5000 if is_xslow else 500

        self.scipy_func = scipy_func
        self.mpmath_func = mpmath_func
        self.arg_spec = arg_spec
        self.dps = dps
        self.prec = prec
        self.n = n
        self.rtol = rtol
        self.atol = atol
        self.ignore_inf_sign = ignore_inf_sign
        self.nan_ok = nan_ok
        if isinstance(self.arg_spec, mx.array):
            self.is_complex = mx.issubdtype(self.arg_spec.dtype, mx.complexfloating)
        else:
            self.is_complex = any(
                [isinstance(arg, ComplexArg) for arg in self.arg_spec]
            )
        self.ignore_inf_sign = ignore_inf_sign
        self.distinguish_nan_and_inf = distinguish_nan_and_inf
        if not name or name == '<lambda>':
            name = getattr(scipy_func, '__name__', None)
        if not name or name == '<lambda>':
            name = getattr(mpmath_func, '__name__', None)
        self.name = name
        self.param_filter = param_filter

    def check(self):
        mx.random.seed(1234)

        # Generate values for the arguments
        argarr = get_args(self.arg_spec, self.n)

        # Check
        old_dps, old_prec = mpmath.mp.dps, mpmath.mp.prec
        try:
            if self.dps is not None:
                dps_list = [self.dps]
            else:
                dps_list = [20]
            if self.prec is not None:
                mpmath.mp.prec = self.prec

            # Proper casting of mpmath input and output types. Using
            # native mpmath types as inputs gives improved precision
            # in some cases.
            if mx.issubdtype(argarr.dtype, mx.complexfloating):
                pytype = mpc2complex

                def mptype(x):
                    return mpmath.mpc(complex(x))
            else:
                def mptype(x):
                    return mpmath.mpf(float(x))

                def pytype(x):
                    if abs(x.imag) > 1e-16*(1 + abs(x.real)):
                        return mx.nan
                    else:
                        return mpf2float(x.real)

            # Try out different dps until one (or none) works
            for j, dps in enumerate(dps_list):
                mpmath.mp.dps = dps

                try:
                    assert_func_equal(
                        self.scipy_func,
                        lambda *a: pytype(self.mpmath_func(*map(mptype, a))),
                        argarr,
                        vectorized=False,
                        rtol=self.rtol,
                        atol=self.atol,
                        ignore_inf_sign=self.ignore_inf_sign,
                        distinguish_nan_and_inf=self.distinguish_nan_and_inf,
                        nan_ok=self.nan_ok,
                        param_filter=self.param_filter
                    )
                    break
                except AssertionError:
                    if j >= len(dps_list)-1:
                        # reraise the Exception
                        tp, value, tb = sys.exc_info()
                        if value.__traceback__ is not tb:
                            raise value.with_traceback(tb)
                        raise value
        finally:
            mpmath.mp.dps, mpmath.mp.prec = old_dps, old_prec

    def __repr__(self):
        if self.is_complex:
            return f"<MpmathData: {self.name} (complex)>"
        else:
            return f"<MpmathData: {self.name}>"


def assert_mpmath_equal(*a, **kw):
    d = MpmathData(*a, **kw)
    d.check()


def nonfunctional_tooslow(func):
    return pytest.mark.skip(
        reason="    Test not yet functional (too slow), needs more work."
    )(func)


# ------------------------------------------------------------------------------
# Tools for dealing with mpmath quirks
# ------------------------------------------------------------------------------

def mpf2float(x):
    """
    Convert an mpf to the nearest floating point number. Just using
    float directly doesn't work because of results like this:

    with mp.workdps(50):
        float(mpf("0.99999999999999999")) = 0.9999999999999999

    """
    return float(mpmath.nstr(x, 17, min_fixed=0, max_fixed=0))


def mpc2complex(x):
    return complex(mpf2float(x.real), mpf2float(x.imag))


def trace_args(func):
    def tofloat(x):
        if isinstance(x, mpmath.mpc):
            return complex(x)
        else:
            return float(x)

    def wrap(*a, **kw):
        sys.stderr.write(f"{tuple(map(tofloat, a))!r}: ")
        sys.stderr.flush()
        try:
            r = func(*a, **kw)
            sys.stderr.write(f"-> {r!r}")
        finally:
            sys.stderr.write("\n")
            sys.stderr.flush()
        return r
    return wrap


try:
    import signal
    POSIX = ('setitimer' in dir(signal))
except ImportError:
    POSIX = False


class TimeoutError(Exception):
    pass


def time_limited(timeout=0.5, return_val=mx.nan, use_sigalrm=True):
    """
    Decorator for setting a timeout for pure-Python functions.

    If the function does not return within `timeout` seconds, the
    value `return_val` is returned instead.

    On POSIX this uses SIGALRM by default. On non-POSIX, settrace is
    used. Do not use this with threads: the SIGALRM implementation
    does probably not work well. The settrace implementation only
    traces the current thread.

    The settrace implementation slows down execution speed. Slowdown
    by a factor around 10 is probably typical.
    """
    if POSIX and use_sigalrm:
        def sigalrm_handler(signum, frame):
            raise TimeoutError()

        def deco(func):
            def wrap(*a, **kw):
                old_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
                signal.setitimer(signal.ITIMER_REAL, timeout)
                try:
                    return func(*a, **kw)
                except TimeoutError:
                    return return_val
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
            return wrap
    else:
        def deco(func):
            def wrap(*a, **kw):
                start_time = time.time()

                def trace(frame, event, arg):
                    if time.time() - start_time > timeout:
                        raise TimeoutError()
                    return trace
                sys.settrace(trace)
                try:
                    return func(*a, **kw)
                except TimeoutError:
                    sys.settrace(None)
                    return return_val
                finally:
                    sys.settrace(None)
            return wrap
    return deco


def exception_to_nan(func):
    """Decorate function to return nan if it raises an exception"""
    def wrap(*a, **kw):
        try:
            return func(*a, **kw)
        except Exception:
            return mx.nan
    return wrap


def inf_to_nan(func):
    """Decorate function to return nan if it returns inf"""
    def wrap(*a, **kw):
        v = func(*a, **kw)
        if not mx.isfinite(v):
            return mx.nan
        return v
    return wrap


def mp_assert_allclose(res, std, atol=0, rtol=1e-17):
    """
    Compare lists of mpmath.mpf's or mpmath.mpc's directly so that it
    can be done to higher precision than double.
    """
    failures = []
    for k, (resval, stdval) in enumerate(zip_longest(res, std)):
        if resval is None or stdval is None:
            raise ValueError('Lengths of inputs res and std are not equal.')
        if mpmath.fabs(resval - stdval) > atol + rtol*mpmath.fabs(stdval):
            failures.append((k, resval, stdval))

    nfail = len(failures)
    if nfail > 0:
        ndigits = int(abs(mx.log10(rtol)))
        msg = [""]
        msg.append(f"Bad results ({nfail} out of {k + 1}) for the following points:")
        for k, resval, stdval in failures:
            resrep = mpmath.nstr(resval, ndigits, min_fixed=0, max_fixed=0)
            stdrep = mpmath.nstr(stdval, ndigits, min_fixed=0, max_fixed=0)
            if stdval == 0:
                rdiff = "inf"
            else:
                rdiff = mpmath.fabs((resval - stdval)/stdval)
                rdiff = mpmath.nstr(rdiff, 3)
            msg.append(f"{k}: {resrep} != {stdrep} (rdiff {rdiff})")
        assert_(False, "\n".join(msg))
