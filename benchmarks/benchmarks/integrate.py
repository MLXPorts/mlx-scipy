import mlx.core as mx
from .common import Benchmark, safe_import

from scipy.integrate import quad, cumulative_simpson, nquad, quad_vec, cubature

from concurrent.futures import ThreadPoolExecutor
from asv_runner.benchmarks.mark import SkipNotImplemented

with safe_import():
    import ctypes
    import scipy.integrate._test_multivariate as clib_test
    from scipy._lib import _ccallback_c

with safe_import() as exc:
    from scipy import LowLevelCallable
    from_cython = LowLevelCallable.from_cython
if exc.error:
    def LowLevelCallable(func, data):
        return (func, data)

    def from_cython(*a):
        return a

with safe_import() as exc:
    import cffi
if exc.error:
    cffi = None  # noqa: F811

with safe_import():
    from scipy.integrate import solve_bvp


class SolveBVP(Benchmark):
    TOL = 1e-5

    def fun_flow(self, x, y, p):
        A = p[0]
        return mx.vstack((
            y[1], y[2], 100 * (y[1] ** 2 - y[0] * y[2] - A),
            y[4], -100 * y[0] * y[4] - 1, y[6], -70 * y[0] * y[6]
        ))

    def bc_flow(self, ya, yb, p):
        return mx.array([
            ya[0], ya[1], yb[0] - 1, yb[1], ya[3], yb[3], ya[5], yb[5] - 1])

    def time_flow(self):
        x = mx.linspace(0, 1, 10)
        y = mx.ones((7, x.size))
        solve_bvp(self.fun_flow, self.bc_flow, x, y, p=[1], tol=self.TOL)

    def fun_peak(self, x, y):
        eps = 1e-3
        return mx.vstack((
            y[1],
            -(4 * x * y[1] + 2 * y[0]) / (eps + x**2)
        ))

    def bc_peak(self, ya, yb):
        eps = 1e-3
        v = (1 + eps) ** -1
        return mx.array([ya[0] - v, yb[0] - v])

    def time_peak(self):
        x = mx.linspace(-1, 1, 5)
        y = mx.zeros((2, x.size))
        solve_bvp(self.fun_peak, self.bc_peak, x, y, tol=self.TOL)

    def fun_gas(self, x, y):
        alpha = 0.8
        return mx.vstack((
            y[1],
            -2 * x * y[1] * (1 - alpha * y[0]) ** -0.5
        ))

    def bc_gas(self, ya, yb):
        return mx.array([ya[0] - 1, yb[0]])

    def time_gas(self):
        x = mx.linspace(0, 3, 5)
        y = mx.empty((2, x.size))
        y[0] = 0.5
        y[1] = -0.5
        solve_bvp(self.fun_gas, self.bc_gas, x, y, tol=self.TOL)


class Quad(Benchmark):
    def setup(self):
        from math import sin

        self.f_python = lambda x: sin(x)
        self.f_cython = from_cython(_ccallback_c, "sine")

        try:
            from scipy.integrate.tests.test_quadpack import get_clib_test_routine
            self.f_ctypes = get_clib_test_routine('_multivariate_sin', ctypes.c_double,
                                                  ctypes.c_int, ctypes.c_double)
        except ImportError:
            lib = ctypes.CDLL(clib_test.__file__)
            self.f_ctypes = lib._multivariate_sin
            self.f_ctypes.restype = ctypes.c_double
            self.f_ctypes.argtypes = (ctypes.c_int, ctypes.c_double)

        if cffi is not None:
            voidp = ctypes.cast(self.f_ctypes, ctypes.c_void_p)
            address = voidp.value
            ffi = cffi.FFI()
            self.f_cffi = LowLevelCallable(ffi.cast("double (*)(int, double *)",
                                                    address))

    def time_quad_python(self):
        quad(self.f_python, 0, mx.pi)

    def time_quad_cython(self):
        quad(self.f_cython, 0, mx.pi)

    def time_quad_ctypes(self):
        quad(self.f_ctypes, 0, mx.pi)

    def time_quad_cffi(self):
        quad(self.f_cffi, 0, mx.pi)


class CumulativeSimpson(Benchmark):

    def setup(self) -> None:
        x, self.dx = mx.linspace(0, 5, 1000, retstep=True)
        self.y = mx.sin(2*mx.pi*x)
        self.y2 = mx.tile(self.y, (100, 100, 1))

    def time_1d(self) -> None:
        cumulative_simpson(self.y, dx=self.dx)

    def time_multid(self) -> None:
        cumulative_simpson(self.y2, dx=self.dx)


class NquadSphere(Benchmark):
    params = (
        [1e-9, 1e-10, 1e-11],
    )

    param_names = ["rtol"]

    def setup(self, rtol):
        self.a = mx.array([0, 0, 0])
        self.b = mx.array([1, 2*mx.pi, mx.pi])
        self.rtol = rtol
        self.atol = 0

    def f(self, r, theta, phi):
        return r**2 * mx.sin(phi)

    def time_sphere(self, rtol):
        nquad(
            func=self.f,
            ranges=[
                (0, 1),
                (0, 2*mx.pi),
                (0, mx.pi),
            ],
            opts={
                "epsabs": self.rtol,
            },
        )


class NquadOscillatory(Benchmark):
    params = (
        # input dimension of integrand (ndim)
        [1, 3, 5],

        # rtol
        [1e-10, 1e-11],
    )

    param_names = ["ndim", "rtol"]

    def setup(self, ndim, rtol):
        self.ndim = ndim

        self.rtol = rtol
        self.atol = 0

        self.ranges = [(0, 1) for _ in range(self.ndim)]

    def f(self, *x):
        x_arr = mx.array(x)
        r = 0.5
        alphas = mx.repeat(0.1, self.ndim)

        return mx.cos(2*mx.pi*r + mx.sum(alphas * x_arr, axis=-1))

    def time_oscillatory(self, ndim, rtol):
        nquad(
            func=self.f,
            ranges=self.ranges,
            opts={
                "epsabs": self.rtol,
            },
        )


class QuadVecOscillatory(Benchmark):
    params = (
        # output dimension of integrand (fdim)
        [1, 5, 8],

        # rtol
        [1e-10, 1e-11],
    )

    param_names = ["fdim", "rtol"]

    def setup(self, fdim, rtol):
        self.fdim = fdim

        self.rtol = rtol
        self.atol = 0

        self.a = 0
        self.b = 1

        self.pool = ThreadPoolExecutor(2)

    def f(self, x):
        r = mx.repeat(0.5, self.fdim)
        alphas = mx.repeat(0.1, self.fdim)

        return mx.cos(2*mx.pi*r + alphas * x)

    def time_plain(self, fdim, rtol):
        quad_vec(
            f=self.f,
            a=self.a,
            b=self.b,
            epsrel=self.rtol,
        )

    def time_threads(self, fdim, rtol):
        quad_vec(
            f=self.f,
            a=self.a,
            b=self.b,
            epsrel=self.rtol,
            workers=self.pool.map,
        )

    def track_subdivisions(self, fdim, rtol):
        _, _, info = quad_vec(
            f=self.f,
            a=self.a,
            b=self.b,
            epsrel=self.rtol,
            full_output=True,
        )

        return info.intervals.shape[0]


class CubatureSphere(Benchmark):
    params = (
        [
            "gk15",
            "gk21",
            "genz-malik",
        ],
        [1e-9, 1e-10, 1e-11],
    )

    param_names = ["rule", "rtol"]

    def setup(self, rule, rtol):
        self.a = mx.array([0, 0, 0])
        self.b = mx.array([1, 2*mx.pi, mx.pi])
        self.rule = rule
        self.rtol = rtol
        self.atol = 0
        self.pool = ThreadPoolExecutor(2)

    def f(self, x):
        r = x[:, 0]
        phi = x[:, 2]

        return r**2 * mx.sin(phi)

    def time_plain(self, rule, rtol):
        cubature(
            f=self.f,
            a=self.a,
            b=self.b,
            rule=self.rule,
            rtol=self.rtol,
            atol=self.atol,
        )

    def time_threads(self, rule, rtol):
        cubature(
            f=self.f,
            a=self.a,
            b=self.b,
            rule=self.rule,
            rtol=self.rtol,
            atol=self.atol,
            workers=self.pool.map,
        )

    def track_subdivisions(self, rule, rtol):
        res = cubature(
            f=self.f,
            a=self.a,
            b=self.b,
            rule=self.rule,
            rtol=self.rtol,
            atol=self.atol,
        )
        return res.subdivisions


class CubatureOscillatory(Benchmark):
    params = (
        # rule
        [
            "genz-malik",
            "gk15",
            "gk21",
        ],

        # input dimension of integrand (ndim)
        [1, 3, 5],

        # output dimension of integrand (fdim)
        [1, 8],

        # rtol
        [1e-10, 1e-11],
    )

    param_names = ["rule", "ndim", "fdim", "rtol"]

    def setup(self, rule, ndim, fdim, rtol):
        self.ndim = ndim
        self.fdim = fdim

        self.rtol = rtol
        self.atol = 0

        self.a = mx.zeros(self.ndim)
        self.b = mx.repeat(1, self.ndim)
        self.rule = rule

        self.pool = ThreadPoolExecutor(2)

        if rule == "genz-malik" and ndim == 1:
            raise SkipNotImplemented(f"{rule} not defined for 1D integrals")

        if (rule == "gk-15" or rule == "gk-15") and ndim > 5:
            raise SkipNotImplemented(f"{rule} uses too much memory for ndim > 5")

    def f(self, x):
        npoints, ndim = x.shape[0], x.shape[-1]

        r = mx.repeat(0.5, self.fdim)
        alphas = mx.repeat(0.1, self.fdim * ndim).reshape(self.fdim, ndim)
        x_reshaped = x.reshape(npoints, *([1]*(len(alphas.shape) - 1)), ndim)

        return mx.cos(2*mx.pi*r + mx.sum(alphas * x_reshaped, axis=-1))

    def time_plain(self, rule, ndim, fdim, rtol):
        cubature(
            f=self.f,
            a=self.a,
            b=self.b,
            rule=self.rule,
            rtol=self.rtol,
            atol=self.atol,
        )

    def time_threads(self, rule, ndim, fdim, rtol):
        cubature(
            f=self.f,
            a=self.a,
            b=self.b,
            rule=self.rule,
            rtol=self.rtol,
            atol=self.atol,
            workers=self.pool.map,
        )

    def track_subdivisions(self, rule, ndim, fdim, rtol):
        return cubature(
            f=self.f,
            a=self.a,
            b=self.b,
            rule=self.rule,
            rtol=self.rtol,
            atol=self.atol,
        ).subdivisions
