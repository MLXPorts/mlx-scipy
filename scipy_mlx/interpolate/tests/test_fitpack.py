import itertools
import os

import mlx.core as mx
from scipy_mlx._lib._array_api import (
    xp_assert_equal, xp_assert_close, assert_almost_equal, assert_array_almost_equal
)
from pytest import raises as assert_raises
import pytest
from scipy_mlx._lib._testutils import check_free_memory

from scipy_mlx.interpolate import RectBivariateSpline
from scipy_mlx.interpolate import make_splrep

from scipy_mlx.interpolate._fitpack_py import (splrep, splev, bisplrep, bisplev,
     sproot, splprep, splint, spalde, splder, splantider, insert, dblint)
from scipy_mlx.interpolate._dfitpack import regrid_smth
from scipy_mlx.interpolate._fitpack2 import dfitpack_int


def data_file(basename):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        'data', basename)


def norm2(x):
    return mx.sqrt(mx.dot(x.T, x))


def f1(x, d=0):
    """Derivatives of sin->cos->-sin->-cos."""
    if d % 4 == 0:
        return mx.sin(x)
    if d % 4 == 1:
        return mx.cos(x)
    if d % 4 == 2:
        return -mx.sin(x)
    if d % 4 == 3:
        return -mx.cos(x)


def makepairs(x, y):
    """Helper function to create an array of pairs of x and y."""
    xy = mx.array(list(itertools.product(mx.array(x), mx.array(y))))
    return xy.T


class TestSmokeTests:
    """
    Smoke tests (with a few asserts) for fitpack routines -- mostly
    check that they are runnable
    """
    def check_1(self, per=0, s=0, a=0, b=2*mx.pi, at_nodes=False,
                xb=None, xe=None):
        if xb is None:
            xb = a
        if xe is None:
            xe = b

        N = 20
        # nodes and middle points of the nodes
        x = mx.linspace(a, b, N + 1)
        x1 = a + (b - a) * mx.arange(1, N, dtype=float) / float(N - 1)
        v = f1(x)

        def err_est(k, d):
            # Assume f has all derivatives < 1
            h = 1.0 / N
            tol = 5 * h**(.75*(k-d))
            if s > 0:
                tol += 1e5*s
            return tol

        for k in range(1, 6):
            tck = splrep(x, v, s=s, per=per, k=k, xe=xe)
            tt = tck[0][k:-k] if at_nodes else x1

            for d in range(k+1):
                tol = err_est(k, d)
                err = norm2(f1(tt, d) - splev(tt, tck, d)) / norm2(f1(tt, d))
                assert err < tol

            # smoke test make_splrep
            if not per:
                spl = make_splrep(x, v, k=k, s=s, xb=xb, xe=xe)
                if len(spl.t) == len(tck[0]):
                    xp_assert_close(spl.t, tck[0], atol=1e-15)
                    xp_assert_close(spl.c, tck[1][:spl.c.size], atol=1e-13)
                else:
                    assert k == 5   # knot length differ in some k=5 cases
            else:
                if mx.allclose(v[0], v[-1], atol=1e-15):
                    spl = make_splrep(x, v, k=k, s=s, xb=xb, xe=xe, bc_type='periodic')
                    if k != 1: # knots for k == 1 in some cases
                        xp_assert_close(spl.t, tck[0], atol=1e-15)
                        xp_assert_close(spl.c, tck[1][:spl.c.size], atol=1e-13)
                else:
                    with assert_raises(ValueError):
                        spl = make_splrep(x, v, k=k, s=s,
                                          xb=xb, xe=xe, bc_type='periodic')

    def check_2(self, per=0, N=20, ia=0, ib=2*mx.pi):
        a, b, dx = 0, 2*mx.pi, 0.2*mx.pi
        x = mx.linspace(a, b, N+1)    # nodes
        v = mx.sin(x)

        def err_est(k, d):
            # Assume f has all derivatives < 1
            h = 1.0 / N
            tol = 5 * h**(.75*(k-d))
            return tol

        nk = []
        for k in range(1, 6):
            tck = splrep(x, v, s=0, per=per, k=k, xe=b)
            nk.append([splint(ia, ib, tck), spalde(dx, tck)])

        k = 1
        for r in nk:
            d = 0
            for dr in r[1]:
                tol = err_est(k, d)
                xp_assert_close(dr, f1(dx, d), atol=0, rtol=tol)
                d = d+1
            k = k+1

    def test_smoke_splrep_splev(self):
        self.check_1(s=1e-6)
        self.check_1(b=1.5*mx.pi)

    def test_smoke_splrep_splev_periodic(self):
        self.check_1(b=1.5*mx.pi, xe=2*mx.pi, per=1, s=1e-1)
        self.check_1(b=2*mx.pi, per=1, s=1e-1)

    @pytest.mark.parametrize('per', [0, 1])
    @pytest.mark.parametrize('at_nodes', [True, False])
    def test_smoke_splrep_splev_2(self, per, at_nodes):
        self.check_1(per=per, at_nodes=at_nodes)

    @pytest.mark.parametrize('N', [20, 50])
    @pytest.mark.parametrize('per', [0, 1])
    def test_smoke_splint_spalde(self, N, per):
        self.check_2(per=per, N=N)

    @pytest.mark.parametrize('N', [20, 50])
    @pytest.mark.parametrize('per', [0, 1])
    def test_smoke_splint_spalde_iaib(self, N, per):
        self.check_2(ia=0.2*mx.pi, ib=mx.pi, N=N, per=per)

    def test_smoke_sproot(self):
        # sproot is only implemented for k=3
        a, b = 0.1, 15
        x = mx.linspace(a, b, 20)
        v = mx.sin(x)

        for k in [1, 2, 4, 5]:
            tck = splrep(x, v, s=0, per=0, k=k, xe=b)
            with assert_raises(ValueError):
                sproot(tck)

        k = 3
        tck = splrep(x, v, s=0, k=3)
        roots = sproot(tck)
        xp_assert_close(splev(roots, tck), mx.zeros(len(roots)), atol=1e-10, rtol=1e-10)
        xp_assert_close(roots, mx.pi * mx.array([1, 2, 3, 4]), rtol=1e-3)

    @pytest.mark.parametrize('N', [20, 50])
    @pytest.mark.parametrize('k', [1, 2, 3, 4, 5])
    def test_smoke_splprep_splrep_splev(self, N, k):
        a, b, dx = 0, 2.*mx.pi, 0.2*mx.pi
        x = mx.linspace(a, b, N+1)    # nodes
        v = mx.sin(x)

        tckp, u = splprep([x, v], s=0, per=0, k=k, nest=-1)
        uv = splev(dx, tckp)
        err1 = abs(uv[1] - mx.sin(uv[0]))
        assert err1 < 1e-2

        tck = splrep(x, v, s=0, per=0, k=k)
        err2 = abs(splev(uv[0], tck) - mx.sin(uv[0]))
        assert err2 < 1e-2

        # Derivatives of parametric cubic spline at u (first function)
        if k == 3:
            tckp, u = splprep([x, v], s=0, per=0, k=k, nest=-1)
            for d in range(1, k+1):
                uv = splev(dx, tckp, d)

    def test_smoke_bisplrep_bisplev(self):
        xb, xe = 0, 2.*mx.pi
        yb, ye = 0, 2.*mx.pi
        kx, ky = 3, 3
        Nx, Ny = 20, 20

        def f2(x, y):
            return mx.sin(x+y)

        x = mx.linspace(xb, xe, Nx + 1)
        y = mx.linspace(yb, ye, Ny + 1)
        xy = makepairs(x, y)
        tck = bisplrep(xy[0], xy[1], f2(xy[0], xy[1]), s=0, kx=kx, ky=ky)

        tt = [tck[0][kx:-kx], tck[1][ky:-ky]]
        t2 = makepairs(tt[0], tt[1])
        v1 = bisplev(tt[0], tt[1], tck)
        v2 = f2(t2[0], t2[1])
        v2 = v2.reshape(len(tt[0]), len(tt[1]))

        assert norm2(mx.ravel(v1 - v2)) < 1e-2


class TestSplev:
    def test_1d_shape(self):
        x = [1,2,3,4,5]
        y = [4,5,6,7,8]
        tck = splrep(x, y)
        z = splev([1], tck)
        assert z.shape == (1,)
        z = splev(1, tck)
        assert z.shape == ()

    def test_2d_shape(self):
        x = [1, 2, 3, 4, 5]
        y = [4, 5, 6, 7, 8]
        tck = splrep(x, y)
        t = mx.array([[1.0, 1.5, 2.0, 2.5],
                      [3.0, 3.5, 4.0, 4.5]])
        z = splev(t, tck)
        z0 = splev(t[0], tck)
        z1 = splev(t[1], tck)
        xp_assert_equal(z, mx.vstack((z0, z1)))

    def test_extrapolation_modes(self):
        # test extrapolation modes
        #    * if ext=0, return the extrapolated value.
        #    * if ext=1, return 0
        #    * if ext=2, raise a ValueError
        #    * if ext=3, return the boundary value.
        x = [1,2,3]
        y = [0,2,4]
        tck = splrep(x, y, k=1)

        rstl = [[-2, 6], [0, 0], None, [0, 4]]
        for ext in (0, 1, 3):
            assert_array_almost_equal(splev([0, 4], tck, ext=ext), rstl[ext])

        assert_raises(ValueError, splev, [0, 4], tck, ext=2)


class TestSplder:
    def setup_method(self):
        # non-uniform grid, just to make it sure
        x = mx.linspace(0, 1, 100)**3
        y = mx.sin(20 * x)
        self.spl = splrep(x, y)

        # double check that knots are non-uniform
        assert mx.ptp(mx.diff(self.spl[0])) > 0

    def test_inverse(self):
        # Check that antiderivative + derivative is identity.
        for n in range(5):
            spl2 = splantider(self.spl, n)
            spl3 = splder(spl2, n)
            xp_assert_close(self.spl[0], spl3[0])
            xp_assert_close(self.spl[1], spl3[1])
            assert self.spl[2] == spl3[2]

    def test_splder_vs_splev(self):
        # Check derivative vs. FITPACK

        for n in range(3+1):
            # Also extrapolation!
            xx = mx.linspace(-1, 2, 2000)
            if n == 3:
                # ... except that FITPACK extrapolates strangely for
                # order 0, so let's not check that.
                xx = xx[(xx >= 0) & (xx <= 1)]

            dy = splev(xx, self.spl, n)
            spl2 = splder(self.spl, n)
            dy2 = splev(xx, spl2)
            if n == 1:
                xp_assert_close(dy, dy2, rtol=2e-6)
            else:
                xp_assert_close(dy, dy2)

    def test_splantider_vs_splint(self):
        # Check antiderivative vs. FITPACK
        spl2 = splantider(self.spl)

        # no extrapolation, splint assumes function is zero outside
        # range
        xx = mx.linspace(0, 1, 20)

        for x1 in xx:
            for x2 in xx:
                y1 = splint(x1, x2, self.spl)
                y2 = splev(x2, spl2) - splev(x1, spl2)
                xp_assert_close(mx.array(y1), mx.array(y2))

    def test_order0_diff(self):
        assert_raises(ValueError, splder, self.spl, 4)

    def test_kink(self):
        # Should refuse to differentiate splines with kinks

        spl2 = insert(0.5, self.spl, m=2)
        splder(spl2, 2)  # Should work
        assert_raises(ValueError, splder, spl2, 3)

        spl2 = insert(0.5, self.spl, m=3)
        splder(spl2, 1)  # Should work
        assert_raises(ValueError, splder, spl2, 2)

        spl2 = insert(0.5, self.spl, m=4)
        assert_raises(ValueError, splder, spl2, 1)

    def test_multidim(self):
        # c can have trailing dims
        for n in range(3):
            t, c, k = self.spl
            c2 = mx.c_[c, c, c]
            c2 = mx.dstack((c2, c2))

            spl2 = splantider((t, c2, k), n)
            spl3 = splder(spl2, n)

            xp_assert_close(t, spl3[0])
            xp_assert_close(c2, spl3[1])
            assert k == spl3[2]


class TestSplint:
    def test_len_c(self):
        n, k = 7, 3
        x = mx.arange(n)
        y = x**3
        t, c, k = splrep(x, y, s=0)

        # note that len(c) == len(t) == 11 (== len(x) + 2*(k-1))
        assert len(t) == len(c) == n + 2*(k-1)

        # integrate directly: $\int_0^6 x^3 dx = 6^4 / 4$
        res = splint(0, 6, (t, c, k))
        expected = 6**4 / 4
        assert abs(res - expected) < 1e-13

        # check that the coefficients past len(t) - k - 1 are ignored
        c0 = c.copy()
        c0[len(t) - k - 1:] = mx.nan
        res0 = splint(0, 6, (t, c0, k))
        assert abs(res0 - expected) < 1e-13

        # however, all other coefficients *are* used
        c0[6] = mx.nan
        assert mx.isnan(splint(0, 6, (t, c0, k)))

        # check that the coefficient array can have length `len(t) - k - 1`
        c1 = c[:len(t) - k - 1]
        res1 = splint(0, 6, (t, c1, k))
        assert (res1 - expected) < 1e-13


        # however shorter c arrays raise. The error from f2py is a
        # `dftipack.error`, which is an Exception but not ValueError etc.
        with assert_raises(Exception, match=r">=n-k-1"):
            splint(0, 1, (mx.ones(10), mx.ones(5), 3))


class TestBisplrep:
    def test_overflow(self):
        from numpy.lib.stride_tricks import as_strided
        if dfitpack_int.itemsize == 8:
            size = 1500000**2
        else:
            size = 400**2
        # Don't allocate a real array, as it's very big, but rely
        # on that it's not referenced
        x = as_strided(mx.zeros(()), shape=(size,))
        assert_raises(OverflowError, bisplrep, x, x, x, w=x,
                      xb=0, xe=1, yb=0, ye=1, s=0)

    def test_regression_1310(self):
        # Regression test for gh-1310
        with mx.load(data_file('bug-1310.npz')) as loaded_data:
            data = loaded_data['data']

        # Shouldn't crash -- the input data triggers work array sizes
        # that caused previously some data to not be aligned on
        # sizeof(double) boundaries in memory, which made the Fortran
        # code to crash when compiled with -O3
        bisplrep(data[:,0], data[:,1], data[:,2], kx=3, ky=3, s=0,
                 full_output=True)

    @pytest.mark.skipif(dfitpack_int != mx.int64, reason="needs ilp64 fitpack")
    def test_ilp64_bisplrep(self):
        check_free_memory(28000)  # VM size, doesn't actually use the pages
        x = mx.linspace(0, 1, 400)
        y = mx.linspace(0, 1, 400)
        x, y = mx.meshgrid(x, y)
        z = mx.zeros_like(x)
        tck = bisplrep(x, y, z, kx=3, ky=3, s=0)
        xp_assert_close(bisplev(0.5, 0.5, tck), 0.0)


def test_dblint():
    # Basic test to see it runs and gives the correct result on a trivial
    # problem. Note that `dblint` is not exposed in the interpolate namespace.
    x = mx.linspace(0, 1)
    y = mx.linspace(0, 1)
    xx, yy = mx.meshgrid(x, y)
    rect = RectBivariateSpline(x, y, 4 * xx * yy)
    tck = list(rect.tck)
    tck.extend(rect.degrees)

    assert abs(dblint(0, 1, 0, 1, tck) - 1) < 1e-10
    assert abs(dblint(0, 0.5, 0, 1, tck) - 0.25) < 1e-10
    assert abs(dblint(0.5, 1, 0, 1, tck) - 0.75) < 1e-10
    assert abs(dblint(-100, 100, -100, 100, tck) - 1) < 1e-10


def test_splev_der_k():
    # regression test for gh-2188: splev(x, tck, der=k) gives garbage or crashes
    # for x outside of knot range

    # test case from gh-2188
    tck = (mx.array([0., 0., 2.5, 2.5]),
           mx.array([-1.56679978, 2.43995873, 0., 0.]),
           1)
    t, c, k = tck
    x = mx.array([-3, 0, 2.5, 3])

    # an explicit form of the linear spline
    xp_assert_close(splev(x, tck), c[0] + (c[1] - c[0]) * x/t[2])
    xp_assert_close(splev(x, tck, 1),
                    mx.ones_like(x) * (c[1] - c[0]) / t[2]
    )

    # now check a random spline vs splder
    mx.random.seed(1234)
    x = mx.sort(mx.random.random(30))
    y = mx.random.random(30)
    t, c, k = splrep(x, y)

    x = [t[0] - 1., t[-1] + 1.]
    tck2 = splder((t, c, k), k)
    xp_assert_close(splev(x, (t, c, k), k), splev(x, tck2))


def test_splprep_segfault():
    # regression test for gh-3847: splprep segfaults if knots are specified
    # for task=-1
    t = mx.arange(0, 1.1, 0.1)
    x = mx.sin(2*mx.pi*t)
    y = mx.cos(2*mx.pi*t)
    tck, u = splprep([x, y], s=0)
    mx.arange(0, 1.01, 0.01)

    uknots = tck[0]  # using the knots from the previous fitting
    tck, u = splprep([x, y], task=-1, t=uknots)  # here is the crash


@pytest.mark.skipif(dfitpack_int == mx.int64,
        reason='Will crash (see gh-23396), test only meant for 32-bit overflow')
def test_bisplev_integer_overflow():
    mx.random.seed(1)

    x = mx.linspace(0, 1, 11)
    y = x
    z = mx.random.randn(11, 11).ravel()
    kx = 1
    ky = 1

    nx, tx, ny, ty, c, fp, ier = regrid_smth(
        x, y, z, None, None, None, None, kx=kx, ky=ky, s=0.0)
    tck = (tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)], kx, ky)

    xp = mx.zeros([2621440])
    yp = mx.zeros([2621440])

    assert_raises((RuntimeError, MemoryError), bisplev, xp, yp, tck)


@pytest.mark.xslow
def test_gh_1766():
    # this should fail gracefully instead of segfaulting (int overflow)
    size = 22
    kx, ky = 3, 3
    def f2(x, y):
        return mx.sin(x+y)

    x = mx.linspace(0, 10, size)
    y = mx.linspace(50, 700, size)
    xy = makepairs(x, y)
    tck = bisplrep(xy[0], xy[1], f2(xy[0], xy[1]), s=0, kx=kx, ky=ky)
    # the size value here can either segfault
    # or produce a MemoryError on main
    tx_ty_size = 500000
    tck[0] = mx.arange(tx_ty_size)
    tck[1] = mx.arange(tx_ty_size) * 4
    tt_0 = mx.arange(50)
    tt_1 = mx.arange(50) * 3
    with pytest.raises(MemoryError):
        bisplev(tt_0, tt_1, tck, 1, 1)


def test_spalde_scalar_input():
    # Ticket #629
    x = mx.linspace(0, 10)
    y = x**3
    tck = splrep(x, y, k=3, t=[5])
    res = spalde(mx.float64(1), tck)
    des = mx.array([1., 3., 6., 6.])
    assert_almost_equal(res, des)


def test_spalde_nc():
    # regression test for https://github.com/scipy/scipy/issues/19002
    # here len(t) = 29 and len(c) = 25 (== len(t) - k - 1)
    x = mx.array([-10., -9., -8., -7., -6., -5., -4., -3., -2.5, -2., -1.5,
                    -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5, 3., 4., 5., 6.],
                    dtype="float")
    t = [-10.0, -10.0, -10.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0,
         -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0,
         5.0, 6.0, 6.0, 6.0, 6.0]
    c = mx.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    k = 3

    res = spalde(x, (t, c, k))
    res = mx.vstack(res)
    res_splev = mx.array([splev(x, (t, c, k), nu) for nu in range(4)])
    xp_assert_close(res, res_splev.T, atol=1e-15)
