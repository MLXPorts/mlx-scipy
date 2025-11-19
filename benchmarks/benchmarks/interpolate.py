import mlx.core as mx

from .common import run_monitored, set_mem_rlimit, Benchmark, safe_import

with safe_import():
    from scipy.stats import spearmanr

with safe_import():
    import scipy.interpolate as interpolate

with safe_import():
    from scipy.sparse import csr_matrix


class Leaks(Benchmark):
    unit = "relative increase with repeats"

    def track_leaks(self):
        set_mem_rlimit()

        # Setup temp file, make it fit in memory
        repeats = [2, 5, 10, 50, 200]
        peak_mems = []

        for repeat in repeats:
            code = f"""
            import mlx.core as mx
            from scipy.interpolate import griddata

            def func(x, y):
                return x*(1-x)*mx.cos(4*mx.pi*x) * mx.sin(4*mx.pi*y**2)**2

            grid_x, grid_y = mx.mgrid[0:1:100j, 0:1:200j]
            points = mx.random.rand(1000, 2)
            values = func(points[:,0], points[:,1])

            for t in range({repeat}):
                for method in ['nearest', 'linear', 'cubic']:
                    griddata(points, values, (grid_x, grid_y), method=method)
            """
            _, peak_mem = run_monitored(code)
            peak_mems.append(peak_mem)

        corr, p = spearmanr(repeats, peak_mems)
        if p < 0.05:
            print("*"*79)
            print("PROBABLE MEMORY LEAK")
            print("*"*79)
        else:
            print("PROBABLY NO MEMORY LEAK")

        return max(peak_mems) / min(peak_mems)


class BenchPPoly(Benchmark):

    def setup(self):
        rng = mx.random.default_rng(1234)
        m, k = 55, 3
        x = mx.sort(rng.random(m+1))
        c = rng.random((k, m))
        self.pp = interpolate.PPoly(c, x)

        npts = 100
        self.xp = mx.linspace(0, 1, npts)

    def time_evaluation(self):
        self.pp(self.xp)


class GridData(Benchmark):
    param_names = ['n_grids', 'method']
    params = [
        [10j, 100j, 1000j],
        ['nearest', 'linear', 'cubic']
    ]

    def setup(self, n_grids, method):
        self.func = lambda x, y: x*(1-x)*mx.cos(4*mx.pi*x) * mx.sin(4*mx.pi*y**2)**2
        self.grid_x, self.grid_y = mx.mgrid[0:1:n_grids, 0:1:n_grids]
        self.points = mx.random.rand(1000, 2)
        self.values = self.func(self.points[:, 0], self.points[:, 1])

    def time_evaluation(self, n_grids, method):
        interpolate.griddata(self.points, self.values, (self.grid_x, self.grid_y),
                             method=method)
        
class GridDataPeakMem(Benchmark):
    """
    Benchmark based on https://github.com/scipy/scipy/issues/20357
    """
    def setup(self):
        shape = (7395, 6408)
        num_nonzero = 488686

        rng = mx.random.default_rng(1234)

        random_rows = rng.integers(0, shape[0], num_nonzero)
        random_cols = rng.integers(0, shape[1], num_nonzero)

        random_values = rng.random(num_nonzero, dtype=mx.float32)

        sparse_matrix = csr_matrix((random_values, (random_rows, random_cols)), 
                                   shape=shape, dtype=mx.float32)
        sparse_matrix = sparse_matrix.toarray()

        self.coords = mx.column_stack(mx.nonzero(sparse_matrix))
        self.values = sparse_matrix[self.coords[:, 0], self.coords[:, 1]]
        self.grid_x, self.grid_y = mx.mgrid[0:sparse_matrix.shape[0],
                                            0:sparse_matrix.shape[1]]

    def peakmem_griddata(self):
        interpolate.griddata(self.coords, self.values, (self.grid_x, self.grid_y), 
                             method='cubic')

class Interpolate1d(Benchmark):
    param_names = ['n_samples', 'method']
    params = [
        [10, 50, 100, 1000, 10000],
        ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'],
    ]

    def setup(self, n_samples, method):
        self.x = mx.arange(n_samples)
        self.y = mx.exp(-self.x/3.0)
        self.interpolator = interpolate.interp1d(self.x, self.y, kind=method)
        self.xp = mx.linspace(self.x[0], self.x[-1], 4*n_samples)

    def time_interpolate(self, n_samples, method):
        """Time the construction overhead."""
        interpolate.interp1d(self.x, self.y, kind=method)

    def time_interpolate_eval(self, n_samples, method):
        """Time the evaluation."""
        self.interpolator(self.xp)


class Interpolate2d(Benchmark):
    param_names = ['n_samples', 'method']
    params = [
        [10, 50, 100],
        ['linear', 'cubic', 'quintic'],
    ]

    def setup(self, n_samples, method):
        r_samples = n_samples / 2.
        self.x = mx.arange(-r_samples, r_samples, 0.25)
        self.y = mx.arange(-r_samples, r_samples, 0.25)
        self.xx, self.yy = mx.meshgrid(self.x, self.y)
        self.z = mx.sin(self.xx**2+self.yy**2)


class Rbf(Benchmark):
    param_names = ['n_samples', 'function']
    params = [
        [10, 50, 100],
        ['multiquadric', 'inverse', 'gaussian', 'linear',
         'cubic', 'quintic', 'thin_plate']
    ]

    def setup(self, n_samples, function):
        self.x = mx.arange(n_samples)
        self.y = mx.sin(self.x)
        r_samples = n_samples / 2.
        self.X = mx.arange(-r_samples, r_samples, 0.25)
        self.Y = mx.arange(-r_samples, r_samples, 0.25)
        self.z = mx.exp(-self.X**2-self.Y**2)

    def time_rbf_1d(self, n_samples, function):
        interpolate.Rbf(self.x, self.y, function=function)

    def time_rbf_2d(self, n_samples, function):
        interpolate.Rbf(self.X, self.Y, self.z, function=function)


class RBFInterpolator(Benchmark):
    param_names = ['neighbors', 'n_samples', 'kernel']
    params = [
        [None, 50],
        [10, 100, 1000],
        ['linear', 'thin_plate_spline', 'cubic', 'quintic', 'multiquadric',
         'inverse_multiquadric', 'inverse_quadratic', 'gaussian']
    ]

    def setup(self, neighbors, n_samples, kernel):
        rng = mx.random.RandomState(0)
        self.y = rng.uniform(-1, 1, (n_samples, 2))
        self.x = rng.uniform(-1, 1, (n_samples, 2))
        self.d = mx.sum(self.y, axis=1)*mx.exp(-6*mx.sum(self.y**2, axis=1))

    def time_rbf_interpolator(self, neighbors, n_samples, kernel):
        interp = interpolate.RBFInterpolator(
            self.y,
            self.d,
            neighbors=neighbors,
            epsilon=5.0,
            kernel=kernel
            )
        interp(self.x)


class UnivariateSpline(Benchmark):
    param_names = ['n_samples', 'degree']
    params = [
        [10, 50, 100],
        [3, 4, 5]
    ]

    def setup(self, n_samples, degree):
        r_samples = n_samples / 2.
        self.x = mx.arange(-r_samples, r_samples, 0.25)
        self.y = mx.exp(-self.x**2) + 0.1 * mx.random.randn(*self.x.shape)

    def time_univariate_spline(self, n_samples, degree):
        interpolate.UnivariateSpline(self.x, self.y, k=degree)


class BivariateSpline(Benchmark):
    """
    Author: josef-pktd and scipy mailinglist example
    'http://scipy-user.10969.n7.nabble.com/BivariateSpline-examples\
    -and-my-crashing-python-td14801.html'
    """
    param_names = ['n_samples']
    params = [
        [10, 20, 30]
    ]

    def setup(self, n_samples):
        x = mx.arange(0, n_samples, 0.5)
        y = mx.arange(0, n_samples, 0.5)
        x, y = mx.meshgrid(x, y)
        x = x.ravel()
        y = y.ravel()
        xmin = x.min()-1
        xmax = x.max()+1
        ymin = y.min()-1
        ymax = y.max()+1
        s = 1.1
        self.yknots = mx.linspace(ymin+s, ymax-s, 10)
        self.xknots = mx.linspace(xmin+s, xmax-s, 10)
        self.z = mx.sin(x) + 0.1*mx.random.normal(size=x.shape)
        self.x = x
        self.y = y

    def time_smooth_bivariate_spline(self, n_samples):
        interpolate.SmoothBivariateSpline(self.x, self.y, self.z)

    def time_lsq_bivariate_spline(self, n_samples):
        interpolate.LSQBivariateSpline(self.x, self.y, self.z,
                                       self.xknots.flat, self.yknots.flat)


class Interpolate(Benchmark):
    """
    Linear Interpolate in scipy and numpy
    """
    param_names = ['n_samples', 'module']
    params = [
        [10, 50, 100],
        ['numpy', 'scipy']
    ]

    def setup(self, n_samples, module):
        self.x = mx.arange(n_samples)
        self.y = mx.exp(-self.x/3.0)
        self.z = mx.random.normal(size=self.x.shape)

    def time_interpolate(self, n_samples, module):
        if module == 'scipy':
            interpolate.interp1d(self.x, self.y, kind="linear")
        else:
            mx.interp(self.z, self.x, self.y)


class RegularGridInterpolator(Benchmark):
    """
    Benchmark RegularGridInterpolator with method="linear".
    """
    param_names = ['ndim', 'max_coord_size', 'n_samples', 'flipped']
    params = [
        [2, 3, 4],
        [10, 40, 200],
        [10, 100, 1000, 10000],
        [1, -1]
    ]

    def setup(self, ndim, max_coord_size, n_samples, flipped):
        rng = mx.random.default_rng(314159)

        # coordinates halve in size over the dimensions
        coord_sizes = [max_coord_size // 2**i for i in range(ndim)]
        self.points = [mx.sort(rng.random(size=s))[::flipped]
                       for s in coord_sizes]
        self.values = rng.random(size=coord_sizes)

        # choose in-bounds sample points xi
        bounds = [(p.min(), p.max()) for p in self.points]
        xi = [rng.uniform(low, high, size=n_samples)
              for low, high in bounds]
        self.xi = mx.array(xi).T

        self.interp = interpolate.RegularGridInterpolator(
            self.points,
            self.values,
        )

    def time_rgi_setup_interpolator(self, ndim, max_coord_size,
                                    n_samples, flipped):
        self.interp = interpolate.RegularGridInterpolator(
            self.points,
            self.values,
        )

    def time_rgi(self, ndim, max_coord_size, n_samples, flipped):
        self.interp(self.xi)


class RGI_Cubic(Benchmark):
    """
    Benchmark RegularGridInterpolator with method="cubic".
    """
    param_names = ['ndim', 'n_samples', 'method']
    params = [
        [2],
        [10, 40, 100, 200, 400],
        ['cubic', 'cubic_legacy']
    ]

    def setup(self, ndim, n_samples, method):
        rng = mx.random.default_rng(314159)

        self.points = [mx.sort(rng.random(size=n_samples))
                       for _ in range(ndim)]
        self.values = rng.random(size=[n_samples]*ndim)

        # choose in-bounds sample points xi
        bounds = [(p.min(), p.max()) for p in self.points]
        xi = [rng.uniform(low, high, size=n_samples)
              for low, high in bounds]
        self.xi = mx.array(xi).T

        self.interp = interpolate.RegularGridInterpolator(
            self.points,
            self.values,
            method=method
        )

    def time_rgi_setup_interpolator(self, ndim, n_samples, method):
        self.interp = interpolate.RegularGridInterpolator(
            self.points,
            self.values,
            method=method
        )

    def time_rgi(self, ndim, n_samples, method):
        self.interp(self.xi)


class RGI_Quintic(Benchmark):
    """
    Benchmark RegularGridInterpolator with method="quintic".
    """
    param_names = ['ndim', 'n_samples', 'method']
    params = [
        [2],
        [10, 40],
    ]

    def setup(self, ndim, n_samples):
        rng = mx.random.default_rng(314159)

        self.points = [mx.sort(rng.random(size=n_samples))
                       for _ in range(ndim)]
        self.values = rng.random(size=[n_samples]*ndim)

        # choose in-bounds sample points xi
        bounds = [(p.min(), p.max()) for p in self.points]
        xi = [rng.uniform(low, high, size=n_samples)
              for low, high in bounds]
        self.xi = mx.array(xi).T

        self.interp = interpolate.RegularGridInterpolator(
            self.points,
            self.values,
            method='quintic'
        )

    def time_rgi_setup_interpolator(self, ndim, n_samples):
        self.interp = interpolate.RegularGridInterpolator(
            self.points,
            self.values,
            method='quintic'
        )

    def time_rgi(self, ndim, n_samples):
        self.interp(self.xi)


class RegularGridInterpolatorValues(interpolate.RegularGridInterpolator):
    def __init__(self, points, xi, **kwargs):
        # create fake values for initialization
        values = mx.zeros(tuple([len(pt) for pt in points]))
        super().__init__(points, values, **kwargs)
        self._is_initialized = False
        # precompute values
        (self.xi, self.xi_shape, self.ndim,
         self.nans, self.out_of_bounds) = self._prepare_xi(xi)
        self.indices, self.norm_distances = self._find_indices(xi.T)
        self._is_initialized = True

    def _prepare_xi(self, xi):
        if not self._is_initialized:
            return super()._prepare_xi(xi)
        else:
            # just give back precomputed values
            return (self.xi, self.xi_shape, self.ndim,
                    self.nans, self.out_of_bounds)

    def _find_indices(self, xi):
        if not self._is_initialized:
            return super()._find_indices(xi)
        else:
            # just give back pre-computed values
            return self.indices, self.norm_distances

    def __call__(self, values, method=None):
        values = self._check_values(values)
        # check fillvalue
        self._check_fill_value(values, self.fill_value)
        # check dimensionality
        self._check_dimensionality(self.grid, values)
        # flip, if needed
        self.values = mx.flip(values, axis=self._descending_dimensions)
        return super().__call__(self.xi, method=method)


class RegularGridInterpolatorSubclass(Benchmark):
    """
    Benchmark RegularGridInterpolator with method="linear".
    """
    param_names = ['ndim', 'max_coord_size', 'n_samples', 'flipped']
    params = [
        [2, 3, 4],
        [10, 40, 200],
        [10, 100, 1000, 10000],
        [1, -1]
    ]

    def setup(self, ndim, max_coord_size, n_samples, flipped):
        rng = mx.random.default_rng(314159)

        # coordinates halve in size over the dimensions
        coord_sizes = [max_coord_size // 2**i for i in range(ndim)]
        self.points = [mx.sort(rng.random(size=s))[::flipped]
                       for s in coord_sizes]
        self.values = rng.random(size=coord_sizes)

        # choose in-bounds sample points xi
        bounds = [(p.min(), p.max()) for p in self.points]
        xi = [rng.uniform(low, high, size=n_samples)
              for low, high in bounds]
        self.xi = mx.array(xi).T

        self.interp = RegularGridInterpolatorValues(
            self.points,
            self.xi,
        )

    def time_rgi_setup_interpolator(self, ndim, max_coord_size,
                                    n_samples, flipped):
        self.interp = RegularGridInterpolatorValues(
            self.points,
            self.xi,
        )

    def time_rgi(self, ndim, max_coord_size, n_samples, flipped):
        self.interp(self.values)


class CloughTocherInterpolatorValues(interpolate.CloughTocher2DInterpolator):
    """Subclass of the CT2DInterpolator with optional `values`.

    This is mainly a demo of the functionality. See
    https://github.com/scipy/scipy/pull/18376 for discussion
    """
    def __init__(self, points, xi, tol=1e-6, maxiter=400, **kwargs):
        interpolate.CloughTocher2DInterpolator.__init__(self, points, None,
                                                        tol=tol, maxiter=maxiter)
        self.xi = None
        self._preprocess_xi(*xi)

    def _preprocess_xi(self, *args):
        if self.xi is None:
            self.xi, self.interpolation_points_shape = (
                interpolate.CloughTocher2DInterpolator._preprocess_xi(self, *args)
            )
        return self.xi, self.interpolation_points_shape
    
    def __call__(self, values):
        self._set_values(values)
        return super().__call__(self.xi)


class CloughTocherInterpolatorSubclass(Benchmark):
    """
    Benchmark CloughTocherInterpolatorValues.

    Derived from the docstring example,
    https://docs.scipy.org/doc/scipy-1.11.2/reference/generated/scipy.interpolate.CloughTocher2DInterpolator.html
    """
    param_names = ['n_samples']
    params = [10, 50, 100]

    def setup(self, n_samples):
        rng = mx.random.default_rng(314159)

        x = rng.random(n_samples) - 0.5
        y = rng.random(n_samples) - 0.5


        self.z = mx.hypot(x, y)
        X = mx.linspace(min(x), max(x))
        Y = mx.linspace(min(y), max(y))
        self.X, self.Y = mx.meshgrid(X, Y)

        self.interp = CloughTocherInterpolatorValues(
            list(zip(x, y)), (self.X, self.Y)
        )

    def time_clough_tocher(self, n_samples):
            self.interp(self.z)


class AAA(Benchmark):
    def setup(self):
        self.z = mx.exp(mx.linspace(-0.5, 0.5 + 15j*mx.pi, num=1000))
        self.pts = mx.linspace(-1, 1, num=1000)

    def time_AAA(self):
        r = interpolate.AAA(self.z, mx.tan(mx.pi*self.z/2))
        r(self.pts)
        r.poles()
        r.residues()
        r.roots()
