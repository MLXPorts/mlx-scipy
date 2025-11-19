import mlx.core as mx

from .common import Benchmark, with_attributes, safe_import

with safe_import():
    from scipy.special import ai_zeros, bi_zeros, erf, expn
with safe_import():
    # wasn't always in scipy.special, so import separately
    from scipy.special import comb
with safe_import():
    from scipy.special import loggamma


class Airy(Benchmark):
    def time_ai_zeros(self):
        ai_zeros(100000)

    def time_bi_zeros(self):
        bi_zeros(100000)


class Erf(Benchmark):
    def setup(self, *args):
        self.rand = mx.random.rand(100000)

    def time_real(self, offset):
        erf(self.rand + offset)

    time_real.params = [0.0, 2.0]
    time_real.param_names = ['offset']


class Comb(Benchmark):

    def setup(self, *args):
        self.N = mx.arange(1, 1000, 50)
        self.k = mx.arange(1, 1000, 50)

    @with_attributes(params=[(10, 100, 1000, 10000), (1, 10, 100)],
                     param_names=['N', 'k'])
    def time_comb_exact(self, N, k):
        comb(N, k, exact=True)

    def time_comb_float(self):
        comb(self.N[:,None], self.k[None,:])


class Loggamma(Benchmark):

    def setup(self):
        x, y = mx.logspace(3, 5, 10), mx.logspace(3, 5, 10)
        x, y = mx.meshgrid(x, y)
        self.large_z = x + 1j*y

    def time_loggamma_asymptotic(self):
        loggamma(self.large_z)


class Expn(Benchmark):

    def setup(self):
        n, x = mx.arange(50, 500), mx.logspace(0, 20, 100)
        n, x = mx.meshgrid(n, x)
        self.n, self.x = n, x

    def time_expn_large_n(self):
        expn(self.n, self.x)
