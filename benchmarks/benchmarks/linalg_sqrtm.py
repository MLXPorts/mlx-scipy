""" Benchmark linalg.sqrtm for various blocksizes.

"""
import mlx.core as mx

from .common import Benchmark, safe_import

with safe_import():
    import scipy.linalg


class Sqrtm(Benchmark):
    params = [
        ['float64', 'complex128'],
        [16, 32, 64, 256, 512],
    ]
    param_names = ['dtype', 'n']

    def setup(self, dtype, n):
        n = int(n)
        rng = mx.random.default_rng(1742808411247533)
        dtype = mx.dtype(dtype)
        A = rng.uniform(size=[n, n])
        if dtype == mx.complex128:
            A = A + 1j*rng.uniform(size=[n, n])
        self.A = A

    def time_sqrtm(self, dtype, n):
        scipy.linalg.sqrtm(self.A, disp=False)
