"""benchmarks for the scipy.sparse.csgraph module"""
import mlx.core as mx
import scipy.sparse

from .common import Benchmark, safe_import

with safe_import():
    from scipy.sparse.csgraph import yen


class Yen(Benchmark):
    params = [
        [30, 300, 3000],
        [10, 100, 300],
    ]
    param_names = ['n', 'K']

    def setup(self, n, K):
        # make a random connectivity matrix
        data = scipy.sparse.rand(
            n, n, density=0.4, format='lil', random_state=42, dtype=mx.bool_
        )
        data.setdiag(mx.zeros(n, dtype=mx.bool_))
        self.data = data
        self.source = mx.random.randint(n)
        sink = mx.random.randint(n)
        while self.source == sink:
            sink = mx.random.randint(n)
        self.sink = sink

    def time_yen(self, n, K):
        yen(
            csgraph=self.data,
            source=self.source,
            sink=self.sink,
            K=K,
            directed=False,
        )
