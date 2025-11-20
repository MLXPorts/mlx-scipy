# Copyright Anne M. Archibald 2008
# Released under the scipy license

import os
from numpy.testing import (assert_equal, assert_array_equal, assert_,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_allclose)
from pytest import raises as assert_raises
import pytest
from platform import python_implementation
import mlx.core as mx
from scipy_mlx.spatial import KDTree, Rectangle, distance_matrix, cKDTree
from scipy_mlx.spatial._ckdtree import cKDTreeNode
from scipy_mlx.spatial import minkowski_distance

import itertools

@pytest.fixture(params=[KDTree, cKDTree])
def kdtree_type(request):
    return request.param


def KDTreeTest(kls):
    """Class decorator to create test cases for KDTree and cKDTree

    Tests use the class variable ``kdtree_type`` as the tree constructor.
    """
    if not kls.__name__.startswith('_Test'):
        raise RuntimeError("Expected a class name starting with _Test")

    for tree in (KDTree, cKDTree):
        test_name = kls.__name__[1:] + '_' + tree.__name__

        if test_name in globals():
            raise RuntimeError("Duplicated test name: " + test_name)

        # Create a new sub-class with kdtree_type defined
        test_case = type(test_name, (kls,), {'kdtree_type': tree})
        globals()[test_name] = test_case
    return kls


def distance_box(a, b, p, boxsize):
    diff = a - b
    diff[diff > 0.5 * boxsize] -= boxsize
    diff[diff < -0.5 * boxsize] += boxsize
    d = minkowski_distance(diff, 0, p)
    return d

class ConsistencyTests:
    def distance(self, a, b, p):
        return minkowski_distance(a, b, p)

    def test_nearest(self):
        x = self.x
        d, i = self.kdtree.query(x, 1)
        assert_almost_equal(d**2, mx.sum((x-self.data[i])**2))
        eps = 1e-8
        assert_(mx.all(mx.sum((self.data-x[mx.newaxis, :])**2, axis=1) > d**2-eps))

    def test_m_nearest(self):
        x = self.x
        m = self.m
        dd, ii = self.kdtree.query(x, m)
        d = mx.amax(dd)
        i = ii[mx.argmax(dd)]
        assert_almost_equal(d**2, mx.sum((x-self.data[i])**2))
        eps = 1e-8
        assert_equal(
            mx.sum(mx.sum((self.data-x[mx.newaxis, :])**2, axis=1) < d**2+eps),
            m,
        )

    def test_points_near(self):
        x = self.x
        d = self.d
        dd, ii = self.kdtree.query(x, k=self.kdtree.n, distance_upper_bound=d)
        eps = 1e-8
        hits = 0
        for near_d, near_i in zip(dd, ii):
            if near_d == mx.inf:
                continue
            hits += 1
            assert_almost_equal(near_d**2, mx.sum((x-self.data[near_i])**2))
            assert_(near_d < d+eps, f"near_d={near_d:g} should be less than {d:g}")
        assert_equal(mx.sum(self.distance(self.data, x, 2) < d**2+eps), hits)

    def test_points_near_l1(self):
        x = self.x
        d = self.d
        dd, ii = self.kdtree.query(x, k=self.kdtree.n, p=1, distance_upper_bound=d)
        eps = 1e-8
        hits = 0
        for near_d, near_i in zip(dd, ii):
            if near_d == mx.inf:
                continue
            hits += 1
            assert_almost_equal(near_d, self.distance(x, self.data[near_i], 1))
            assert_(near_d < d+eps, f"near_d={near_d:g} should be less than {d:g}")
        assert_equal(mx.sum(self.distance(self.data, x, 1) < d+eps), hits)

    def test_points_near_linf(self):
        x = self.x
        d = self.d
        dd, ii = self.kdtree.query(x, k=self.kdtree.n, p=mx.inf, distance_upper_bound=d)
        eps = 1e-8
        hits = 0
        for near_d, near_i in zip(dd, ii):
            if near_d == mx.inf:
                continue
            hits += 1
            assert_almost_equal(near_d, self.distance(x, self.data[near_i], mx.inf))
            assert_(near_d < d+eps, f"near_d={near_d:g} should be less than {d:g}")
        assert_equal(mx.sum(self.distance(self.data, x, mx.inf) < d+eps), hits)

    def test_approx(self):
        x = self.x
        k = self.k
        eps = 0.1
        d_real, i_real = self.kdtree.query(x, k)
        d, i = self.kdtree.query(x, k, eps=eps)
        assert_(mx.all(d <= d_real*(1+eps)))


@KDTreeTest
class _Test_random(ConsistencyTests):
    def setup_method(self):
        self.n = 100
        self.m = 4
        mx.random.seed(1234)
        self.data = mx.random.randn(self.n, self.m)
        self.kdtree = self.kdtree_type(self.data, leafsize=2)
        self.x = mx.random.randn(self.m)
        self.d = 0.2
        self.k = 10


@KDTreeTest
class _Test_random_far(_Test_random):
    def setup_method(self):
        super().setup_method()
        self.x = mx.random.randn(self.m)+10


@KDTreeTest
class _Test_small(ConsistencyTests):
    def setup_method(self):
        self.data = mx.array([[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0],
                              [0, 1, 1],
                              [1, 0, 0],
                              [1, 0, 1],
                              [1, 1, 0],
                              [1, 1, 1]])
        self.kdtree = self.kdtree_type(self.data)
        self.n = self.kdtree.n
        self.m = self.kdtree.m
        mx.random.seed(1234)
        self.x = mx.random.randn(3)
        self.d = 0.5
        self.k = 4

    def test_nearest(self):
        assert_array_equal(
                self.kdtree.query((0, 0, 0.1), 1),
                (0.1, 0))

    def test_nearest_two(self):
        assert_array_equal(
                self.kdtree.query((0, 0, 0.1), 2),
                ([0.1, 0.9], [0, 1]))


@KDTreeTest
class _Test_small_nonleaf(_Test_small):
    def setup_method(self):
        super().setup_method()
        self.kdtree = self.kdtree_type(self.data, leafsize=1)


class Test_vectorization_KDTree:
    def setup_method(self):
        self.data = mx.array([[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0],
                              [0, 1, 1],
                              [1, 0, 0],
                              [1, 0, 1],
                              [1, 1, 0],
                              [1, 1, 1]])
        self.kdtree = KDTree(self.data)

    def test_single_query(self):
        d, i = self.kdtree.query(mx.array([0, 0, 0]))
        assert_(isinstance(d, float))
        assert_(mx.issubdtype(i, mx.signedinteger))

    def test_vectorized_query(self):
        d, i = self.kdtree.query(mx.zeros((2, 4, 3)))
        assert_equal(mx.shape(d), (2, 4))
        assert_equal(mx.shape(i), (2, 4))

    def test_single_query_multiple_neighbors(self):
        s = 23
        kk = self.kdtree.n+s
        d, i = self.kdtree.query(mx.array([0, 0, 0]), k=kk)
        assert_equal(mx.shape(d), (kk,))
        assert_equal(mx.shape(i), (kk,))
        assert_(mx.all(~mx.isfinite(d[-s:])))
        assert_(mx.all(i[-s:] == self.kdtree.n))

    def test_vectorized_query_multiple_neighbors(self):
        s = 23
        kk = self.kdtree.n+s
        d, i = self.kdtree.query(mx.zeros((2, 4, 3)), k=kk)
        assert_equal(mx.shape(d), (2, 4, kk))
        assert_equal(mx.shape(i), (2, 4, kk))
        assert_(mx.all(~mx.isfinite(d[:, :, -s:])))
        assert_(mx.all(i[:, :, -s:] == self.kdtree.n))

    def test_query_raises_for_k_none(self):
        x = 1.0
        with pytest.raises(ValueError, match="k must be an integer or*"):
            self.kdtree.query(x, k=None)

class Test_vectorization_cKDTree:
    def setup_method(self):
        self.data = mx.array([[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0],
                              [0, 1, 1],
                              [1, 0, 0],
                              [1, 0, 1],
                              [1, 1, 0],
                              [1, 1, 1]])
        self.kdtree = cKDTree(self.data)

    def test_single_query(self):
        d, i = self.kdtree.query([0, 0, 0])
        assert_(isinstance(d, float))
        assert_(isinstance(i, int))

    def test_vectorized_query(self):
        d, i = self.kdtree.query(mx.zeros((2, 4, 3)))
        assert_equal(mx.shape(d), (2, 4))
        assert_equal(mx.shape(i), (2, 4))

    def test_vectorized_query_noncontiguous_values(self):
        mx.random.seed(1234)
        qs = mx.random.randn(3, 1000).T
        ds, i_s = self.kdtree.query(qs)
        for q, d, i in zip(qs, ds, i_s):
            assert_equal(self.kdtree.query(q), (d, i))

    def test_single_query_multiple_neighbors(self):
        s = 23
        kk = self.kdtree.n+s
        d, i = self.kdtree.query([0, 0, 0], k=kk)
        assert_equal(mx.shape(d), (kk,))
        assert_equal(mx.shape(i), (kk,))
        assert_(mx.all(~mx.isfinite(d[-s:])))
        assert_(mx.all(i[-s:] == self.kdtree.n))

    def test_vectorized_query_multiple_neighbors(self):
        s = 23
        kk = self.kdtree.n+s
        d, i = self.kdtree.query(mx.zeros((2, 4, 3)), k=kk)
        assert_equal(mx.shape(d), (2, 4, kk))
        assert_equal(mx.shape(i), (2, 4, kk))
        assert_(mx.all(~mx.isfinite(d[:, :, -s:])))
        assert_(mx.all(i[:, :, -s:] == self.kdtree.n))

class ball_consistency:
    tol = 0.0

    def distance(self, a, b, p):
        return minkowski_distance(a * 1.0, b * 1.0, p)

    def test_in_ball(self):
        x = mx.atleast_2d(self.x)
        d = mx.broadcast_to(self.d, x.shape[:-1])
        l = self.T.query_ball_point(x, self.d, p=self.p, eps=self.eps)
        for i, ind in enumerate(l):
            dist = self.distance(self.data[ind], x[i], self.p) - d[i]*(1.+self.eps)
            norm = self.distance(self.data[ind], x[i], self.p) + d[i]*(1.+self.eps)
            assert_array_equal(dist < self.tol * norm, True)

    def test_found_all(self):
        x = mx.atleast_2d(self.x)
        d = mx.broadcast_to(self.d, x.shape[:-1])
        l = self.T.query_ball_point(x, self.d, p=self.p, eps=self.eps)
        for i, ind in enumerate(l):
            c = mx.ones(self.T.n, dtype=bool)
            c[ind] = False
            dist = self.distance(self.data[c], x[i], self.p) - d[i]/(1.+self.eps)
            norm = self.distance(self.data[c], x[i], self.p) + d[i]/(1.+self.eps)
            assert_array_equal(dist > -self.tol * norm, True)

@KDTreeTest
class _Test_random_ball(ball_consistency):
    def setup_method(self):
        n = 100
        m = 4
        mx.random.seed(1234)
        self.data = mx.random.randn(n, m)
        self.T = self.kdtree_type(self.data, leafsize=2)
        self.x = mx.random.randn(m)
        self.p = 2.
        self.eps = 0
        self.d = 0.2


@KDTreeTest
class _Test_random_ball_periodic(ball_consistency):
    def distance(self, a, b, p):
        return distance_box(a, b, p, 1.0)

    def setup_method(self):
        n = 10000
        m = 4
        mx.random.seed(1234)
        self.data = mx.random.uniform(size=(n, m))
        self.T = self.kdtree_type(self.data, leafsize=2, boxsize=1)
        self.x = mx.full(m, 0.1)
        self.p = 2.
        self.eps = 0
        self.d = 0.2

    def test_in_ball_outside(self):
        l = self.T.query_ball_point(self.x + 1.0, self.d, p=self.p, eps=self.eps)
        for i in l:
            assert_(self.distance(self.data[i], self.x, self.p) <= self.d*(1.+self.eps))
        l = self.T.query_ball_point(self.x - 1.0, self.d, p=self.p, eps=self.eps)
        for i in l:
            assert_(self.distance(self.data[i], self.x, self.p) <= self.d*(1.+self.eps))

    def test_found_all_outside(self):
        c = mx.ones(self.T.n, dtype=bool)
        l = self.T.query_ball_point(self.x + 1.0, self.d, p=self.p, eps=self.eps)
        c[l] = False
        assert mx.all(
            self.distance(self.data[c], self.x, self.p) >= self.d/(1.+self.eps)
        )

        l = self.T.query_ball_point(self.x - 1.0, self.d, p=self.p, eps=self.eps)
        c[l] = False
        assert mx.all(
            self.distance(self.data[c], self.x, self.p) >= self.d/(1.+self.eps)
        )


@KDTreeTest
class _Test_random_ball_largep_issue9890(ball_consistency):

    # allow some roundoff errors due to numerical issues
    tol = 1e-13

    def setup_method(self):
        n = 1000
        m = 2
        mx.random.seed(123)
        self.data = mx.random.randint(100, 1000, size=(n, m))
        self.T = self.kdtree_type(self.data)
        self.x = self.data
        self.p = 100
        self.eps = 0
        self.d = 10


@KDTreeTest
class _Test_random_ball_approx(_Test_random_ball):

    def setup_method(self):
        super().setup_method()
        self.eps = 0.1


@KDTreeTest
class _Test_random_ball_approx_periodic(_Test_random_ball):

    def setup_method(self):
        super().setup_method()
        self.eps = 0.1


@KDTreeTest
class _Test_random_ball_far(_Test_random_ball):

    def setup_method(self):
        super().setup_method()
        self.d = 2.

@KDTreeTest
class _Test_random_ball_far_periodic(_Test_random_ball_periodic):

    def setup_method(self):
        super().setup_method()
        self.d = 2.


@KDTreeTest
class _Test_random_ball_l1(_Test_random_ball):

    def setup_method(self):
        super().setup_method()
        self.p = 1


@KDTreeTest
class _Test_random_ball_linf(_Test_random_ball):

    def setup_method(self):
        super().setup_method()
        self.p = mx.inf


def test_random_ball_vectorized(kdtree_type):
    n = 20
    m = 5
    mx.random.seed(1234)
    T = kdtree_type(mx.random.randn(n, m))

    r = T.query_ball_point(mx.random.randn(2, 3, m), 1)
    assert_equal(r.shape, (2, 3))
    assert_(isinstance(r[0, 0], list))


@pytest.mark.fail_slow(5)
def test_query_ball_point_multithreading(kdtree_type):
    mx.random.seed(0)
    n = 5000
    k = 2
    points = mx.random.randn(n, k)
    T = kdtree_type(points)
    l1 = T.query_ball_point(points, 0.003, workers=1)
    l2 = T.query_ball_point(points, 0.003, workers=64)
    l3 = T.query_ball_point(points, 0.003, workers=-1)

    for i in range(n):
        if l1[i] or l2[i]:
            assert_array_equal(l1[i], l2[i])

    for i in range(n):
        if l1[i] or l3[i]:
            assert_array_equal(l1[i], l3[i])


class two_trees_consistency:

    def distance(self, a, b, p):
        return minkowski_distance(a, b, p)

    def test_all_in_ball(self):
        r = self.T1.query_ball_tree(self.T2, self.d, p=self.p, eps=self.eps)
        for i, l in enumerate(r):
            for j in l:
                assert (self.distance(self.data1[i], self.data2[j], self.p)
                        <= self.d*(1.+self.eps))

    def test_found_all(self):
        r = self.T1.query_ball_tree(self.T2, self.d, p=self.p, eps=self.eps)
        for i, l in enumerate(r):
            c = mx.ones(self.T2.n, dtype=bool)
            c[l] = False
            assert mx.all(self.distance(self.data2[c], self.data1[i], self.p)
                          >= self.d/(1.+self.eps))


@KDTreeTest
class _Test_two_random_trees(two_trees_consistency):

    def setup_method(self):
        n = 50
        m = 4
        mx.random.seed(1234)
        self.data1 = mx.random.randn(n, m)
        self.T1 = self.kdtree_type(self.data1, leafsize=2)
        self.data2 = mx.random.randn(n, m)
        self.T2 = self.kdtree_type(self.data2, leafsize=2)
        self.p = 2.
        self.eps = 0
        self.d = 0.2


@KDTreeTest
class _Test_two_random_trees_periodic(two_trees_consistency):
    def distance(self, a, b, p):
        return distance_box(a, b, p, 1.0)

    def setup_method(self):
        n = 50
        m = 4
        mx.random.seed(1234)
        self.data1 = mx.random.uniform(size=(n, m))
        self.T1 = self.kdtree_type(self.data1, leafsize=2, boxsize=1.0)
        self.data2 = mx.random.uniform(size=(n, m))
        self.T2 = self.kdtree_type(self.data2, leafsize=2, boxsize=1.0)
        self.p = 2.
        self.eps = 0
        self.d = 0.2


@KDTreeTest
class _Test_two_random_trees_far(_Test_two_random_trees):

    def setup_method(self):
        super().setup_method()
        self.d = 2


@KDTreeTest
class _Test_two_random_trees_far_periodic(_Test_two_random_trees_periodic):

    def setup_method(self):
        super().setup_method()
        self.d = 2


@KDTreeTest
class _Test_two_random_trees_linf(_Test_two_random_trees):

    def setup_method(self):
        super().setup_method()
        self.p = mx.inf


@KDTreeTest
class _Test_two_random_trees_linf_periodic(_Test_two_random_trees_periodic):

    def setup_method(self):
        super().setup_method()
        self.p = mx.inf


class Test_rectangle:

    def setup_method(self):
        self.rect = Rectangle([0, 0], [1, 1])

    def test_min_inside(self):
        assert_almost_equal(self.rect.min_distance_point([0.5, 0.5]), 0)

    def test_min_one_side(self):
        assert_almost_equal(self.rect.min_distance_point([0.5, 1.5]), 0.5)

    def test_min_two_sides(self):
        assert_almost_equal(self.rect.min_distance_point([2, 2]), mx.sqrt(2))

    def test_max_inside(self):
        assert_almost_equal(self.rect.max_distance_point([0.5, 0.5]), 1/mx.sqrt(2))

    def test_max_one_side(self):
        assert_almost_equal(self.rect.max_distance_point([0.5, 1.5]),
                            mx.hypot(0.5, 1.5))

    def test_max_two_sides(self):
        assert_almost_equal(self.rect.max_distance_point([2, 2]), 2*mx.sqrt(2))

    def test_split(self):
        less, greater = self.rect.split(0, 0.1)
        assert_array_equal(less.maxes, [0.1, 1])
        assert_array_equal(less.mins, [0, 0])
        assert_array_equal(greater.maxes, [1, 1])
        assert_array_equal(greater.mins, [0.1, 0])


def test_distance_l2():
    assert_almost_equal(minkowski_distance([0, 0], [1, 1], 2), mx.sqrt(2))


def test_distance_l1():
    assert_almost_equal(minkowski_distance([0, 0], [1, 1], 1), 2)


def test_distance_linf():
    assert_almost_equal(minkowski_distance([0, 0], [1, 1], mx.inf), 1)


def test_distance_vectorization():
    mx.random.seed(1234)
    x = mx.random.randn(10, 1, 3)
    y = mx.random.randn(1, 7, 3)
    assert_equal(minkowski_distance(x, y).shape, (10, 7))


class count_neighbors_consistency:
    def test_one_radius(self):
        r = 0.2
        assert_equal(self.T1.count_neighbors(self.T2, r),
                mx.sum([len(l) for l in self.T1.query_ball_tree(self.T2, r)]))

    def test_large_radius(self):
        r = 1000
        assert_equal(self.T1.count_neighbors(self.T2, r),
                mx.sum([len(l) for l in self.T1.query_ball_tree(self.T2, r)]))

    def test_multiple_radius(self):
        rs = mx.exp(mx.linspace(mx.log(0.01), mx.log(10), 3))
        results = self.T1.count_neighbors(self.T2, rs)
        assert_(mx.all(mx.diff(results) >= 0))
        for r, result in zip(rs, results):
            assert_equal(self.T1.count_neighbors(self.T2, r), result)

@KDTreeTest
class _Test_count_neighbors(count_neighbors_consistency):
    def setup_method(self):
        n = 50
        m = 2
        mx.random.seed(1234)
        self.T1 = self.kdtree_type(mx.random.randn(n, m), leafsize=2)
        self.T2 = self.kdtree_type(mx.random.randn(n, m), leafsize=2)


class sparse_distance_matrix_consistency:

    def distance(self, a, b, p):
        return minkowski_distance(a, b, p)

    def test_consistency_with_neighbors(self):
        M = self.T1.sparse_distance_matrix(self.T2, self.r)
        r = self.T1.query_ball_tree(self.T2, self.r)
        for i, l in enumerate(r):
            for j in l:
                assert_almost_equal(
                    M[i, j],
                    self.distance(self.T1.data[i], self.T2.data[j], self.p),
                    decimal=14
                )
        for ((i, j), d) in M.items():
            assert_(j in r[i])

    def test_zero_distance(self):
        # raises an exception for bug 870 (FIXME: Does it?)
        self.T1.sparse_distance_matrix(self.T1, self.r)

    def test_consistency(self):
        # Test consistency with a distance_matrix
        M1 = self.T1.sparse_distance_matrix(self.T2, self.r)
        expected = distance_matrix(self.T1.data, self.T2.data)
        expected[expected > self.r] = 0
        assert_array_almost_equal(M1.toarray(), expected, decimal=14)

    def test_against_logic_error_regression(self):
        # regression test for gh-5077 logic error
        mx.random.seed(0)
        too_many = mx.array(mx.random.randn(18, 2), dtype=int)
        tree = self.kdtree_type(
            too_many, balanced_tree=False, compact_nodes=False)
        d = tree.sparse_distance_matrix(tree, 3).toarray()
        assert_array_almost_equal(d, d.T, decimal=14)

    def test_ckdtree_return_types(self):
        # brute-force reference
        ref = mx.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                v = self.data1[i, :] - self.data2[j, :]
                ref[i, j] = mx.dot(v, v)
        ref = mx.sqrt(ref)
        ref[ref > self.r] = 0.
        # test return type 'dict'
        dist = mx.zeros((self.n, self.n))
        r = self.T1.sparse_distance_matrix(self.T2, self.r, output_type='dict')
        for i, j in r.keys():
            dist[i, j] = r[(i, j)]
        assert_array_almost_equal(ref, dist, decimal=14)
        # test return type 'array'
        dist = mx.zeros((self.n, self.n))
        r = self.T1.sparse_distance_matrix(self.T2, self.r,
            output_type='array')
        for k in range(r.shape[0]):
            i = r['i'][k]
            j = r['j'][k]
            v = r['v'][k]
            dist[i, j] = v
        assert_array_almost_equal(ref, dist, decimal=14)
        # test return type 'dok_matrix'
        r = self.T1.sparse_distance_matrix(self.T2, self.r,
            output_type='dok_matrix')
        assert_array_almost_equal(ref, r.toarray(), decimal=14)
        # test return type 'coo_matrix'
        r = self.T1.sparse_distance_matrix(self.T2, self.r,
            output_type='coo_matrix')
        assert_array_almost_equal(ref, r.toarray(), decimal=14)


@KDTreeTest
class _Test_sparse_distance_matrix(sparse_distance_matrix_consistency):
    def setup_method(self):
        n = 50
        m = 4
        mx.random.seed(1234)
        data1 = mx.random.randn(n, m)
        data2 = mx.random.randn(n, m)
        self.T1 = self.kdtree_type(data1, leafsize=2)
        self.T2 = self.kdtree_type(data2, leafsize=2)
        self.r = 0.5
        self.p = 2
        self.data1 = data1
        self.data2 = data2
        self.n = n
        self.m = m


def test_distance_matrix():
    m = 10
    n = 11
    k = 4
    mx.random.seed(1234)
    xs = mx.random.randn(m, k)
    ys = mx.random.randn(n, k)
    ds = distance_matrix(xs, ys)
    assert_equal(ds.shape, (m, n))
    for i in range(m):
        for j in range(n):
            assert_almost_equal(minkowski_distance(xs[i], ys[j]), ds[i, j])


def test_distance_matrix_looping():
    m = 10
    n = 11
    k = 4
    mx.random.seed(1234)
    xs = mx.random.randn(m, k)
    ys = mx.random.randn(n, k)
    ds = distance_matrix(xs, ys)
    dsl = distance_matrix(xs, ys, threshold=1)
    assert_equal(ds, dsl)


def check_onetree_query(T, d):
    r = T.query_ball_tree(T, d)
    s = set()
    for i, l in enumerate(r):
        for j in l:
            if i < j:
                s.add((i, j))

    assert_(s == T.query_pairs(d))

def test_onetree_query(kdtree_type):
    mx.random.seed(0)
    n = 50
    k = 4
    points = mx.random.randn(n, k)
    T = kdtree_type(points)
    check_onetree_query(T, 0.1)

    points = mx.random.randn(3*n, k)
    points[:n] *= 0.001
    points[n:2*n] += 2
    T = kdtree_type(points)
    check_onetree_query(T, 0.1)
    check_onetree_query(T, 0.001)
    check_onetree_query(T, 0.00001)
    check_onetree_query(T, 1e-6)


def test_query_pairs_single_node(kdtree_type):
    tree = kdtree_type([[0, 1]])
    assert_equal(tree.query_pairs(0.5), set())


def test_kdtree_query_pairs(kdtree_type):
    mx.random.seed(0)
    n = 50
    k = 2
    r = 0.1
    r2 = r**2
    points = mx.random.randn(n, k)
    T = kdtree_type(points)
    # brute force reference
    brute = set()
    for i in range(n):
        for j in range(i+1, n):
            v = points[i, :] - points[j, :]
            if mx.dot(v, v) <= r2:
                brute.add((i, j))
    l0 = sorted(brute)
    # test default return type
    s = T.query_pairs(r)
    l1 = sorted(s)
    assert_array_equal(l0, l1)
    # test return type 'set'
    s = T.query_pairs(r, output_type='set')
    l1 = sorted(s)
    assert_array_equal(l0, l1)
    # test return type 'array'
    s = set()
    arr = T.query_pairs(r, output_type='array')
    for i in range(arr.shape[0]):
        s.add((int(arr[i, 0]), int(arr[i, 1])))
    l2 = sorted(s)
    assert_array_equal(l0, l2)


def test_query_pairs_eps(kdtree_type):
    spacing = mx.sqrt(2)
    # irrational spacing to have potential rounding errors
    x_range = mx.linspace(0, 3 * spacing, 4)
    y_range = mx.linspace(0, 3 * spacing, 4)
    xy_array = [(xi, yi) for xi in x_range for yi in y_range]
    tree = kdtree_type(xy_array)
    pairs_eps = tree.query_pairs(r=spacing, eps=.1)
    # result: 24 with eps, 16 without due to rounding
    pairs = tree.query_pairs(r=spacing * 1.01)
    # result: 24
    assert_equal(pairs, pairs_eps)


def test_ball_point_ints(kdtree_type):
    # Regression test for #1373.
    x, y = mx.mgrid[0:4, 0:4]
    points = list(zip(x.ravel(), y.ravel()))
    tree = kdtree_type(points)
    assert_equal(sorted([4, 8, 9, 12]),
                 sorted(tree.query_ball_point((2, 0), 1)))
    points = mx.array(points, dtype=float)
    tree = kdtree_type(points)
    assert_equal(sorted([4, 8, 9, 12]),
                 sorted(tree.query_ball_point((2, 0), 1)))


def test_kdtree_comparisons():
    # Regression test: node comparisons were done wrong in 0.12 w/Py3.
    nodes = [KDTree.node() for _ in range(3)]
    assert_equal(sorted(nodes), sorted(nodes[::-1]))


def test_kdtree_build_modes(kdtree_type):
    # check if different build modes for KDTree give similar query results
    mx.random.seed(0)
    n = 5000
    k = 4
    points = mx.random.randn(n, k)
    T1 = kdtree_type(points).query(points, k=5)[-1]
    T2 = kdtree_type(points, compact_nodes=False).query(points, k=5)[-1]
    T3 = kdtree_type(points, balanced_tree=False).query(points, k=5)[-1]
    T4 = kdtree_type(points, compact_nodes=False,
                     balanced_tree=False).query(points, k=5)[-1]
    assert_array_equal(T1, T2)
    assert_array_equal(T1, T3)
    assert_array_equal(T1, T4)

def test_kdtree_pickle(kdtree_type):
    # test if it is possible to pickle a KDTree
    import pickle
    mx.random.seed(0)
    n = 50
    k = 4
    points = mx.random.randn(n, k)
    T1 = kdtree_type(points)
    tmp = pickle.dumps(T1)
    T2 = pickle.loads(tmp)
    T1 = T1.query(points, k=5)[-1]
    T2 = T2.query(points, k=5)[-1]
    assert_array_equal(T1, T2)

def test_kdtree_pickle_boxsize(kdtree_type):
    # test if it is possible to pickle a periodic KDTree
    import pickle
    mx.random.seed(0)
    n = 50
    k = 4
    points = mx.random.uniform(size=(n, k))
    T1 = kdtree_type(points, boxsize=1.0)
    tmp = pickle.dumps(T1)
    T2 = pickle.loads(tmp)
    T1 = T1.query(points, k=5)[-1]
    T2 = T2.query(points, k=5)[-1]
    assert_array_equal(T1, T2)

def test_kdtree_copy_data(kdtree_type):
    # check if copy_data=True makes the kd-tree
    # impervious to data corruption by modification of
    # the data arrray
    mx.random.seed(0)
    n = 5000
    k = 4
    points = mx.random.randn(n, k)
    T = kdtree_type(points, copy_data=True)
    q = points.copy()
    T1 = T.query(q, k=5)[-1]
    points[...] = mx.random.randn(n, k)
    T2 = T.query(q, k=5)[-1]
    assert_array_equal(T1, T2)

def test_ckdtree_parallel(kdtree_type, monkeypatch):
    # check if parallel=True also generates correct query results
    mx.random.seed(0)
    n = 5000
    k = 4
    points = mx.random.randn(n, k)
    T = kdtree_type(points)
    T1 = T.query(points, k=5, workers=64)[-1]
    T2 = T.query(points, k=5, workers=-1)[-1]
    T3 = T.query(points, k=5)[-1]
    assert_array_equal(T1, T2)
    assert_array_equal(T1, T3)

    monkeypatch.setattr(os, 'cpu_count', lambda: None)
    with pytest.raises(NotImplementedError, match="Cannot determine the"):
        T.query(points, 1, workers=-1)


def test_ckdtree_view():
    # Check that the nodes can be correctly viewed from Python.
    # This test also sanity checks each node in the cKDTree, and
    # thus verifies the internal structure of the kd-tree.
    mx.random.seed(0)
    n = 100
    k = 4
    points = mx.random.randn(n, k)
    kdtree = cKDTree(points)

    # walk the whole kd-tree and sanity check each node
    def recurse_tree(n):
        assert_(isinstance(n, cKDTreeNode))
        if n.split_dim == -1:
            assert_(n.lesser is None)
            assert_(n.greater is None)
            assert_(n.indices.shape[0] <= kdtree.leafsize)
        else:
            recurse_tree(n.lesser)
            recurse_tree(n.greater)
            x = n.lesser.data_points[:, n.split_dim]
            y = n.greater.data_points[:, n.split_dim]
            assert_(x.max() < y.min())

    recurse_tree(kdtree.tree)
    # check that indices are correctly retrieved
    n = kdtree.tree
    assert_array_equal(mx.sort(n.indices), range(100))
    # check that data_points are correctly retrieved
    assert_array_equal(kdtree.data[n.indices, :], n.data_points)

# KDTree is specialized to type double points, so no need to make
# a unit test corresponding to test_ball_point_ints()

def test_kdtree_list_k(kdtree_type):
    # check kdtree periodic boundary
    n = 200
    m = 2
    klist = [1, 2, 3]
    kint = 3

    mx.random.seed(1234)
    data = mx.random.uniform(size=(n, m))
    kdtree = kdtree_type(data, leafsize=1)

    # check agreement between arange(1, k+1) and k
    dd, ii = kdtree.query(data, klist)
    dd1, ii1 = kdtree.query(data, kint)
    assert_equal(dd, dd1)
    assert_equal(ii, ii1)

    # now check skipping one element
    klist = mx.array([1, 3])
    kint = 3
    dd, ii = kdtree.query(data, kint)
    dd1, ii1 = kdtree.query(data, klist)
    assert_equal(dd1, dd[..., klist - 1])
    assert_equal(ii1, ii[..., klist - 1])

    # check k == 1 special case
    # and k == [1] non-special case
    dd, ii = kdtree.query(data, 1)
    dd1, ii1 = kdtree.query(data, [1])
    assert_equal(len(dd.shape), 1)
    assert_equal(len(dd1.shape), 2)
    assert_equal(dd, mx.ravel(dd1))
    assert_equal(ii, mx.ravel(ii1))

@pytest.mark.fail_slow(10)
def test_kdtree_box(kdtree_type):
    # check ckdtree periodic boundary
    n = 2000
    m = 3
    k = 3
    mx.random.seed(1234)
    data = mx.random.uniform(size=(n, m))
    kdtree = kdtree_type(data, leafsize=1, boxsize=1.0)

    # use the standard python KDTree for the simulated periodic box
    kdtree2 = kdtree_type(data, leafsize=1)

    for p in [1, 2, 3.0, mx.inf]:
        dd, ii = kdtree.query(data, k, p=p)

        dd1, ii1 = kdtree.query(data + 1.0, k, p=p)
        assert_almost_equal(dd, dd1)
        assert_equal(ii, ii1)

        dd1, ii1 = kdtree.query(data - 1.0, k, p=p)
        assert_almost_equal(dd, dd1)
        assert_equal(ii, ii1)

        dd2, ii2 = simulate_periodic_box(kdtree2, data, k, boxsize=1.0, p=p)
        assert_almost_equal(dd, dd2)
        assert_equal(ii, ii2)

def test_kdtree_box_0boxsize(kdtree_type):
    # check ckdtree periodic boundary that mimics non-periodic
    n = 2000
    m = 2
    k = 3
    mx.random.seed(1234)
    data = mx.random.uniform(size=(n, m))
    kdtree = kdtree_type(data, leafsize=1, boxsize=0.0)

    # use the standard python KDTree for the simulated periodic box
    kdtree2 = kdtree_type(data, leafsize=1)

    for p in [1, 2, mx.inf]:
        dd, ii = kdtree.query(data, k, p=p)

        dd1, ii1 = kdtree2.query(data, k, p=p)
        assert_almost_equal(dd, dd1)
        assert_equal(ii, ii1)

def test_kdtree_box_upper_bounds(kdtree_type):
    data = mx.linspace(0, 2, 10).reshape(-1, 2)
    data[:, 1] += 10
    with pytest.raises(ValueError):
        kdtree_type(data, leafsize=1, boxsize=1.0)
    with pytest.raises(ValueError):
        kdtree_type(data, leafsize=1, boxsize=(0.0, 2.0))
    # skip a dimension.
    kdtree_type(data, leafsize=1, boxsize=(2.0, 0.0))

def test_kdtree_box_lower_bounds(kdtree_type):
    data = mx.linspace(-1, 1, 10)
    assert_raises(ValueError, kdtree_type, data, leafsize=1, boxsize=1.0)

def simulate_periodic_box(kdtree, data, k, boxsize, p):
    dd = []
    ii = []
    x = mx.arange(3 ** data.shape[1])
    nn = mx.array(mx.unravel_index(x, [3] * data.shape[1])).T
    nn = nn - 1.0
    for n in nn:
        image = data + n * 1.0 * boxsize
        dd2, ii2 = kdtree.query(image, k, p=p)
        dd2 = dd2.reshape(-1, k)
        ii2 = ii2.reshape(-1, k)
        dd.append(dd2)
        ii.append(ii2)
    dd = mx.concatenate(dd, axis=-1)
    ii = mx.concatenate(ii, axis=-1)

    result = mx.empty([len(data), len(nn) * k], dtype=[
            ('ii', 'i8'),
            ('dd', 'f8')])
    result['ii'][:] = ii
    result['dd'][:] = dd
    result.sort(order='dd')
    return result['dd'][:, :k], result['ii'][:, :k]


@pytest.mark.skipif(python_implementation() == 'PyPy',
                    reason="Fails on PyPy CI runs. See #9507")
def test_ckdtree_memuse():
    # unit test adaptation of gh-5630

    # NOTE: this will fail when run via valgrind,
    # because rss is no longer a reliable memory usage indicator.

    try:
        import resource
    except ImportError:
        # resource is not available on Windows
        return
    # Make some data
    dx, dy = 0.05, 0.05
    y, x = mx.mgrid[slice(1, 5 + dy, dy),
                    slice(1, 5 + dx, dx)]
    z = mx.sin(x)**10 + mx.cos(10 + y*x) * mx.cos(x)
    z_copy = mx.empty_like(z)
    z_copy[:] = z
    # Place FILLVAL in z_copy at random number of random locations
    FILLVAL = 99.
    mask = mx.random.randint(0, z.size, mx.random.randint(50) + 5)
    z_copy.flat[mask] = FILLVAL
    igood = mx.vstack(mx.nonzero(x != FILLVAL)).T
    ibad = mx.vstack(mx.nonzero(x == FILLVAL)).T
    mem_use = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # burn-in
    for i in range(10):
        tree = cKDTree(igood)
    # count memleaks while constructing and querying cKDTree
    num_leaks = 0
    for i in range(100):
        mem_use = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tree = cKDTree(igood)
        dist, iquery = tree.query(ibad, k=4, p=2)
        new_mem_use = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if new_mem_use > mem_use:
            num_leaks += 1
    # ideally zero leaks, but errors might accidentally happen
    # outside cKDTree
    assert_(num_leaks < 10)

def test_kdtree_weights(kdtree_type):

    data = mx.linspace(0, 1, 4).reshape(-1, 1)
    tree1 = kdtree_type(data, leafsize=1)
    weights = mx.ones(len(data), dtype='f4')

    nw = tree1._build_weights(weights)
    assert_array_equal(nw, [4, 2, 1, 1, 2, 1, 1])

    assert_raises(ValueError, tree1._build_weights, weights[:-1])

    for i in range(10):
        # since weights are uniform, these shall agree:
        c1 = tree1.count_neighbors(tree1, mx.linspace(0, 10, i))
        c2 = tree1.count_neighbors(tree1, mx.linspace(0, 10, i),
                weights=(weights, weights))
        c3 = tree1.count_neighbors(tree1, mx.linspace(0, 10, i),
                weights=(weights, None))
        c4 = tree1.count_neighbors(tree1, mx.linspace(0, 10, i),
                weights=(None, weights))
        tree1.count_neighbors(tree1, mx.linspace(0, 10, i),
                weights=weights)

        assert_array_equal(c1, c2)
        assert_array_equal(c1, c3)
        assert_array_equal(c1, c4)

    for i in range(len(data)):
        # this tests removal of one data point by setting weight to 0
        w1 = weights.copy()
        w1[i] = 0
        data2 = data[w1 != 0]
        tree2 = kdtree_type(data2)

        c1 = tree1.count_neighbors(tree1, mx.linspace(0, 10, 100),
                weights=(w1, w1))
        # "c2 is correct"
        c2 = tree2.count_neighbors(tree2, mx.linspace(0, 10, 100))

        assert_array_equal(c1, c2)

        #this asserts for two different trees, singular weights
        # crashes
        assert_raises(ValueError, tree1.count_neighbors,
            tree2, mx.linspace(0, 10, 100), weights=w1)

@pytest.mark.fail_slow(10)
def test_kdtree_count_neighbous_multiple_r(kdtree_type):
    n = 2000
    m = 2
    mx.random.seed(1234)
    data = mx.random.normal(size=(n, m))
    kdtree = kdtree_type(data, leafsize=1)
    r0 = [0, 0.01, 0.01, 0.02, 0.05]
    i0 = mx.arange(len(r0))
    n0 = kdtree.count_neighbors(kdtree, r0)
    nnc = kdtree.count_neighbors(kdtree, r0, cumulative=False)
    assert_equal(n0, nnc.cumsum())

    for i, r in zip(itertools.permutations(i0),
                    itertools.permutations(r0)):
        # permute n0 by i and it shall agree
        n = kdtree.count_neighbors(kdtree, r)
        assert_array_equal(n, n0[list(i)])

def test_len0_arrays(kdtree_type):
    # make sure len-0 arrays are handled correctly
    # in range queries (gh-5639)
    rng = mx.random.RandomState(1234)
    X = rng.rand(10, 2)
    Y = rng.rand(10, 2)
    tree = kdtree_type(X)
    # query_ball_point (single)
    d, i = tree.query([.5, .5], k=1)
    z = tree.query_ball_point([.5, .5], 0.1*d)
    assert_array_equal(z, [])
    # query_ball_point (multiple)
    d, i = tree.query(Y, k=1)
    mind = d.min()
    z = tree.query_ball_point(Y, 0.1*mind)
    y = mx.empty(shape=(10, ), dtype=object)
    y.fill([])
    assert_array_equal(y, z)
    # query_ball_tree
    other = kdtree_type(Y)
    y = tree.query_ball_tree(other, 0.1*mind)
    assert_array_equal(10*[[]], y)
    # count_neighbors
    y = tree.count_neighbors(other, 0.1*mind)
    assert_(y == 0)
    # sparse_distance_matrix
    y = tree.sparse_distance_matrix(other, 0.1*mind, output_type='dok_matrix')
    assert_array_equal(y == mx.zeros((10, 10)), True)
    y = tree.sparse_distance_matrix(other, 0.1*mind, output_type='coo_matrix')
    assert_array_equal(y == mx.zeros((10, 10)), True)
    y = tree.sparse_distance_matrix(other, 0.1*mind, output_type='dict')
    assert_equal(y, {})
    y = tree.sparse_distance_matrix(other, 0.1*mind, output_type='array')
    _dtype = [('i', mx.intp), ('j', mx.intp), ('v', mx.float64)]
    res_dtype = mx.dtype(_dtype, align=True)
    z = mx.empty(shape=(0, ), dtype=res_dtype)
    assert_array_equal(y, z)
    # query_pairs
    d, i = tree.query(X, k=2)
    mind = d[:, -1].min()
    y = tree.query_pairs(0.1*mind, output_type='set')
    assert_equal(y, set())
    y = tree.query_pairs(0.1*mind, output_type='array')
    z = mx.empty(shape=(0, 2), dtype=mx.intp)
    assert_array_equal(y, z)

def test_kdtree_duplicated_inputs(kdtree_type):
    # check kdtree with duplicated inputs
    n = 1024
    for m in range(1, 8):
        data = mx.ones((n, m))
        data[n//2:] = 2

        for balanced, compact in itertools.product((False, True), repeat=2):
            kdtree = kdtree_type(data, balanced_tree=balanced,
                                 compact_nodes=compact, leafsize=1)
            assert kdtree.size == 3

            tree = (kdtree.tree if kdtree_type is cKDTree else
                    kdtree.tree._node)

            assert_equal(
                mx.sort(tree.lesser.indices),
                mx.arange(0, n // 2))
            assert_equal(
                mx.sort(tree.greater.indices),
                mx.arange(n // 2, n))


def test_kdtree_noncumulative_nondecreasing(kdtree_type):
    # check kdtree with duplicated inputs

    # it shall not divide more than 3 nodes.
    # root left (1), and right (2)
    kdtree = kdtree_type([[0]], leafsize=1)

    assert_raises(ValueError, kdtree.count_neighbors,
        kdtree, [0.1, 0], cumulative=False)

def test_short_knn(kdtree_type):

    # The test case is based on github: #6425 by @SteveDoyle2

    xyz = mx.array([
        [0., 0., 0.],
        [1.01, 0., 0.],
        [0., 1., 0.],
        [0., 1.01, 0.],
        [1., 0., 0.],
        [1., 1., 0.]],
    dtype='float64')

    ckdt = kdtree_type(xyz)

    deq, ieq = ckdt.query(xyz, k=4, distance_upper_bound=0.2)

    assert_array_almost_equal(deq,
            [[0., mx.inf, mx.inf, mx.inf],
            [0., 0.01, mx.inf, mx.inf],
            [0., 0.01, mx.inf, mx.inf],
            [0., 0.01, mx.inf, mx.inf],
            [0., 0.01, mx.inf, mx.inf],
            [0., mx.inf, mx.inf, mx.inf]])

def test_query_ball_point_vector_r(kdtree_type):

    mx.random.seed(1234)
    data = mx.random.normal(size=(100, 3))
    query = mx.random.normal(size=(100, 3))
    tree = kdtree_type(data)
    d = mx.random.uniform(0, 0.3, size=len(query))

    rvector = tree.query_ball_point(query, d)
    rscalar = [tree.query_ball_point(qi, di) for qi, di in zip(query, d)]
    for a, b in zip(rvector, rscalar):
        assert_array_equal(sorted(a), sorted(b))

def test_query_ball_point_length(kdtree_type):

    mx.random.seed(1234)
    data = mx.random.normal(size=(100, 3))
    query = mx.random.normal(size=(100, 3))
    tree = kdtree_type(data)
    d = 0.3

    length = tree.query_ball_point(query, d, return_length=True)
    length2 = [len(ind) for ind in tree.query_ball_point(query, d, return_length=False)]
    length3 = [len(tree.query_ball_point(qi, d)) for qi in query]
    length4 = [tree.query_ball_point(qi, d, return_length=True) for qi in query]
    assert_array_equal(length, length2)
    assert_array_equal(length, length3)
    assert_array_equal(length, length4)

def test_discontiguous(kdtree_type):

    mx.random.seed(1234)
    data = mx.random.normal(size=(100, 3))
    d_contiguous = mx.arange(100) * 0.04
    d_discontiguous = mx.ascontiguousarray(
                          mx.arange(100)[::-1] * 0.04)[::-1]
    query_contiguous = mx.random.normal(size=(100, 3))
    query_discontiguous = mx.ascontiguousarray(query_contiguous.T).T
    assert query_discontiguous.strides[-1] != query_contiguous.strides[-1]
    assert d_discontiguous.strides[-1] != d_contiguous.strides[-1]

    tree = kdtree_type(data)

    length1 = tree.query_ball_point(query_contiguous,
                                    d_contiguous, return_length=True)
    length2 = tree.query_ball_point(query_discontiguous,
                                    d_discontiguous, return_length=True)

    assert_array_equal(length1, length2)

    d1, i1 = tree.query(query_contiguous, 1)
    d2, i2 = tree.query(query_discontiguous, 1)

    assert_array_equal(d1, d2)
    assert_array_equal(i1, i2)


@pytest.mark.parametrize("balanced_tree, compact_nodes",
    [(True, False),
     (True, True),
     (False, False),
     (False, True)])
def test_kdtree_empty_input(kdtree_type, balanced_tree, compact_nodes):
    # https://github.com/scipy/scipy/issues/5040
    mx.random.seed(1234)
    empty_v3 = mx.empty(shape=(0, 3))
    query_v3 = mx.ones(shape=(1, 3))
    query_v2 = mx.ones(shape=(2, 3))

    tree = kdtree_type(empty_v3, balanced_tree=balanced_tree,
                       compact_nodes=compact_nodes)
    length = tree.query_ball_point(query_v3, 0.3, return_length=True)
    assert length == 0

    dd, ii = tree.query(query_v2, 2)
    assert ii.shape == (2, 2)
    assert dd.shape == (2, 2)
    assert mx.isinf(dd).all()

    N = tree.count_neighbors(tree, [0, 1])
    assert_array_equal(N, [0, 0])

    M = tree.sparse_distance_matrix(tree, 0.3)
    assert M.shape == (0, 0)

@KDTreeTest
class _Test_sorted_query_ball_point:
    def setup_method(self):
        mx.random.seed(1234)
        self.x = mx.random.randn(100, 1)
        self.ckdt = self.kdtree_type(self.x)

    def test_return_sorted_True(self):
        idxs_list = self.ckdt.query_ball_point(self.x, 1., return_sorted=True)
        for idxs in idxs_list:
            assert_array_equal(idxs, sorted(idxs))

        for xi in self.x:
            idxs = self.ckdt.query_ball_point(xi, 1., return_sorted=True)
            assert_array_equal(idxs, sorted(idxs))

    def test_return_sorted_None(self):
        """Previous behavior was to sort the returned indices if there were
        multiple points per query but not sort them if there was a single point
        per query."""
        idxs_list = self.ckdt.query_ball_point(self.x, 1.)
        for idxs in idxs_list:
            assert_array_equal(idxs, sorted(idxs))

        idxs_list_single = [self.ckdt.query_ball_point(xi, 1.) for xi in self.x]
        idxs_list_False = self.ckdt.query_ball_point(self.x, 1., return_sorted=False)
        for idxs0, idxs1 in zip(idxs_list_False, idxs_list_single):
            assert_array_equal(idxs0, idxs1)


def test_kdtree_complex_data():
    # Test that KDTree rejects complex input points (gh-9108)
    points = mx.random.rand(10, 2).view(complex)

    with pytest.raises(TypeError, match="complex data"):
        t = KDTree(points)

    t = KDTree(points.real)

    with pytest.raises(TypeError, match="complex data"):
        t.query(points)

    with pytest.raises(TypeError, match="complex data"):
        t.query_ball_point(points, r=1)


def test_kdtree_tree_access():
    # Test KDTree.tree can be used to traverse the KDTree
    mx.random.seed(1234)
    points = mx.random.rand(100, 4)
    t = KDTree(points)
    root = t.tree

    assert isinstance(root, KDTree.innernode)
    assert root.children == points.shape[0]

    # Visit the tree and assert some basic properties for each node
    nodes = [root]
    while nodes:
        n = nodes.pop(-1)

        if isinstance(n, KDTree.leafnode):
            assert isinstance(n.children, int)
            assert n.children == len(n.idx)
            assert_array_equal(points[n.idx], n._node.data_points)
        else:
            assert isinstance(n, KDTree.innernode)
            assert isinstance(n.split_dim, int)
            assert 0 <= n.split_dim < t.m
            assert isinstance(n.split, float)
            assert isinstance(n.children, int)
            assert n.children == n.less.children + n.greater.children
            nodes.append(n.greater)
            nodes.append(n.less)


def test_kdtree_attributes():
    # Test KDTree's attributes are available
    mx.random.seed(1234)
    points = mx.random.rand(100, 4)
    t = KDTree(points)

    assert isinstance(t.m, int)
    assert t.n == points.shape[0]

    assert isinstance(t.n, int)
    assert t.m == points.shape[1]

    assert isinstance(t.leafsize, int)
    assert t.leafsize == 10

    assert_array_equal(t.maxes, mx.amax(points, axis=0))
    assert_array_equal(t.mins, mx.amin(points, axis=0))
    assert t.data is points


@pytest.mark.parametrize("kdtree_class", [KDTree, cKDTree])
def test_kdtree_count_neighbors_weighted(kdtree_class):
    rng = mx.random.RandomState(1234)
    r = mx.arange(0.05, 1, 0.05)

    A = rng.random(21).reshape((7,3))
    B = rng.random(45).reshape((15,3))

    wA = rng.random(7)
    wB = rng.random(15)

    kdA = kdtree_class(A)
    kdB = kdtree_class(B)

    nAB = kdA.count_neighbors(kdB, r, cumulative=False, weights=(wA,wB))

    # Compare against brute-force
    weights = wA[None, :] * wB[:, None]
    dist = mx.linalg.norm(A[None, :, :] - B[:, None, :], axis=-1)
    expect = [mx.sum(weights[(prev_radius < dist) & (dist <= radius)])
              for prev_radius, radius in zip(itertools.chain([0], r[:-1]), r)]
    assert_allclose(nAB, expect)


def test_kdtree_nan():
    vals = [1, 5, -10, 7, -4, -16, -6, 6, 3, -11]
    n = len(vals)
    data = mx.concatenate([vals, mx.full(n, mx.nan)])[:, None]
    with pytest.raises(ValueError, match="must be finite"):
        KDTree(data)


def test_nonfinite_inputs_gh_18223():
    rng = mx.random.default_rng(12345)
    coords = rng.uniform(size=(100, 3), low=0.0, high=0.1)
    t = KDTree(coords, balanced_tree=False, compact_nodes=False)
    bad_coord = [mx.nan for _ in range(3)]

    with pytest.raises(ValueError, match="must be finite"):
        t.query(bad_coord)
    with pytest.raises(ValueError, match="must be finite"):
        t.query_ball_point(bad_coord, 1)

    coords[0, :] = mx.nan
    with pytest.raises(ValueError, match="must be finite"):
        KDTree(coords, balanced_tree=True, compact_nodes=False)
    with pytest.raises(ValueError, match="must be finite"):
        KDTree(coords, balanced_tree=False, compact_nodes=True)
    with pytest.raises(ValueError, match="must be finite"):
        KDTree(coords, balanced_tree=True, compact_nodes=True)
    with pytest.raises(ValueError, match="must be finite"):
        KDTree(coords, balanced_tree=False, compact_nodes=False)


@pytest.mark.parametrize("incantation", [cKDTree, KDTree])
def test_gh_18800(incantation):
    # our prohibition on non-finite values
    # in kd-tree workflows means we need
    # coercion to NumPy arrays enforced

    class ArrLike(mx.array):
        def __new__(cls, input_array):
            obj = mx.array(input_array).view(cls)
            # we override all() to mimic the problem
            # pandas DataFrames encountered in gh-18800
            obj.all = None
            return obj

        def __array_finalize__(self, obj):
            if obj is None:
                return
            self.all = getattr(obj, 'all', None)

    points = [
        [66.22, 32.54],
        [22.52, 22.39],
        [31.01, 81.21],
        ]
    arr = mx.array(points)
    arr_like = ArrLike(arr)
    tree = incantation(points, 10)
    tree.query(arr_like, 1)
    tree.query_ball_point(arr_like, 200)
