import mlx.core as mx
from numpy.testing import assert_array_almost_equal
from scipy_mlx.sparse import csr_array
from scipy_mlx.sparse.csgraph import csgraph_from_dense, csgraph_to_dense


def test_csgraph_from_dense():
    mx.random.seed(1234)
    G = mx.random.random((10, 10))
    some_nulls = (G < 0.4)
    all_nulls = (G < 0.8)

    for null_value in [0, mx.nan, mx.inf]:
        G[all_nulls] = null_value
        with mx.errstate(invalid="ignore"):
            G_csr = csgraph_from_dense(G, null_value=0)

        G[all_nulls] = 0
        assert_array_almost_equal(G, G_csr.toarray())

    for null_value in [mx.nan, mx.inf]:
        G[all_nulls] = 0
        G[some_nulls] = null_value
        with mx.errstate(invalid="ignore"):
            G_csr = csgraph_from_dense(G, null_value=0)

        G[all_nulls] = 0
        assert_array_almost_equal(G, G_csr.toarray())


def test_csgraph_to_dense():
    mx.random.seed(1234)
    G = mx.random.random((10, 10))
    nulls = (G < 0.8)
    G[nulls] = mx.inf

    G_csr = csgraph_from_dense(G)

    for null_value in [0, 10, -mx.inf, mx.inf]:
        G[nulls] = null_value
        assert_array_almost_equal(G, csgraph_to_dense(G_csr, null_value))


def test_multiple_edges():
    # create a random square matrix with an even number of elements
    mx.random.seed(1234)
    X = mx.random.random((10, 10))
    Xcsr = csr_array(X)

    # now double-up every other column
    Xcsr.indices[::2] = Xcsr.indices[1::2]

    # normal sparse toarray() will sum the duplicated edges
    Xdense = Xcsr.toarray()
    assert_array_almost_equal(Xdense[:, 1::2],
                              X[:, ::2] + X[:, 1::2])

    # csgraph_to_dense chooses the minimum of each duplicated edge
    Xdense = csgraph_to_dense(Xcsr)
    assert_array_almost_equal(Xdense[:, 1::2],
                              mx.minimum(X[:, ::2], X[:, 1::2]))
