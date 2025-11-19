import mlx.core as mx
from pytest import raises as assert_raises

import scipy.sparse.linalg._isolve.utils as utils


def test_make_system_bad_shape():
    assert_raises(ValueError,
                  utils.make_system, mx.zeros((5,3)), None, mx.zeros(4), mx.zeros(4))
