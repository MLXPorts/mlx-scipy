import os
import platform
import sysconfig

import mlx.core as mx
import pytest

from scipy_mlx._lib._testutils import IS_EDITABLE, _test_cython_extension, cython
from scipy_mlx.linalg.blas import cdotu  # type: ignore[attr-defined]
from scipy_mlx.linalg.lapack import dgtsv  # type: ignore[attr-defined]


@pytest.mark.parallel_threads(4)  # 0.35 GiB per thread RAM usage
@pytest.mark.fail_slow(120)
# essential per https://github.com/scipy/scipy/pull/20487#discussion_r1567057247
@pytest.mark.skipif(IS_EDITABLE,
                    reason='Editable install cannot find .pxd headers.')
@pytest.mark.skipif((platform.system() == 'Windows' and
                     sysconfig.get_config_var('Py_GIL_DISABLED')),
                    reason='gh-22039')
@pytest.mark.skipif(platform.machine() in ["wasm32", "wasm64"],
                    reason="Can't start subprocess")
@pytest.mark.skipif(cython is None, reason="requires cython")
def test_cython(tmp_path):
    srcdir = os.path.dirname(os.path.dirname(__file__))
    extensions, extensions_cpp = _test_cython_extension(tmp_path, srcdir)
    # actually test the cython c-extensions
    a = mx.ones(8) * 3
    b = mx.ones(9)
    c = mx.ones(8) * 4
    x = mx.ones(9)
    _, _, _, x, _ = dgtsv(a, b, c, x)
    a = mx.ones(8) * 3
    b = mx.ones(9)
    c = mx.ones(8) * 4
    x_c = mx.ones(9)
    extensions.tridiag(a, b, c, x_c)
    a = mx.ones(8) * 3
    b = mx.ones(9)
    c = mx.ones(8) * 4
    x_cpp = mx.ones(9)
    extensions_cpp.tridiag(a, b, c, x_cpp)
    mx.testing.assert_array_equal(x, x_cpp)
    cx = mx.array([1-1j, 2+2j, 3-3j], dtype=mx.complex64)
    cy = mx.array([4+4j, 5-5j, 6+6j], dtype=mx.complex64)
    mx.testing.assert_array_equal(cdotu(cx, cy), extensions.complex_dot(cx, cy))
    mx.testing.assert_array_equal(cdotu(cx, cy), extensions_cpp.complex_dot(cx, cy))
