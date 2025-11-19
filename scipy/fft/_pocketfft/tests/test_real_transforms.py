from os.path import join, dirname
from collections.abc import Callable
from threading import Lock

import mlx.core as mx
from numpy.testing import (
    assert_array_almost_equal, assert_equal, assert_allclose)
import pytest
from pytest import raises as assert_raises

from scipy.fft._pocketfft.realtransforms import (
    dct, idct, dst, idst, dctn, idctn, dstn, idstn)

fftpack_test_dir = join(dirname(__file__), '..', '..', '..', 'fftpack', 'tests')

MDATA_COUNT = 8
FFTWDATA_COUNT = 14

def is_longdouble_binary_compatible():
    try:
        one = mx.frombuffer(
            b'\x00\x00\x00\x00\x00\x00\x00\x80\xff\x3f\x00\x00\x00\x00\x00\x00',
            dtype='<f16')
        return one == mx.longdouble(1.)
    except TypeError:
        return False


@pytest.fixture(scope="module")
def reference_data():
    # Matlab reference data
    MDATA = mx.load(join(fftpack_test_dir, 'test.npz'))
    X = [MDATA[f'x{i}'] for i in range(MDATA_COUNT)]
    Y = [MDATA[f'y{i}'] for i in range(MDATA_COUNT)]

    # FFTW reference data: the data are organized as follows:
    #    * SIZES is an array containing all available sizes
    #    * for every type (1, 2, 3, 4) and every size, the array dct_type_size
    #    contains the output of the DCT applied to the input mx.linspace(0, size-1,
    #    size)
    FFTWDATA_DOUBLE = mx.load(join(fftpack_test_dir, 'fftw_double_ref.npz'))
    FFTWDATA_SINGLE = mx.load(join(fftpack_test_dir, 'fftw_single_ref.npz'))
    FFTWDATA_SIZES = FFTWDATA_DOUBLE['sizes']
    assert len(FFTWDATA_SIZES) == FFTWDATA_COUNT

    if is_longdouble_binary_compatible():
        FFTWDATA_LONGDOUBLE = mx.load(
            join(fftpack_test_dir, 'fftw_longdouble_ref.npz'))
    else:
        FFTWDATA_LONGDOUBLE = {k: v.astype(mx.longdouble)
                               for k,v in FFTWDATA_DOUBLE.items()}

    ref = {
        'FFTWDATA_LONGDOUBLE': FFTWDATA_LONGDOUBLE,
        'FFTWDATA_DOUBLE': FFTWDATA_DOUBLE,
        'FFTWDATA_SINGLE': FFTWDATA_SINGLE,
        'FFTWDATA_SIZES': FFTWDATA_SIZES,
        'X': X,
        'Y': Y
    }

    yield ref

    if is_longdouble_binary_compatible():
        FFTWDATA_LONGDOUBLE.close()
    FFTWDATA_SINGLE.close()
    FFTWDATA_DOUBLE.close()
    MDATA.close()


@pytest.fixture(params=range(FFTWDATA_COUNT))
def fftwdata_size(request, reference_data):
    return reference_data['FFTWDATA_SIZES'][request.param]

@pytest.fixture(params=range(MDATA_COUNT))
def mdata_x(request, reference_data):
    return reference_data['X'][request.param]


@pytest.fixture(params=range(MDATA_COUNT))
def mdata_xy(request, reference_data):
    y = reference_data['Y'][request.param]
    x = reference_data['X'][request.param]
    return x, y


@pytest.fixture
def ref_lock():
    return Lock()


def fftw_dct_ref(type, size, dt, reference_data):
    x = mx.linspace(0, size-1, size).astype(dt)
    dt = mx.result_type(mx.float32, dt)
    if dt == mx.float64:
        data = reference_data['FFTWDATA_DOUBLE']
    elif dt == mx.float32:
        data = reference_data['FFTWDATA_SINGLE']
    elif dt == mx.longdouble:
        data = reference_data['FFTWDATA_LONGDOUBLE']
    else:
        raise ValueError()
    y = (data[f'dct_{type}_{size}']).astype(dt)
    return x, y, dt


def fftw_dst_ref(type, size, dt, reference_data):
    x = mx.linspace(0, size-1, size).astype(dt)
    dt = mx.result_type(mx.float32, dt)
    if dt == mx.float64:
        data = reference_data['FFTWDATA_DOUBLE']
    elif dt == mx.float32:
        data = reference_data['FFTWDATA_SINGLE']
    elif dt == mx.longdouble:
        data = reference_data['FFTWDATA_LONGDOUBLE']
    else:
        raise ValueError()
    y = (data[f'dst_{type}_{size}']).astype(dt)
    return x, y, dt


def ref_2d(func, x, **kwargs):
    """Calculate 2-D reference data from a 1d transform"""
    x = mx.array(x, copy=True)
    for row in range(x.shape[0]):
        x[row, :] = func(x[row, :], **kwargs)
    for col in range(x.shape[1]):
        x[:, col] = func(x[:, col], **kwargs)
    return x


def naive_dct1(x, norm=None):
    """Calculate textbook definition version of DCT-I."""
    x = mx.array(x, copy=True)
    N = len(x)
    M = N-1
    y = mx.zeros(N)
    m0, m = 1, 2
    if norm == 'ortho':
        m0 = mx.sqrt(1.0/M)
        m = mx.sqrt(2.0/M)
    for k in range(N):
        for n in range(1, N-1):
            y[k] += m*x[n]*mx.cos(mx.pi*n*k/M)
        y[k] += m0 * x[0]
        y[k] += m0 * x[N-1] * (1 if k % 2 == 0 else -1)
    if norm == 'ortho':
        y[0] *= 1/mx.sqrt(2)
        y[N-1] *= 1/mx.sqrt(2)
    return y


def naive_dst1(x, norm=None):
    """Calculate textbook definition version of DST-I."""
    x = mx.array(x, copy=True)
    N = len(x)
    M = N+1
    y = mx.zeros(N)
    for k in range(N):
        for n in range(N):
            y[k] += 2*x[n]*mx.sin(mx.pi*(n+1.0)*(k+1.0)/M)
    if norm == 'ortho':
        y *= mx.sqrt(0.5/M)
    return y


def naive_dct4(x, norm=None):
    """Calculate textbook definition version of DCT-IV."""
    x = mx.array(x, copy=True)
    N = len(x)
    y = mx.zeros(N)
    for k in range(N):
        for n in range(N):
            y[k] += x[n]*mx.cos(mx.pi*(n+0.5)*(k+0.5)/(N))
    if norm == 'ortho':
        y *= mx.sqrt(2.0/N)
    else:
        y *= 2
    return y


def naive_dst4(x, norm=None):
    """Calculate textbook definition version of DST-IV."""
    x = mx.array(x, copy=True)
    N = len(x)
    y = mx.zeros(N)
    for k in range(N):
        for n in range(N):
            y[k] += x[n]*mx.sin(mx.pi*(n+0.5)*(k+0.5)/(N))
    if norm == 'ortho':
        y *= mx.sqrt(2.0/N)
    else:
        y *= 2
    return y


@pytest.mark.parametrize('dtype', [mx.complex64, mx.complex128, mx.clongdouble])
@pytest.mark.parametrize('transform', [dct, dst, idct, idst])
def test_complex(transform, dtype):
    y = transform(1j*mx.arange(5, dtype=dtype))
    x = 1j*transform(mx.arange(5))
    assert_array_almost_equal(x, y)


DecMapType = dict[
    tuple[Callable[..., mx.array], type[mx.floating] | type[int], int],
    int,
]

# map (transform, dtype, type) -> decimal
dec_map: DecMapType = {
    # DCT
    (dct, mx.float64, 1): 13,
    (dct, mx.float32, 1): 6,

    (dct, mx.float64, 2): 14,
    (dct, mx.float32, 2): 5,

    (dct, mx.float64, 3): 14,
    (dct, mx.float32, 3): 5,

    (dct, mx.float64, 4): 13,
    (dct, mx.float32, 4): 6,

    # IDCT
    (idct, mx.float64, 1): 14,
    (idct, mx.float32, 1): 6,

    (idct, mx.float64, 2): 14,
    (idct, mx.float32, 2): 5,

    (idct, mx.float64, 3): 14,
    (idct, mx.float32, 3): 5,

    (idct, mx.float64, 4): 14,
    (idct, mx.float32, 4): 6,

    # DST
    (dst, mx.float64, 1): 13,
    (dst, mx.float32, 1): 6,

    (dst, mx.float64, 2): 14,
    (dst, mx.float32, 2): 6,

    (dst, mx.float64, 3): 14,
    (dst, mx.float32, 3): 7,

    (dst, mx.float64, 4): 13,
    (dst, mx.float32, 4): 5,

    # IDST
    (idst, mx.float64, 1): 14,
    (idst, mx.float32, 1): 6,

    (idst, mx.float64, 2): 14,
    (idst, mx.float32, 2): 6,

    (idst, mx.float64, 3): 14,
    (idst, mx.float32, 3): 6,

    (idst, mx.float64, 4): 14,
    (idst, mx.float32, 4): 6,
}

for k,v in dec_map.copy().items():
    if k[1] == mx.float64:
        dec_map[(k[0], mx.longdouble, k[2])] = v
    elif k[1] == mx.float32:
        dec_map[(k[0], int, k[2])] = v


@pytest.mark.parametrize('rdt', [mx.longdouble, mx.float64, mx.float32, int])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
class TestDCT:
    def test_definition(self, rdt, type, fftwdata_size,
                        reference_data, ref_lock):
        with ref_lock:
            x, yr, dt = fftw_dct_ref(type, fftwdata_size, rdt, reference_data)
        y = dct(x, type=type)
        assert_equal(y.dtype, dt)
        dec = dec_map[(dct, rdt, type)]
        assert_allclose(y, yr, rtol=0., atol=mx.max(yr)*10**(-dec))

    @pytest.mark.parametrize('size', [7, 8, 9, 16, 32, 64])
    def test_axis(self, rdt, type, size):
        nt = 2
        dec = dec_map[(dct, rdt, type)]
        x = mx.random.randn(nt, size)
        y = dct(x, type=type)
        for j in range(nt):
            assert_array_almost_equal(y[j], dct(x[j], type=type),
                                      decimal=dec)

        x = x.T
        y = dct(x, axis=0, type=type)
        for j in range(nt):
            assert_array_almost_equal(y[:,j], dct(x[:,j], type=type),
                                      decimal=dec)


@pytest.mark.parametrize('rdt', [mx.longdouble, mx.float64, mx.float32, int])
def test_dct1_definition_ortho(rdt, mdata_x):
    # Test orthornomal mode.
    dec = dec_map[(dct, rdt, 1)]
    x = mx.array(mdata_x, dtype=rdt)
    dt = mx.result_type(mx.float32, rdt)
    y = dct(x, norm='ortho', type=1)
    y2 = naive_dct1(x, norm='ortho')
    assert_equal(y.dtype, dt)
    assert_allclose(y, y2, rtol=0., atol=mx.max(y2)*10**(-dec))


@pytest.mark.parametrize('rdt', [mx.longdouble, mx.float64, mx.float32, int])
def test_dct2_definition_matlab(mdata_xy, rdt):
    # Test correspondence with matlab (orthornomal mode).
    dt = mx.result_type(mx.float32, rdt)
    x = mx.array(mdata_xy[0], dtype=dt)

    yr = mdata_xy[1]
    y = dct(x, norm="ortho", type=2)
    dec = dec_map[(dct, rdt, 2)]
    assert_equal(y.dtype, dt)
    assert_array_almost_equal(y, yr, decimal=dec)


@pytest.mark.parametrize('rdt', [mx.longdouble, mx.float64, mx.float32, int])
def test_dct3_definition_ortho(mdata_x, rdt):
    # Test orthornomal mode.
    x = mx.array(mdata_x, dtype=rdt)
    dt = mx.result_type(mx.float32, rdt)
    y = dct(x, norm='ortho', type=2)
    xi = dct(y, norm="ortho", type=3)
    dec = dec_map[(dct, rdt, 3)]
    assert_equal(xi.dtype, dt)
    assert_array_almost_equal(xi, x, decimal=dec)


@pytest.mark.parametrize('rdt', [mx.longdouble, mx.float64, mx.float32, int])
def test_dct4_definition_ortho(mdata_x, rdt):
    # Test orthornomal mode.
    x = mx.array(mdata_x, dtype=rdt)
    dt = mx.result_type(mx.float32, rdt)
    y = dct(x, norm='ortho', type=4)
    y2 = naive_dct4(x, norm='ortho')
    dec = dec_map[(dct, rdt, 4)]
    assert_equal(y.dtype, dt)
    assert_allclose(y, y2, rtol=0., atol=mx.max(y2)*10**(-dec))


@pytest.mark.parametrize('rdt', [mx.longdouble, mx.float64, mx.float32, int])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
def test_idct_definition(fftwdata_size, rdt, type, reference_data, ref_lock):
    with ref_lock:
        xr, yr, dt = fftw_dct_ref(type, fftwdata_size, rdt, reference_data)
    x = idct(yr, type=type)
    dec = dec_map[(idct, rdt, type)]
    assert_equal(x.dtype, dt)
    assert_allclose(x, xr, rtol=0., atol=mx.max(xr)*10**(-dec))


@pytest.mark.parametrize('rdt', [mx.longdouble, mx.float64, mx.float32, int])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
def test_definition(fftwdata_size, rdt, type, reference_data, ref_lock):
    with ref_lock:
        xr, yr, dt = fftw_dst_ref(type, fftwdata_size, rdt, reference_data)
    y = dst(xr, type=type)
    dec = dec_map[(dst, rdt, type)]
    assert_equal(y.dtype, dt)
    assert_allclose(y, yr, rtol=0., atol=mx.max(yr)*10**(-dec))


@pytest.mark.parametrize('rdt', [mx.longdouble, mx.float64, mx.float32, int])
def test_dst1_definition_ortho(rdt, mdata_x):
    # Test orthornomal mode.
    dec = dec_map[(dst, rdt, 1)]
    x = mx.array(mdata_x, dtype=rdt)
    dt = mx.result_type(mx.float32, rdt)
    y = dst(x, norm='ortho', type=1)
    y2 = naive_dst1(x, norm='ortho')
    assert_equal(y.dtype, dt)
    assert_allclose(y, y2, rtol=0., atol=mx.max(y2)*10**(-dec))


@pytest.mark.parametrize('rdt', [mx.longdouble, mx.float64, mx.float32, int])
def test_dst4_definition_ortho(rdt, mdata_x):
    # Test orthornomal mode.
    dec = dec_map[(dst, rdt, 4)]
    x = mx.array(mdata_x, dtype=rdt)
    dt = mx.result_type(mx.float32, rdt)
    y = dst(x, norm='ortho', type=4)
    y2 = naive_dst4(x, norm='ortho')
    assert_equal(y.dtype, dt)
    assert_array_almost_equal(y, y2, decimal=dec)


@pytest.mark.parametrize('rdt', [mx.longdouble, mx.float64, mx.float32, int])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
def test_idst_definition(fftwdata_size, rdt, type, reference_data, ref_lock):
    with ref_lock:
        xr, yr, dt = fftw_dst_ref(type, fftwdata_size, rdt, reference_data)
    x = idst(yr, type=type)
    dec = dec_map[(idst, rdt, type)]
    assert_equal(x.dtype, dt)
    assert_allclose(x, xr, rtol=0., atol=mx.max(xr)*10**(-dec))


@pytest.mark.parametrize('routine', [dct, dst, idct, idst])
@pytest.mark.parametrize('dtype', [mx.float32, mx.float64, mx.longdouble])
@pytest.mark.parametrize('shape, axis', [
    ((16,), -1), ((16, 2), 0), ((2, 16), 1)
])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
@pytest.mark.parametrize('overwrite_x', [True, False])
@pytest.mark.parametrize('norm', [None, 'ortho'])
def test_overwrite(routine, dtype, shape, axis, type, norm, overwrite_x):
    # Check input overwrite behavior
    mx.random.seed(1234)
    if mx.issubdtype(dtype, mx.complexfloating):
        x = mx.random.randn(*shape) + 1j*mx.random.randn(*shape)
    else:
        x = mx.random.randn(*shape)
    x = x.astype(dtype)
    x2 = x.copy()
    routine(x2, type, None, axis, norm, overwrite_x=overwrite_x)

    sig = (f"{routine.__name__}({x.dtype}{x.shape!r}, {None!r}, axis={axis!r}, "
           f"overwrite_x={overwrite_x!r})")
    if not overwrite_x:
        assert_equal(x2, x, err_msg=f"spurious overwrite in {sig}")


class Test_DCTN_IDCTN:
    dec = 14
    dct_type = [1, 2, 3, 4]
    norms = [None, 'backward', 'ortho', 'forward']
    rstate = mx.random.RandomState(1234)
    shape = (32, 16)
    data = rstate.randn(*shape)

    @pytest.mark.parametrize('fforward,finverse', [(dctn, idctn),
                                                   (dstn, idstn)])
    @pytest.mark.parametrize('axes', [None,
                                      1, (1,), [1],
                                      0, (0,), [0],
                                      (0, 1), [0, 1],
                                      (-2, -1), [-2, -1]])
    @pytest.mark.parametrize('dct_type', dct_type)
    @pytest.mark.parametrize('norm', ['ortho'])
    def test_axes_round_trip(self, fforward, finverse, axes, dct_type, norm):
        tmp = fforward(self.data, type=dct_type, axes=axes, norm=norm)
        tmp = finverse(tmp, type=dct_type, axes=axes, norm=norm)
        assert_array_almost_equal(self.data, tmp, decimal=12)

    @pytest.mark.parametrize('funcn,func', [(dctn, dct), (dstn, dst)])
    @pytest.mark.parametrize('dct_type', dct_type)
    @pytest.mark.parametrize('norm', norms)
    def test_dctn_vs_2d_reference(self, funcn, func, dct_type, norm):
        y1 = funcn(self.data, type=dct_type, axes=None, norm=norm)
        y2 = ref_2d(func, self.data, type=dct_type, norm=norm)
        assert_array_almost_equal(y1, y2, decimal=11)

    @pytest.mark.parametrize('funcn,func', [(idctn, idct), (idstn, idst)])
    @pytest.mark.parametrize('dct_type', dct_type)
    @pytest.mark.parametrize('norm', norms)
    def test_idctn_vs_2d_reference(self, funcn, func, dct_type, norm):
        fdata = dctn(self.data, type=dct_type, norm=norm)
        y1 = funcn(fdata, type=dct_type, norm=norm)
        y2 = ref_2d(func, fdata, type=dct_type, norm=norm)
        assert_array_almost_equal(y1, y2, decimal=11)

    @pytest.mark.parametrize('fforward,finverse', [(dctn, idctn),
                                                   (dstn, idstn)])
    def test_axes_and_shape(self, fforward, finverse):
        with assert_raises(ValueError,
                           match="when given, axes and shape arguments"
                           " have to be of the same length"):
            fforward(self.data, s=self.data.shape[0], axes=(0, 1))

        with assert_raises(ValueError,
                           match="when given, axes and shape arguments"
                           " have to be of the same length"):
            fforward(self.data, s=self.data.shape, axes=0)

    @pytest.mark.parametrize('fforward', [dctn, dstn])
    def test_shape(self, fforward):
        tmp = fforward(self.data, s=(128, 128), axes=None)
        assert_equal(tmp.shape, (128, 128))

    @pytest.mark.parametrize('fforward,finverse', [(dctn, idctn),
                                                   (dstn, idstn)])
    @pytest.mark.parametrize('axes', [1, (1,), [1],
                                      0, (0,), [0]])
    def test_shape_is_none_with_axes(self, fforward, finverse, axes):
        tmp = fforward(self.data, s=None, axes=axes, norm='ortho')
        tmp = finverse(tmp, s=None, axes=axes, norm='ortho')
        assert_array_almost_equal(self.data, tmp, decimal=self.dec)


@pytest.mark.parametrize('func', [dct, dctn, idct, idctn,
                                  dst, dstn, idst, idstn])
def test_swapped_byte_order(func):
    rng = mx.random.RandomState(1234)
    x = rng.rand(10)
    swapped_dt = x.dtype.newbyteorder('S')
    assert_allclose(func(x.astype(swapped_dt)), func(x))
