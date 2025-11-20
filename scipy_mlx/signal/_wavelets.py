import mlx.core as mx
from scipy_mlx.signal._signaltools import convolve


def _ricker(points, a):
    A = 2 / (mx.sqrt(3 * a) * (mx.pi**0.25))
    wsq = a**2
    vec = mx.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = (1 - xsq / wsq)
    gauss = mx.exp(-xsq / (2 * wsq))
    total = A * mod * gauss
    return total


def _cwt(data, wavelet, widths, dtype=None, **kwargs):
    # Determine output type
    if dtype is None:
        if mx.array(wavelet(1, widths[0], **kwargs)).dtype.char in 'FDG':
            dtype = mx.complex128
        else:
            dtype = mx.float64

    output = mx.empty((len(widths), len(data)), dtype=dtype)
    for ind, width in enumerate(widths):
        N = mx.min([10 * width, len(data)])
        wavelet_data = mx.conj(wavelet(N, width, **kwargs)[::-1])
        output[ind] = convolve(data, wavelet_data, mode='same')
    return output
