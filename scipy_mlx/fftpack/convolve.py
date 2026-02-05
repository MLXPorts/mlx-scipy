"""
MLX fallback implementation for legacy `scipy.fftpack.convolve`.

Upstream SciPy exposes a compiled `fftpack.convolve` module. This MLX port does
not ship compiled extensions; provide minimal, import-safe implementations.

These routines are used by `scipy_mlx.fftpack._pseudo_diffs` for circular
convolution kernels.
"""

from __future__ import annotations

import mlx.core as mx


def init_convolution_kernel(n, kernel, d=0, zero_nyquist=False):
    """
    Prepare a circular convolution kernel of length `n`.

    This is a simplified implementation that returns the FFT of the kernel.
    """
    k = mx.array(kernel)
    if int(k.size) != int(n):
        # Pad or crop to length n.
        k = mx.reshape(k, (-1,))
        cur = int(k.shape[0])
        if cur > int(n):
            k = k[: int(n)]
        elif cur < int(n):
            k = mx.pad(k, [(0, int(n) - cur)], mode="constant", constant_values=0.0)

    # Derivative order `d` corresponds to multiplying by (i*omega)^d in freq.
    omega = mx.arange(int(n))
    # Map frequencies to [-n/2, n/2) like fftfreq.
    half = int(n) // 2
    omega = mx.where(omega <= half, omega, omega - int(n))
    iomega = mx.multiply(mx.array(1j), mx.array(2.0) * mx.pi * mx.array(omega) / mx.array(float(n)))
    factor = mx.power(iomega, mx.array(int(d))) if int(d) != 0 else mx.array(1.0)
    if int(d) != 0:
        factor = mx.array(factor)

    K = mx.fft.fft(k)
    if int(d) != 0:
        K = mx.multiply(K, factor)
    return K


def convolve(x, omega, swap_real_imag=0, overwrite_x=False):
    """Circular convolution using an FFT-domain kernel `omega`."""
    x = mx.array(x)
    X = mx.fft.fft(x)
    y = mx.fft.ifft(mx.multiply(X, mx.array(omega)))
    if swap_real_imag:
        y = mx.add(mx.multiply(mx.array(1j), mx.imag(y)), mx.real(y))
    return y


def convolve_z(x, omega_real, omega_imag, overwrite_x=False):
    """Circular convolution for complex-valued kernels split into real/imag parts."""
    x = mx.array(x)
    omega = mx.add(mx.array(omega_real), mx.multiply(mx.array(1j), mx.array(omega_imag)))
    return convolve(x, omega, overwrite_x=overwrite_x)


def destroy_convolve_cache():
    return None

