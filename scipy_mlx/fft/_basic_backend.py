from scipy_mlx._lib._array_api import (
    array_namespace, xp_unsupported_param_msg, is_complex, xp_float_to_complex
)
import mlx.core as mx


def _validate_fft_args(workers, plan, norm):
    if workers is not None:
        raise ValueError(xp_unsupported_param_msg("workers"))
    if plan is not None:
        raise ValueError(xp_unsupported_param_msg("plan"))
    if norm is None:
        norm = 'backward'
    return norm


# these functions expect complex input in the fft standard extension
complex_funcs = {'fft', 'ifft', 'fftn', 'ifftn', 'hfft', 'irfft', 'irfftn'}

# pocketfft is used whenever SCIPY_ARRAY_API is not set,
# or x is a NumPy array or array-like.
# When SCIPY_ARRAY_API is set, we try to use xp.fft for CuPy arrays,
# PyTorch arrays and other array API standard supporting objects.
# If xp.fft does not exist, we attempt to convert to np and back to use pocketfft.

def _norm_scale(norm: str, forward: bool, n: int):
    # MLX follows NumPy's default FFT normalization ("backward"):
    # forward: unscaled, inverse: scaled by 1/n.
    if norm is None:
        norm = "backward"
    if norm not in ("backward", "ortho", "forward"):
        raise ValueError(f'Invalid norm value {norm!r}, should be "backward", "ortho" or "forward"')

    n_f = mx.array(float(n))
    if norm == "backward":
        return mx.array(1.0)
    if norm == "ortho":
        return mx.divide(mx.array(1.0), mx.sqrt(n_f)) if forward else mx.sqrt(n_f)
    # norm == "forward"
    return mx.divide(mx.array(1.0), n_f) if forward else n_f


def _pad_or_crop_1d(x: mx.array, n: int | None, axis: int):
    if n is None:
        return x
    axis = int(axis)
    cur = int(x.shape[axis])
    if cur == n:
        return x
    if cur > n:
        slc = [slice(None)] * x.ndim
        slc[axis] = slice(0, n)
        return x[tuple(slc)]
    pad = n - cur
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad)
    return mx.pad(x, pad_width, mode="constant", constant_values=0.0)


def _execute_1D(func_str, x, n, axis, norm, overwrite_x, workers, plan):
    xp = array_namespace(x)
    norm = _validate_fft_args(workers, plan, norm)

    if xp is not mx:
        raise ValueError(xp_unsupported_param_msg("backend"))

    x = mx.array(x)
    x = _pad_or_crop_1d(x, n, axis)
    n_eff = int(x.shape[int(axis)])

    if func_str == "fft":
        y = mx.fft.fft(x, n=n_eff, axis=axis)
        return mx.multiply(y, _norm_scale(norm, True, n_eff))
    if func_str == "ifft":
        y = mx.fft.ifft(x, n=n_eff, axis=axis)
        return mx.multiply(y, _norm_scale(norm, False, n_eff))
    if func_str == "rfft":
        y = mx.fft.rfft(x, n=n_eff, axis=axis)
        return mx.multiply(y, _norm_scale(norm, True, n_eff))
    if func_str == "irfft":
        y = mx.fft.irfft(x, n=n_eff, axis=axis)
        return mx.multiply(y, _norm_scale(norm, False, n_eff))
    if func_str == "hfft":
        y = mx.fft.irfft(mx.conjugate(x), n=n_eff, axis=axis)
        return mx.multiply(y, _norm_scale(norm, True, n_eff))
    if func_str == "ihfft":
        y = mx.conjugate(mx.fft.rfft(x, n=n_eff, axis=axis))
        return mx.multiply(y, _norm_scale(norm, False, n_eff))
    raise ValueError(f"unsupported FFT function {func_str!r}")


def _pad_or_crop_nd(x: mx.array, s, axes):
    if s is None:
        return x, axes
    if axes is None:
        axes = tuple(range(x.ndim - len(s), x.ndim))
    axes = tuple(int(a) for a in axes)
    if len(s) != len(axes):
        raise ValueError("when given, axes and shape arguments have to be of the same length")
    out = x
    for n, ax in zip(s, axes):
        out = _pad_or_crop_1d(out, int(n), ax)
    return out, axes


def _execute_nD(func_str, x, s, axes, norm, overwrite_x, workers, plan):
    xp = array_namespace(x)
    
    norm = _validate_fft_args(workers, plan, norm)
    if xp is not mx:
        raise ValueError(xp_unsupported_param_msg("backend"))

    x = mx.array(x)
    x, axes = _pad_or_crop_nd(x, s, axes)
    n_eff = 1
    for ax in axes:
        n_eff *= int(x.shape[ax])

    if func_str == "fftn":
        y = mx.fft.fftn(x, axes=axes)
        return mx.multiply(y, _norm_scale(norm, True, n_eff))
    if func_str == "ifftn":
        y = mx.fft.ifftn(x, axes=axes)
        return mx.multiply(y, _norm_scale(norm, False, n_eff))
    if func_str == "rfftn":
        y = mx.fft.rfftn(x, axes=axes)
        return mx.multiply(y, _norm_scale(norm, True, n_eff))
    if func_str == "irfftn":
        y = mx.fft.irfftn(x, s=s, axes=axes)
        return mx.multiply(y, _norm_scale(norm, False, n_eff))
    raise ValueError(f"unsupported FFT function {func_str!r}")


def fft(x, n=None, axis=-1, norm=None,
        overwrite_x=False, workers=None, *, plan=None):
    return _execute_1D('fft', x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def ifft(x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *,
         plan=None):
    return _execute_1D('ifft', x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def rfft(x, n=None, axis=-1, norm=None,
         overwrite_x=False, workers=None, *, plan=None):
    return _execute_1D('rfft', x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def irfft(x, n=None, axis=-1, norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return _execute_1D('irfft', x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def hfft(x, n=None, axis=-1, norm=None,
         overwrite_x=False, workers=None, *, plan=None):
    return _execute_1D('hfft', x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def ihfft(x, n=None, axis=-1, norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return _execute_1D('ihfft', x, n=n, axis=axis, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def fftn(x, s=None, axes=None, norm=None,
         overwrite_x=False, workers=None, *, plan=None):
    return _execute_nD('fftn', x, s=s, axes=axes, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)



def ifftn(x, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return _execute_nD('ifftn', x, s=s, axes=axes, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def fft2(x, s=None, axes=(-2, -1), norm=None,
         overwrite_x=False, workers=None, *, plan=None):
    return fftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


def ifft2(x, s=None, axes=(-2, -1), norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return ifftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


def rfftn(x, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return _execute_nD('rfftn', x, s=s, axes=axes, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def rfft2(x, s=None, axes=(-2, -1), norm=None,
         overwrite_x=False, workers=None, *, plan=None):
    return rfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


def irfftn(x, s=None, axes=None, norm=None,
           overwrite_x=False, workers=None, *, plan=None):
    return _execute_nD('irfftn', _pocketfft.irfftn, x, s=s, axes=axes, norm=norm,
                       overwrite_x=overwrite_x, workers=workers, plan=plan)


def irfft2(x, s=None, axes=(-2, -1), norm=None,
           overwrite_x=False, workers=None, *, plan=None):
    return irfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


def _swap_direction(norm):
    if norm in (None, 'backward'):
        norm = 'forward'
    elif norm == 'forward':
        norm = 'backward'
    elif norm != 'ortho':
        raise ValueError(f'Invalid norm value {norm}; should be "backward", '
                         '"ortho", or "forward".')
    return norm


def hfftn(x, s=None, axes=None, norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    xp = array_namespace(x)
    if is_numpy(xp):
        x = mx.array(x)
        return _pocketfft.hfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)
    if is_complex(x, xp):
        x = xp.conj(x)
    return irfftn(x, s, axes, _swap_direction(norm),
                  overwrite_x, workers, plan=plan)


def hfft2(x, s=None, axes=(-2, -1), norm=None,
          overwrite_x=False, workers=None, *, plan=None):
    return hfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)


def ihfftn(x, s=None, axes=None, norm=None,
           overwrite_x=False, workers=None, *, plan=None):
    xp = array_namespace(x)
    if is_numpy(xp):
        x = mx.array(x)
        return _pocketfft.ihfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)
    return xp.conj(rfftn(x, s, axes, _swap_direction(norm),
                         overwrite_x, workers, plan=plan))

def ihfft2(x, s=None, axes=(-2, -1), norm=None,
           overwrite_x=False, workers=None, *, plan=None):
    return ihfftn(x, s, axes, norm, overwrite_x, workers, plan=plan)
