"""
MLX-compatible implementations of scipy.constants functions.

This module demonstrates converting scipy.constants functions to work with MLX arrays
while maintaining backward compatibility with NumPy.
"""

import math as _math
from typing import TYPE_CHECKING, Any

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except (ImportError, OSError):
    mx = None
    _MLX_AVAILABLE = False

if TYPE_CHECKING:
    import numpy.typing as npt

from scipy._lib._array_api import array_namespace, _asarray, xp_capabilities
from scipy.constants._codata import value as _cd


__all__ = ['convert_temperature_mlx', 'lambda2nu_mlx', 'nu2lambda_mlx']


# Physical constants (same as numpy version)
c = speed_of_light = _cd('speed of light in vacuum')
zero_Celsius = 273.15


@xp_capabilities()
def convert_temperature_mlx(
    val: "npt.ArrayLike",
    old_scale: str,
    new_scale: str,
) -> Any:
    """
    Convert from a temperature scale to another one among Celsius, Kelvin,
    Fahrenheit, and Rankine scales.

    This is an MLX-compatible version that works with both NumPy and MLX arrays.

    Parameters
    ----------
    val : array_like
        Value(s) of the temperature(s) to be converted expressed in the
        original scale. Can be NumPy array or MLX array.
    old_scale : str
        Specifies as a string the original scale from which the temperature
        value(s) will be converted. Supported scales are Celsius ('Celsius',
        'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),
        Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f'), and Rankine
        ('Rankine', 'rankine', 'R', 'r').
    new_scale : str
        Specifies as a string the new scale to which the temperature
        value(s) will be converted. Supported scales are Celsius ('Celsius',
        'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),
        Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f'), and Rankine
        ('Rankine', 'rankine', 'R', 'r').

    Returns
    -------
    res : float or array of floats
        Value(s) of the converted temperature(s) expressed in the new scale.

    Notes
    -----
    This function automatically detects whether the input is a NumPy array or
    MLX array and uses the appropriate backend.

    Examples
    --------
    >>> from scipy.constants._mlx_impl import convert_temperature_mlx
    >>> import numpy as np
    >>> convert_temperature_mlx(np.array([-40, 40]), 'Celsius', 'Kelvin')
    array([ 233.15,  313.15])

    With MLX (if available):

    >>> import mlx.core as mx  # doctest: +SKIP
    >>> convert_temperature_mlx(mx.array([-40, 40]), 'Celsius', 'Kelvin')  # doctest: +SKIP
    array([ 233.15,  313.15], dtype=float32)

    """
    xp = array_namespace(val)
    _val = _asarray(val, xp=xp, subok=True)
    
    # Convert from `old_scale` to Kelvin
    if old_scale.lower() in ['celsius', 'c']:
        tempo = _val + zero_Celsius
    elif old_scale.lower() in ['kelvin', 'k']:
        tempo = _val
    elif old_scale.lower() in ['fahrenheit', 'f']:
        tempo = (_val - 32) * 5 / 9 + zero_Celsius
    elif old_scale.lower() in ['rankine', 'r']:
        tempo = _val * 5 / 9
    else:
        raise NotImplementedError(f"{old_scale=} is unsupported: supported scales "
                                  "are Celsius, Kelvin, Fahrenheit, and "
                                  "Rankine")
    
    # and from Kelvin to `new_scale`.
    if new_scale.lower() in ['celsius', 'c']:
        res = tempo - zero_Celsius
    elif new_scale.lower() in ['kelvin', 'k']:
        res = tempo
    elif new_scale.lower() in ['fahrenheit', 'f']:
        res = (tempo - zero_Celsius) * 9 / 5 + 32
    elif new_scale.lower() in ['rankine', 'r']:
        res = tempo * 9 / 5
    else:
        raise NotImplementedError(f"{new_scale=} is unsupported: supported "
                                  "scales are 'Celsius', 'Kelvin', "
                                  "'Fahrenheit', and 'Rankine'")
    
    return res


@xp_capabilities()
def lambda2nu_mlx(lambda_: "npt.ArrayLike") -> Any:
    """
    Convert wavelength to optical frequency (MLX-compatible version).

    Parameters
    ----------
    lambda_ : array_like
        Wavelength(s) to be converted. Can be NumPy or MLX array.

    Returns
    -------
    nu : float or array of floats
        Equivalent optical frequency.

    Notes
    -----
    Computes ``nu = c / lambda`` where c = 299792458.0, i.e., the
    (vacuum) speed of light in meters/second.

    This function automatically detects the array backend (NumPy or MLX)
    and uses the appropriate implementation.

    Examples
    --------
    >>> from scipy.constants._mlx_impl import lambda2nu_mlx, speed_of_light
    >>> import numpy as np
    >>> lambda2nu_mlx(np.array((1, speed_of_light)))
    array([  2.99792458e+08,   1.00000000e+00])

    With MLX:

    >>> import mlx.core as mx  # doctest: +SKIP
    >>> lambda2nu_mlx(mx.array((1, speed_of_light)))  # doctest: +SKIP
    array([  2.99792458e+08,   1.00000000e+00], dtype=float32)

    """
    xp = array_namespace(lambda_)
    return c / _asarray(lambda_, xp=xp, subok=True)


@xp_capabilities()
def nu2lambda_mlx(nu: "npt.ArrayLike") -> Any:
    """
    Convert optical frequency to wavelength (MLX-compatible version).

    Parameters
    ----------
    nu : array_like
        Optical frequency to be converted. Can be NumPy or MLX array.

    Returns
    -------
    lambda : float or array of floats
        Equivalent wavelength(s).

    Notes
    -----
    Computes ``lambda = c / nu`` where c = 299792458.0, i.e., the
    (vacuum) speed of light in meters/second.

    This function automatically detects the array backend (NumPy or MLX)
    and uses the appropriate implementation.

    Examples
    --------
    >>> from scipy.constants._mlx_impl import nu2lambda_mlx, speed_of_light
    >>> import numpy as np
    >>> nu2lambda_mlx(np.array((1, speed_of_light)))
    array([  2.99792458e+08,   1.00000000e+00])

    With MLX:

    >>> import mlx.core as mx  # doctest: +SKIP
    >>> nu2lambda_mlx(mx.array((1, speed_of_light)))  # doctest: +SKIP
    array([  2.99792458e+08,   1.00000000e+00], dtype=float32)

    """
    xp = array_namespace(nu)
    return c / _asarray(nu, xp=xp, subok=True)


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    print("Testing with NumPy arrays:")
    print("-" * 50)
    
    # Test temperature conversion
    temps_c = np.array([0, 100, -40])
    temps_f = convert_temperature_mlx(temps_c, 'Celsius', 'Fahrenheit')
    print(f"Celsius: {temps_c}")
    print(f"Fahrenheit: {temps_f}")
    
    # Test lambda2nu
    wavelengths = np.array([1, speed_of_light])
    frequencies = lambda2nu_mlx(wavelengths)
    print(f"\nWavelengths: {wavelengths}")
    print(f"Frequencies: {frequencies}")
    
    # Test nu2lambda
    freqs = np.array([1, speed_of_light])
    waves = nu2lambda_mlx(freqs)
    print(f"\nFrequencies: {freqs}")
    print(f"Wavelengths: {waves}")
    
    # Test with MLX if available
    if _MLX_AVAILABLE:
        print("\n" + "=" * 50)
        print("Testing with MLX arrays:")
        print("-" * 50)
        
        try:
            temps_c_mlx = mx.array([0, 100, -40])
            temps_f_mlx = convert_temperature_mlx(temps_c_mlx, 'Celsius', 'Fahrenheit')
            print(f"Celsius (MLX): {temps_c_mlx}")
            print(f"Fahrenheit (MLX): {temps_f_mlx}")
            
            wavelengths_mlx = mx.array([1, speed_of_light])
            frequencies_mlx = lambda2nu_mlx(wavelengths_mlx)
            print(f"\nWavelengths (MLX): {wavelengths_mlx}")
            print(f"Frequencies (MLX): {frequencies_mlx}")
        except Exception as e:
            print(f"MLX test failed: {e}")
    else:
        print("\n" + "=" * 50)
        print("MLX not available. Install MLX to test with MLX arrays.")
