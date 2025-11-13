"""
Standalone test for MLX conversion patterns.

This file can be run without building scipy to verify the conversion logic.
"""

import numpy as np

# Simulate the array_namespace function
def array_namespace(x):
    """Simple version of array namespace detection."""
    if hasattr(x, '__array_namespace__'):
        return x.__array_namespace__()
    # Default to numpy
    import numpy as np
    return np

def _asarray(x, xp=None, **kwargs):
    """Simple version of asarray."""
    if xp is None:
        xp = array_namespace(x)
    if hasattr(xp, 'asarray'):
        return xp.asarray(x)
    return xp.array(x)


# Physical constants
c = speed_of_light = 299792458.0
zero_Celsius = 273.15


def convert_temperature_mlx(val, old_scale, new_scale):
    """
    MLX-compatible temperature conversion.
    
    Works with NumPy arrays and would work with MLX arrays too.
    """
    xp = array_namespace(val)
    _val = _asarray(val, xp=xp)
    
    # Convert from old_scale to Kelvin
    if old_scale.lower() in ['celsius', 'c']:
        tempo = _val + zero_Celsius
    elif old_scale.lower() in ['kelvin', 'k']:
        tempo = _val
    elif old_scale.lower() in ['fahrenheit', 'f']:
        tempo = (_val - 32) * 5 / 9 + zero_Celsius
    elif old_scale.lower() in ['rankine', 'r']:
        tempo = _val * 5 / 9
    else:
        raise NotImplementedError(f"Unsupported scale: {old_scale}")
    
    # Convert from Kelvin to new_scale
    if new_scale.lower() in ['celsius', 'c']:
        res = tempo - zero_Celsius
    elif new_scale.lower() in ['kelvin', 'k']:
        res = tempo
    elif new_scale.lower() in ['fahrenheit', 'f']:
        res = (tempo - zero_Celsius) * 9 / 5 + 32
    elif new_scale.lower() in ['rankine', 'r']:
        res = tempo * 9 / 5
    else:
        raise NotImplementedError(f"Unsupported scale: {new_scale}")
    
    return res


def lambda2nu_mlx(lambda_):
    """Convert wavelength to optical frequency."""
    xp = array_namespace(lambda_)
    return c / _asarray(lambda_, xp=xp)


def nu2lambda_mlx(nu):
    """Convert optical frequency to wavelength."""
    xp = array_namespace(nu)
    return c / _asarray(nu, xp=xp)


def test_conversions():
    """Test the conversion functions."""
    print("=" * 60)
    print("Testing MLX-Compatible Conversion Functions")
    print("=" * 60)
    
    # Test temperature conversion
    print("\n1. Temperature Conversion:")
    print("-" * 40)
    temps_c = np.array([0, 100, -40])
    temps_f = convert_temperature_mlx(temps_c, 'Celsius', 'Fahrenheit')
    temps_k = convert_temperature_mlx(temps_c, 'Celsius', 'Kelvin')
    
    print(f"Celsius:    {temps_c}")
    print(f"Fahrenheit: {temps_f}")
    print(f"Kelvin:     {temps_k}")
    
    # Verify some known conversions
    assert abs(temps_f[0] - 32.0) < 1e-6, "0°C should be 32°F"
    assert abs(temps_f[1] - 212.0) < 1e-6, "100°C should be 212°F"
    assert abs(temps_f[2] - (-40.0)) < 1e-6, "-40°C should be -40°F"
    assert abs(temps_k[0] - 273.15) < 1e-6, "0°C should be 273.15K"
    print("✓ Temperature conversion tests passed!")
    
    # Test wavelength to frequency conversion
    print("\n2. Wavelength to Frequency Conversion:")
    print("-" * 40)
    wavelengths = np.array([1, speed_of_light])
    frequencies = lambda2nu_mlx(wavelengths)
    print(f"Wavelengths (m): {wavelengths}")
    print(f"Frequencies (Hz): {frequencies}")
    
    # Verify: lambda * nu = c
    products = wavelengths * frequencies
    print(f"λ × ν = {products}")
    print(f"Speed of light c = {c}")
    assert np.allclose(products, c), "λ × ν should equal c"
    print("✓ Wavelength to frequency conversion tests passed!")
    
    # Test frequency to wavelength conversion
    print("\n3. Frequency to Wavelength Conversion:")
    print("-" * 40)
    freqs = np.array([1e9, 1e15])  # 1 GHz and 1 PHz
    waves = nu2lambda_mlx(freqs)
    print(f"Frequencies (Hz): {freqs}")
    print(f"Wavelengths (m): {waves}")
    
    # Verify: lambda * nu = c
    products = freqs * waves
    print(f"ν × λ = {products}")
    assert np.allclose(products, c), "ν × λ should equal c"
    print("✓ Frequency to wavelength conversion tests passed!")
    
    # Test MLX if available
    print("\n4. Testing MLX Backend:")
    print("-" * 40)
    try:
        import mlx.core as mx
        print("✓ MLX is available!")
        
        # Try to create an MLX array and convert
        temps_c_mlx = mx.array([0.0, 100.0, -40.0])
        print(f"MLX array created: {temps_c_mlx}")
        
        # This would work if MLX is properly installed
        temps_f_mlx = convert_temperature_mlx(temps_c_mlx, 'Celsius', 'Fahrenheit')
        print(f"Converted to Fahrenheit (MLX): {temps_f_mlx}")
        print("✓ MLX conversion successful!")
        
    except (ImportError, OSError) as e:
        print(f"✗ MLX not available: {type(e).__name__}")
        print("  (This is expected on non-Apple Silicon systems)")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_conversions()
