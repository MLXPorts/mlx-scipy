"""
Example conversion of NumPy code to MLX.

This demonstrates the pattern for converting scipy functions from NumPy to MLX.
MLX is Apple's machine learning framework optimized for Apple Silicon.

Key conversion steps:
1. Import mlx.core as mx instead of numpy as np
2. Replace np.array() with mx.array()
3. Most NumPy operations have direct MLX equivalents
4. Use the array API compatibility layer for flexible backend support
"""

# Example 1: Basic array operations
# Before (NumPy):
# import numpy as np
# def add_arrays(a, b):
#     arr_a = np.array(a)
#     arr_b = np.array(b)
#     return arr_a + arr_b

# After (MLX):
# import mlx.core as mx
# def add_arrays(a, b):
#     arr_a = mx.array(a)
#     arr_b = mx.array(b)
#     return arr_a + arr_b


# Example 2: Using array API for flexibility
from scipy._lib._array_api import array_namespace, _asarray


def convert_temperature_mlx(val, old_scale, new_scale):
    """
    Example of temperature conversion using array API.
    
    This works with NumPy, MLX, or any other array API compatible backend.
    """
    # Get the appropriate array namespace (numpy, mlx, etc.)
    xp = array_namespace(val)
    
    # Convert to array using the detected backend
    _val = _asarray(val, xp=xp, subok=True)
    
    # Constants work the same across backends
    zero_Celsius = 273.15
    
    # Convert from old_scale to Kelvin
    if old_scale.lower() in ['celsius', 'c']:
        tempo = _val + zero_Celsius
    elif old_scale.lower() in ['kelvin', 'k']:
        tempo = _val
    elif old_scale.lower() in ['fahrenheit', 'f']:
        tempo = (_val - 32) * 5 / 9 + zero_Celsius
    else:
        raise NotImplementedError(f"{old_scale=} is unsupported")
    
    # Convert from Kelvin to new_scale
    if new_scale.lower() in ['celsius', 'c']:
        res = tempo - zero_Celsius
    elif new_scale.lower() in ['kelvin', 'k']:
        res = tempo
    elif new_scale.lower() in ['fahrenheit', 'f']:
        res = (tempo - zero_Celsius) * 9 / 5 + 32
    else:
        raise NotImplementedError(f"{new_scale=} is unsupported")
    
    return res


# Example 3: MLX-specific array creation
"""
Common NumPy to MLX conversions:

NumPy                          MLX
------                         ----
np.array([1, 2, 3])           mx.array([1, 2, 3])
np.zeros((3, 3))              mx.zeros((3, 3))
np.ones((2, 2))               mx.ones((2, 2))
np.arange(10)                 mx.arange(10)
np.linspace(0, 1, 10)         mx.linspace(0, 1, 10)
np.random.rand(3, 3)          mx.random.uniform(shape=(3, 3))
np.dot(a, b)                  mx.matmul(a, b) or a @ b
np.sum(a, axis=0)             mx.sum(a, axis=0)
np.mean(a)                    mx.mean(a)
np.sqrt(a)                    mx.sqrt(a)
np.exp(a)                     mx.exp(a)
"""


if __name__ == "__main__":
    # Test with NumPy (fallback)
    import numpy as np
    result = convert_temperature_mlx(np.array([0, 100]), 'Celsius', 'Fahrenheit')
    print(f"NumPy backend result: {result}")
    
    # Test with MLX (if available)
    try:
        import mlx.core as mx
        result_mlx = convert_temperature_mlx(mx.array([0, 100]), 'Celsius', 'Fahrenheit')
        print(f"MLX backend result: {result_mlx}")
    except (ImportError, OSError):
        print("MLX not available (requires Apple Silicon or compatible hardware)")
