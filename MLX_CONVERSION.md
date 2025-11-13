# MLX Conversion Guide for SciPy

This document outlines the process of converting SciPy code from NumPy to MLX.

## Overview

MLX (Machine Learning eXtensions) is Apple's machine learning framework optimized for Apple Silicon. It provides a NumPy-like API, making conversion relatively straightforward.

## Conversion Strategy

### Phase 1: Infrastructure (COMPLETED)
- [x] Add MLX as a dependency in `pyproject.toml`
- [x] Create MLX compatibility layer (`scipy/_lib/_mlx_compat.py`)
- [x] Integrate MLX detection into array API (`scipy/_lib/_array_api.py`)
- [x] Create conversion examples

### Phase 2: Core Module Conversion (IN PROGRESS)
- [ ] scipy.constants - minimal NumPy usage, good starting point
- [ ] scipy.special - mathematical functions
- [ ] scipy.linalg - linear algebra operations
- [ ] scipy.fft - Fast Fourier Transform
- [ ] Other modules progressively

### Phase 3: Testing and Validation
- [ ] Add MLX-specific tests
- [ ] Verify numerical accuracy
- [ ] Performance benchmarking
- [ ] Documentation updates

## Key Files Modified

1. **pyproject.toml**
   - Added `mlx>=0.29.0` dependency

2. **scipy/_lib/_mlx_compat.py** (NEW)
   - `is_mlx_array()` - Check if object is MLX array
   - `is_mlx_namespace()` - Check if module is MLX
   - `mlx_available` - Boolean flag for MLX availability

3. **scipy/_lib/_array_api.py**
   - Imported and exported MLX detection functions
   - Integrated MLX into array API layer

4. **scipy/_lib/_mlx_conversion_example.py** (NEW)
   - Comprehensive conversion examples
   - Common operation mappings
   - Template for future conversions

## Conversion Patterns

### Pattern 1: Direct NumPy to MLX Replacement

```python
# Before (NumPy)
import numpy as np
result = np.array([1, 2, 3])
output = np.sum(result)

# After (MLX)
import mlx.core as mx
result = mx.array([1, 2, 3])
output = mx.sum(result)
```

### Pattern 2: Array API Compatibility (RECOMMENDED)

```python
# This works with NumPy, MLX, CuPy, JAX, etc.
from scipy._lib._array_api import array_namespace, _asarray

def my_function(x):
    xp = array_namespace(x)
    x_arr = _asarray(x, xp=xp)
    return xp.sum(x_arr)
```

### Pattern 3: Conditional Import

```python
try:
    import mlx.core as mx
    USE_MLX = True
except (ImportError, OSError):
    import numpy as np
    mx = np
    USE_MLX = False
```

## Common API Mappings

| NumPy | MLX | Notes |
|-------|-----|-------|
| `np.array()` | `mx.array()` | Basic array creation |
| `np.zeros()` | `mx.zeros()` | Create zeros array |
| `np.ones()` | `mx.ones()` | Create ones array |
| `np.arange()` | `mx.arange()` | Create range array |
| `np.linspace()` | `mx.linspace()` | Linear spacing |
| `np.random.rand()` | `mx.random.uniform()` | Random numbers |
| `np.dot()` | `mx.matmul()` or `@` | Matrix multiplication |
| `np.sum()` | `mx.sum()` | Sum operation |
| `np.mean()` | `mx.mean()` | Mean operation |
| `np.sqrt()` | `mx.sqrt()` | Square root |
| `np.exp()` | `mx.exp()` | Exponential |

## Important Considerations

### 1. Hardware Requirements
- MLX is optimized for Apple Silicon (M1, M2, M3, etc.)
- CPU fallback available but may be slower
- Linux x86_64 packages exist but require compatible hardware

### 2. Lazy Evaluation
- MLX uses lazy evaluation for better performance
- Operations are computed only when needed
- Use `mx.eval()` to force computation if needed

### 3. Memory Model
- MLX uses unified memory on Apple Silicon
- No explicit CPU/GPU transfers needed
- Automatic optimization for available hardware

### 4. Backward Compatibility
- Use Array API for maximum compatibility
- Keep NumPy as fallback for non-Apple hardware
- Test on multiple platforms

## Testing

### Running Tests
```bash
# Test with NumPy (default)
pytest scipy/constants/tests/

# Test with MLX (if available)
SCIPY_ARRAY_API=1 pytest scipy/constants/tests/
```

### Adding MLX Tests
```python
import pytest
from scipy._lib._mlx_compat import mlx_available

@pytest.mark.skipif(not mlx_available, reason="MLX not available")
def test_with_mlx():
    import mlx.core as mx
    # Test code here
```

## Next Steps

1. Continue converting scipy.constants module
2. Add comprehensive tests for MLX backend
3. Document any API differences
4. Create migration guide for users
5. Performance benchmarking

## References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [Array API Standard](https://data-apis.org/array-api/latest/)
- [NumPy to MLX Migration](https://ml-explore.github.io/mlx/build/html/usage/numpy.html)
