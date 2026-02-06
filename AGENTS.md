# MLX Conversion Guidelines

## Overview
This document outlines the patterns and standards for converting SciPy modules from NumPy to MLX (Apple Metal Performance Shaders) backend.

## Core Conversion Patterns

### 1. Import Replacement
```python
import mlx.core as mx
from mlx.core import array, zeros, ones
```

### 2. Function Name Mapping
Most NumPy functions have direct MLX equivalents:
- `np.array()` → `mx.array()`
- `np.zeros()` → `mx.zeros()`
- `np.ones()` → `mx.ones()`
- `np.reshape()` → `mx.reshape()`
- `np.concatenate()` → `mx.concatenate()`
- `np.dot()` → `mx.matmul()` (or `mx.multiply()` for element-wise)
- `np.diag()` → `mx.diag()`
- `np.triu()` → `mx.triu()`
- `np.tril()` → `mx.tril()`
- `np.argsort()` → `mx.argsort()`
- `np.where()` → `mx.where()`
- `np.clip()` → `mx.clip()`
- `np.conj()` → `mx.conjugate()`

### 3. Type Hint Updates
```python
from typing import Any as ArrayLike
def func(val: "ArrayLike") -> Any: ...
```

### 4. Docstring Examples
```python
>>> import mlx.core as mx
>>> convert_temperature(mx.array([-40, 40]), 'Celsius', 'Kelvin')
```

## Critical Standards

### 5. No `.item()` Calls
Never call `.item()` on MLX arrays in compute paths - it breaks lazy execution:
```python
# ❌ BAD - breaks lazy execution
mx.array([[mx.exp(a.item())]])

# ✅ GOOD - preserves lazy execution
mx.reshape(mx.exp(a), (1, 1))
```

### 6. Wrap Python Scalars
Always wrap Python scalars in `mx.array()` to avoid float64 promotion:
```python
# ❌ BAD - promotes to float64
2.**(-s)
-0.5j * x
1j * A

# ✅ GOOD - preserves dtype
mx.power(mx.array(2.0), mx.array(-s))
mx.multiply(mx.array(-0.5j), x)
mx.multiply(mx.array(1j), A)
```

### 7. Use Explicit MLX Functions
Prefer explicit MLX functions over Python operators with tensors:
```python
# ❌ BAD - may cause issues
result = a + b * c

# ✅ GOOD - explicit and clear
result = mx.add(a, mx.multiply(b, c))
```

### 8. Compatibility Layers
For MLX-missing functions, create compatibility modules:
```python
# In scipy/linalg/_mlx_compat.py
def asarray_chkfinite(a, dtype=None, order=None):
    """MLX version with NaN/Inf checking."""
    a = mx.asarray(a, dtype=dtype)
    if mx.any(mx.isnan(a)) or mx.any(mx.isinf(a)):
        raise ValueError("array must not contain infs or NaNs")
    return a

def flatnonzero(a):
    """MLX version of flatnonzero."""
    return mx.nonzero(mx.reshape(a, [-1]))[0]
```

## Testing Guidelines

### 9. Test File Updates
Update test files to use MLX, and use plain `pytest` assertions:
```python
# Test files should import
import mlx.core as mx
```

### 10. Verification Steps
After conversion:
1. ✅ No NumPy imports in compute paths
2. ✅ No `.item()` or `.numpy()` calls mid-graph
3. ✅ All literals wrapped in `mx.array()`
4. ✅ Lazy execution preserved (no premature evaluation)
5. ✅ No device hops or buffer breaks
6. ✅ Type hints updated
7. ✅ Docstring examples updated

## Conversion Process Example

### After (MLX):
```python
import mlx.core as mx

def compute(x, scale=2.0):
    y = mx.array(x)
    # ✅ All operations use MLX, scalars wrapped
    result = mx.add(mx.multiply(mx.exp(y), mx.array(scale)), mx.array(1.0))
    return result  # ✅ No .item() call
```

## Module Structure

### 11. Compatibility Module Pattern
```python
# scipy/[module]/_mlx_compat.py
"""
MLX compatibility layer for [module] module.

Provides MLX implementations of functions that don't exist in MLX core.
"""

import mlx.core as mx
from typing import Union, Optional

def missing_function1(x, ...):
    """MLX implementation of missing function."""
    ...

def missing_function2(x, ...):
    """Another MLX implementation."""
    ...
```

### 12. Module __init__.py Update
```python
# scipy/[module]/__init__.py
from ._mlx_compat import *  # Import compat functions

# Rest of module exports...
```

## Common Pitfalls

1. **Python scalar math**: Always wrap in `mx.array()`
2. **`.item()` calls**: Never in compute paths
3. **Forgotten imports**: Check all files for NumPy remnants
4. **Type hints**: Update from `numpy.typing` to `typing.Any`
5. **Docstrings**: Update examples to use `mlx.core as mx`
6. **Constants**: Wrap module-level constants in `mx.array()`
7. **Boolean indexing**: MLX may handle differently - verify behavior
8. **Stride/tranpose ops**: MLX uses different conventions

## Commit Message Format

Use this format for MLX conversion commits:

```
Replace NumPy with MLX in scipy.[module] module

Complete migration of scipy.[module] from NumPy to MLX backend.

## Changes

### Core Replacements ([n] files):
- Removed all NumPy imports
- Replaced NumPy functions with MLX equivalents:
  * Array creation: asarray, zeros, ones → mx.*
  * Array manipulation: reshape, concatenate → mx.*
  * Math operations: dot, diag, triu, tril → mx.*
  * Type checking: iscomplexobj, flatnonzero → mx.*

### Fixed Critical Issues:
- **filename.py**: Specific fixes made

### New MLX Compatibility Layer:
- Added `_mlx_compat.py` for missing MLX functions
- Imported in `__init__.py` for module-wide availability

## Benefits
✅ Zero NumPy in compute paths
✅ Lazy execution preserved
✅ No device hops or buffer breaks
✅ Proper Metal acceleration
✅ Bit-exact reproducibility
```
