# Code Examples: Differences Between Branches

This document provides concrete code examples showing the differences between the `copilot/compare-branch-to-main` branch and the `main` branch.

---

## 1. Import Statement Changes

### Example 1: Basic Package Import

**Main branch:**
```python
# scipy/_lib/__init__.py
from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
```

**This branch:**
```python
# scipy_mlx/_lib/__init__.py
from scipy_mlx._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
```

**Change:** `scipy` → `scipy_mlx` in import path

---

### Example 2: Array API Imports

**Main branch:**
```python
# scipy/_lib/_array_api.py
import numpy as np

from scipy._lib.array_api_compat import (
    is_array_api_obj,
    # ...
)
from scipy._lib.array_api_compat.common._helpers import _compat_module_name
from scipy._lib.array_api_extra.testing import lazy_xp_function
from scipy._lib._array_api_override import (
    array_namespace, SCIPY_ARRAY_API, SCIPY_DEVICE
)
from scipy._lib._docscrape import FunctionDoc
from scipy._lib import array_api_extra as xpx
```

**This branch:**
```python
# scipy_mlx/_lib/_array_api.py
import mlx.core as mx

from scipy_mlx._lib.array_api_compat import (
    is_array_api_obj,
    # ...
)
from scipy_mlx._lib.array_api_compat.common._helpers import _compat_module_name
from scipy_mlx._lib.array_api_extra.testing import lazy_xp_function
from scipy_mlx._lib._array_api_override import (
    array_namespace, SCIPY_ARRAY_API, SCIPY_DEVICE
)
from scipy_mlx._lib._docscrape import FunctionDoc
from scipy_mlx._lib import array_api_extra as xpx
```

**Changes:** 
1. `scipy._lib` → `scipy_mlx._lib` in all imports
2. `import numpy as np` → `import mlx.core as mx` (fundamental change!)

---

### Example 3: Main Package Init

**Main branch:**
```python
# scipy/__init__.py
"""
SciPy: A scientific computing package for Python
"""

import numpy as np
# ... rest of imports
```

**This branch:**
```python
# scipy_mlx/__init__.py
"""
SciPy: A scientific computing package for Python
"""

import mlx.core as mx
# ... rest of imports
```

**Change:** The core array library changed from NumPy to MLX

---

## 2. Configuration File Changes

### pyproject.toml

**Main branch:**
```toml
[project]
name = "scipy"
version = "1.17.0.dev0"
# ...

[tool.spin]
package = 'scipy'
```

**This branch:**
```toml
[project]
name = "scipy-mlx"
version = "1.17.0.dev0"
# ...

[tool.spin]
package = 'scipy_mlx'
```

**Changes:**
- Package name: `scipy` → `scipy-mlx` (with hyphen for PyPI)
- Module name: `scipy` → `scipy_mlx` (with underscore for Python imports)

---

### pytest.ini

**Main branch:**
```ini
[pytest]
testpaths =
    scipy
```

**This branch:**
```ini
[pytest]
testpaths =
    scipy_mlx
```

**Change:** Test directory path updated

---

### meson.build

**Main branch:**
```python
project(
  'SciPy',
  'c', 'cpp', 'cython',
  # ...
)
py.install_sources(
  ['scipy/__init__.py'],
  # ...
)
```

**This branch:**
```python
project(
  'SciPy',
  'c', 'cpp', 'cython',
  # ...
)
py.install_sources(
  ['scipy_mlx/__init__.py'],
  # ...
)
```

**Change:** Install source paths updated to `scipy_mlx`

---

## 3. Directory Structure Changes

### Before (main branch):
```
mlx-scipy/
├── scipy/
│   ├── __init__.py
│   ├── _lib/
│   │   ├── __init__.py
│   │   ├── _array_api.py
│   │   └── ...
│   ├── linalg/
│   ├── special/
│   ├── stats/
│   └── ...
├── doc/
├── benchmarks/
└── pyproject.toml
```

### After (this branch):
```
mlx-scipy/
├── scipy_mlx/          ← Renamed!
│   ├── __init__.py
│   ├── _lib/
│   │   ├── __init__.py
│   │   ├── _array_api.py
│   │   └── ...
│   ├── linalg/
│   ├── special/
│   ├── stats/
│   └── ...
├── doc/
├── benchmarks/
└── pyproject.toml
```

---

## 4. Usage Examples

### Installing the Package

**Main branch:**
```bash
pip install scipy
# or
pip install -e .
```

**This branch:**
```bash
pip install scipy-mlx
# or
pip install -e .
```

---

### Using the Package in Code

**Main branch:**
```python
import scipy
import scipy.linalg as la
from scipy.special import gamma
from scipy import stats

# Use NumPy arrays
import numpy as np
x = np.array([1, 2, 3, 4, 5])
result = gamma(x)

# Linear algebra
A = np.array([[1, 2], [3, 4]])
inv_A = la.inv(A)

# Statistics
mean = stats.norm.mean()
```

**This branch:**
```python
import scipy_mlx
import scipy_mlx.linalg as la
from scipy_mlx.special import gamma
from scipy_mlx import stats

# Use MLX arrays
import mlx.core as mx
x = mx.array([1, 2, 3, 4, 5])
result = gamma(x)

# Linear algebra
A = mx.array([[1, 2], [3, 4]])
inv_A = la.inv(A)

# Statistics
mean = stats.norm.mean()
```

**Key Differences:**
1. Import name: `scipy` → `scipy_mlx`
2. Array library: `numpy` → `mlx.core`
3. Array creation: `np.array()` → `mx.array()`
4. All array operations use MLX (GPU-accelerated if available)

---

## 5. Test File Changes

### Example Test File

**Main branch:**
```python
# scipy/linalg/tests/test_basic.py
import numpy as np
from numpy.testing import assert_allclose
from scipy import linalg

def test_inv():
    A = np.array([[1, 2], [3, 4]])
    Ainv = linalg.inv(A)
    assert_allclose(A @ Ainv, np.eye(2))
```

**This branch:**
```python
# scipy_mlx/linalg/tests/test_basic.py
import mlx.core as mx
from scipy_mlx._lib._testutils import assert_allclose
from scipy_mlx import linalg

def test_inv():
    A = mx.array([[1, 2], [3, 4]])
    Ainv = linalg.inv(A)
    assert_allclose(A @ Ainv, mx.eye(2))
```

**Changes:**
1. `numpy` → `mlx.core`
2. `scipy` → `scipy_mlx`
3. `np.array()` → `mx.array()`
4. `np.eye()` → `mx.eye()`

---

## 6. Docstring Changes

**Main branch:**
```python
"""
scipy.linalg.inv
================

Compute the inverse of a matrix.

Examples
--------
>>> import numpy as np
>>> from scipy import linalg
>>> A = np.array([[1, 2], [3, 4]])
>>> Ainv = linalg.inv(A)
"""
```

**This branch:**
```python
"""
scipy_mlx.linalg.inv
====================

Compute the inverse of a matrix.

Examples
--------
>>> import mlx.core as mx
>>> from scipy_mlx import linalg
>>> A = mx.array([[1, 2], [3, 4]])
>>> Ainv = linalg.inv(A)
"""
```

---

## 7. Migration Path Example

### Migrating Your Code

If you have existing code using scipy:

```python
# Old code (works with main branch)
import numpy as np
from scipy import linalg, special, stats

x = np.array([1, 2, 3])
y = special.gamma(x)
A = np.random.randn(3, 3)
eigvals = linalg.eigvals(A)
```

Migrate to:

```python
# New code (works with this branch)
import mlx.core as mx
from scipy_mlx import linalg, special, stats

x = mx.array([1, 2, 3])
y = special.gamma(x)
A = mx.random.normal((3, 3))
eigvals = linalg.eigvals(A)
```

**Steps:**
1. Replace `numpy` imports with `mlx.core`
2. Replace `scipy` imports with `scipy_mlx`
3. Change `np.array()` to `mx.array()`
4. Update NumPy-specific functions to MLX equivalents

---

## Summary

The changes are **systematic and comprehensive**:

1. **Every occurrence** of `scipy` in import paths → `scipy_mlx`
2. **Core array library** changed from NumPy → MLX
3. **Directory structure** renamed: `scipy/` → `scipy_mlx/`
4. **Configuration files** updated to reflect new package name
5. **Documentation** updated throughout

The API remains the same, but the underlying implementation uses MLX instead of NumPy, enabling GPU acceleration and compatibility with Apple Silicon.
