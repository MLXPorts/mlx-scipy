# Branch Differences: copilot/compare-branch-to-main vs main

## Executive Summary

This document details the differences between the `copilot/compare-branch-to-main` branch and the `main` branch of the mlx-scipy repository.

**Key Finding:** This branch contains **2 commits** that are not in main, representing a major package renaming effort.

---

## Commit History

### Commits on this branch (not in main):

1. **3e448c205** - "Rename package from scipy to scipy-mlx" (grafted)
2. **87fff0a5d** - "Initial plan" (HEAD)

### Latest commits on main branch:

- **b9e44dd3e** - Fix invalid input handling in cosine_invcdf
- **fdc815401** - Fix coefficient ordering in _polevl to match cephes behavior
- **5882bea36** - Fix dtype issues and boolean indexing in scipy.special
- *(and more...)*

---

## Overall Statistics

- **Files changed:** 2,260 files
- **Lines added:** 3,479 insertions(+)
- **Lines deleted:** 3,813 deletions(-)
- **Net change:** -334 lines

---

## Major Changes

### 1. Package Renaming (Commit 3e448c205)

The primary change in this branch is a comprehensive package renaming from `scipy` to `scipy-mlx`:

#### Package Names Changed:
- **PyPI package name:** `scipy` → `scipy-mlx`
- **Python import name:** `scipy` → `scipy_mlx`
- **Directory structure:** `scipy/` → `scipy_mlx/`

#### Purpose (from commit message):
> Package rename to enable side-by-side installation with standard scipy
> for numeric stability testing and clear separation of MLX implementation.

#### Rationale:
- Allows `scipy-mlx` to be installed alongside standard `scipy` without conflicts
- Provides clear separation of the MLX-based implementation
- Enables numeric stability testing by comparing outputs
- Establishes scipy-mlx as an independent project

---

## Detailed File Changes

### A. Configuration Files Modified

#### 1. `pyproject.toml`
```diff
- name = "scipy"
+ name = "scipy-mlx"

- package = 'scipy'
+ package = 'scipy_mlx'
```

#### 2. `meson.build`
- Updated package references from `scipy` to `scipy_mlx`
- Modified 2 lines

#### 3. `pytest.ini`
- Updated test paths and configuration
- Modified 8 lines to reference `scipy_mlx`

#### 4. `mypy.ini`
- Updated type checking configuration
- Modified 222 lines (many path references updated)

#### 5. `doc/source/conf.py`
- Updated documentation configuration
- Modified 5 lines to reflect new package name

---

### B. Directory Rename

The entire `scipy/` directory tree was renamed to `scipy_mlx/`:

```
scipy/ → scipy_mlx/
```

This includes all subdirectories:
- `scipy_mlx/__init__.py`
- `scipy_mlx/_lib/`
- `scipy_mlx/_build_utils/`
- `scipy_mlx/cluster/`
- `scipy_mlx/constants/`
- `scipy_mlx/datasets/`
- `scipy_mlx/differentiate/`
- `scipy_mlx/fft/`
- `scipy_mlx/fftpack/`
- `scipy_mlx/integrate/`
- `scipy_mlx/interpolate/`
- `scipy_mlx/io/`
- `scipy_mlx/linalg/`
- `scipy_mlx/ndimage/`
- `scipy_mlx/odr/`
- `scipy_mlx/optimize/`
- `scipy_mlx/signal/`
- `scipy_mlx/sparse/`
- `scipy_mlx/spatial/`
- `scipy_mlx/special/`
- `scipy_mlx/stats/`
- And all their subdirectories...

---

### C. Import Statements Updated

All internal imports were updated throughout the codebase:

```python
# Before:
import scipy
from scipy import linalg
from scipy.special import gamma

# After:
import scipy_mlx
from scipy_mlx import linalg
from scipy_mlx.special import gamma
```

This affected:
- All `.py` Python files
- All `.pyx` Cython implementation files
- All `.pxd` Cython definition files
- All `.pyi` stub files for type hints
- All test files

---

### D. Documentation Updates

Documentation was updated to reflect the new package name:
- `doc/source/conf.py` - Documentation configuration
- Various docstrings mentioning the package name

---

### E. Files Removed/Deleted

Several files were removed from the old `scipy/` structure:

1. **Submodule references:**
   - `scipy/_lib/array_api_compat` (submodule removed)
   - `scipy/_lib/array_api_extra` (submodule removed)
   - `scipy/_lib/cobyqa` (submodule removed)
   - `scipy/_lib/pocketfft` (submodule removed)
   - `scipy/_lib/unuran` (submodule removed)

2. **Patch files:**
   - `scipy/_lib/_uarray/scipychanges.patch` (13 lines deleted)
   - `scipy/sparse/linalg/_dsolve/SuperLU/scipychanges.patch` (439 lines deleted)

3. **Cython definition files:**
   - `scipy/linalg/__init__.pxd` (1 line deleted)
   - `scipy/special/__init__.pxd` (1 line deleted)

---

### F. Key Code Changes

#### `scipy_mlx/__init__.py`

Key changes in the main package initialization:

```python
# New MLX import added:
import mlx.core as mx
```

The file now imports MLX as the primary array library instead of NumPy, reflecting the core purpose of this fork.

Import statements throughout updated from:
```python
from scipy._lib import ...
```
to:
```python
from scipy_mlx._lib import ...
```

---

## Branch Structure

The current branch has a **grafted history**, meaning:

```
* 87fff0a (HEAD) Initial plan
* 3e448c205 (grafted) Rename package from scipy to scipy-mlx
```

The "grafted" designation on commit 3e448c205 indicates this commit was created without its full history - it represents a snapshot of the rename work.

The branch diverged from main at commit **b9e44dd3e** ("Fix invalid input handling in cosine_invcdf").

---

## What This Branch Does NOT Include

This branch does **not** include the following commits that exist in main:

- b9e44dd3e - Fix invalid input handling in cosine_invcdf
- fdc815401 - Fix coefficient ordering in _polevl to match cephes behavior
- 5882bea36 - Fix dtype issues and boolean indexing in scipy.special
- (and other recent main branch improvements)

These would need to be merged if you want the latest fixes from main.

---

## Impact Analysis

### Compatibility Impact

1. **Installation:** Users can now install both packages side-by-side:
   ```bash
   pip install scipy        # Original NumPy-based scipy
   pip install scipy-mlx    # MLX-based scipy
   ```

2. **Import Statements:** All code using this library must update imports:
   ```python
   import scipy_mlx  # instead of scipy
   ```

3. **API Compatibility:** The API should remain functionally equivalent, but:
   - All functions now expect/return MLX arrays instead of NumPy arrays
   - Performance characteristics may differ (GPU acceleration via MLX)

### Development Impact

1. **Testing:** Test infrastructure updated to use `scipy_mlx` paths
2. **Documentation:** Docs now reference `scipy_mlx` 
3. **Type Checking:** mypy configuration updated for new paths
4. **Build System:** Meson build updated for new package name

---

## Migration Path

If you want to use this branch:

1. **Uninstall old scipy (if conflicting):** `pip uninstall scipy`
2. **Install from this branch:** Build and install the `scipy-mlx` package
3. **Update all import statements** in your code from `scipy` to `scipy_mlx`
4. **Update dependencies** in requirements files from `scipy` to `scipy-mlx`
5. **Ensure MLX is installed:** This package depends on `mlx.core`

---

## Summary

This branch represents a **complete package rename** from `scipy` to `scipy-mlx`, affecting:
- Package distribution name
- Python import namespace  
- Directory structure
- All internal references
- Build and test configuration

The changes enable side-by-side installation with standard scipy and clearly identify this as an MLX-accelerated variant of the library.

**Recommendation:** If you need the latest bug fixes from main, you should merge or rebase this branch onto the latest main branch.
