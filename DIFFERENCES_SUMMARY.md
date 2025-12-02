# Quick Summary: Branch Differences

## TL;DR

**This branch (`copilot/compare-branch-to-main`) differs from `main` by 2 commits that rename the entire package from `scipy` to `scipy-mlx`.**

---

## Quick Stats

- **Commits ahead:** 2
- **Commits behind:** 0  
- **Files changed:** 2,260
- **Net lines changed:** -334 lines (3,479 added, 3,813 deleted)

---

## What Changed?

### Primary Change: Package Renamed

| Aspect | Before (main) | After (this branch) |
|--------|--------------|---------------------|
| **Package name** | `scipy` | `scipy-mlx` |
| **Import statement** | `import scipy` | `import scipy_mlx` |
| **Directory** | `scipy/` | `scipy_mlx/` |
| **PyPI install** | `pip install scipy` | `pip install scipy-mlx` |

---

## Commits on This Branch

1. **3e448c205** (grafted) - "Rename package from scipy to scipy-mlx"
   - Complete package rename
   - Directory rename: `scipy/` → `scipy_mlx/`
   - Updated ~4,500 files with new import paths
   - Modified configuration files (pyproject.toml, meson.build, pytest.ini, mypy.ini)
   - Removed submodule references
   - Added MLX imports

2. **87fff0a5d** (HEAD) - "Initial plan"
   - Empty commit for tracking

---

## Files Modified (Key Ones)

**Configuration:**
- `pyproject.toml` - Package name changed
- `meson.build` - Build config updated
- `pytest.ini` - Test paths updated  
- `mypy.ini` - Type checking paths updated
- `doc/source/conf.py` - Docs config updated

**Code:**
- All files in `scipy/` → moved to `scipy_mlx/`
- All imports updated: `scipy` → `scipy_mlx`
- New MLX import added: `import mlx.core as mx`

---

## Why This Change?

**Purpose:** Enable side-by-side installation with standard scipy

**Benefits:**
- ✅ Both packages can coexist on the same system
- ✅ Clear separation of MLX vs NumPy implementations
- ✅ Enables numeric stability testing and comparison
- ✅ Establishes scipy-mlx as independent project

---

## What's Missing from Main?

This branch does **NOT** include the latest fixes from main:

- Fix invalid input handling in cosine_invcdf (b9e44dd3e)
- Fix coefficient ordering in _polevl (fdc815401)
- Fix dtype issues and boolean indexing (5882bea36)
- Various other recent improvements

**Action:** Consider merging/rebasing with latest main to get these fixes.

---

## Usage Example

**Before (main branch):**
```python
import scipy
from scipy import linalg
from scipy.special import gamma

result = gamma(5)
```

**After (this branch):**
```python
import scipy_mlx
from scipy_mlx import linalg
from scipy_mlx.special import gamma

result = gamma(5)  # Now uses MLX arrays
```

---

## Next Steps

1. **Review** the detailed analysis in `BRANCH_DIFFERENCES.md`
2. **Decide** if you want to:
   - Keep this branch as-is for the rename
   - Merge latest changes from main
   - Make additional modifications
3. **Test** the renamed package works correctly
4. **Update** any dependent code to use new import names

---

## Visual Comparison

```
main branch:
├── scipy/
│   ├── __init__.py
│   ├── linalg/
│   ├── special/
│   └── ...

This branch:
├── scipy_mlx/          ← Renamed!
│   ├── __init__.py     ← MLX imports added
│   ├── linalg/
│   ├── special/
│   └── ...
```

---

## Questions?

- Full details: See `BRANCH_DIFFERENCES.md`
- Git commands used:
  ```bash
  git log --oneline origin/main..HEAD
  git diff --stat origin/main HEAD
  ```
