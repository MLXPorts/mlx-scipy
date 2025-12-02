# Branch Comparison Analysis

This repository contains a comprehensive analysis of the differences between the `copilot/compare-branch-to-main` branch and the `main` branch of the mlx-scipy project.

## 📚 Documentation Files

This analysis consists of three complementary documents:

### 1. 📋 [DIFFERENCES_SUMMARY.md](DIFFERENCES_SUMMARY.md) - Quick Overview
**Read this first for a quick understanding**

- TL;DR summary
- Quick statistics
- Visual comparison diagrams
- High-level overview of changes

**Time to read:** 2-3 minutes

---

### 2. 📖 [BRANCH_DIFFERENCES.md](BRANCH_DIFFERENCES.md) - Detailed Analysis
**Read this for comprehensive understanding**

- Complete commit history
- Detailed file-by-file changes
- Impact analysis
- Migration guidance
- Technical details

**Time to read:** 10-15 minutes

---

### 3. 💻 [CODE_EXAMPLES.md](CODE_EXAMPLES.md) - Concrete Examples
**Read this to see actual code changes**

- Before/after code comparisons
- Import statement changes
- Configuration file examples
- Usage examples
- Migration code samples

**Time to read:** 5-10 minutes

---

## 🎯 Quick Answer

**What's different?**

This branch renames the entire package from `scipy` to `scipy-mlx`:

```
scipy → scipy-mlx (package name)
scipy → scipy_mlx (import name)
scipy/ → scipy_mlx/ (directory)
numpy → mlx (core array library)
```

**Why?**

To enable side-by-side installation with standard scipy and provide a clear MLX-accelerated variant.

**Impact:**

- 2,260 files changed
- 2 commits ahead of main
- Complete package namespace change
- All imports must be updated

---

## 📊 Key Statistics

| Metric | Value |
|--------|-------|
| **Commits ahead** | 2 |
| **Commits behind** | 0 |
| **Files changed** | 2,260 |
| **Lines added** | 3,479 |
| **Lines deleted** | 3,813 |
| **Net change** | -334 lines |

---

## 🔍 Quick Comparison

### Installation

```bash
# Main branch
pip install scipy

# This branch
pip install scipy-mlx
```

### Usage

```python
# Main branch
import scipy
from scipy import linalg

# This branch
import scipy_mlx
from scipy_mlx import linalg
```

### Core Array Library

```python
# Main branch
import numpy as np
x = np.array([1, 2, 3])

# This branch
import mlx.core as mx
x = mx.array([1, 2, 3])
```

---

## 🚀 Getting Started

### If you want a quick summary:
→ Read [DIFFERENCES_SUMMARY.md](DIFFERENCES_SUMMARY.md)

### If you need complete details:
→ Read [BRANCH_DIFFERENCES.md](BRANCH_DIFFERENCES.md)

### If you want to see code examples:
→ Read [CODE_EXAMPLES.md](CODE_EXAMPLES.md)

### If you want all three:
1. Start with DIFFERENCES_SUMMARY.md (quick overview)
2. Move to CODE_EXAMPLES.md (see concrete changes)
3. Finish with BRANCH_DIFFERENCES.md (complete details)

---

## 🔧 How This Analysis Was Generated

This analysis was created by comparing the git history and file differences between branches:

```bash
# Fetch main branch
git fetch origin main

# Compare commits
git log --oneline origin/main..HEAD

# Compare files
git diff --stat origin/main HEAD
git diff --numstat origin/main HEAD

# View specific commits
git show <commit-hash>
```

---

## ⚠️ Important Notes

1. **This branch is ahead of main** by 2 commits but is based on an older version of main
2. **Latest fixes from main are missing** (cosine_invcdf fix, _polevl fix, dtype fixes, etc.)
3. **The main commit (3e448c205) is grafted**, meaning it has incomplete history
4. **All imports must be updated** if switching to this branch
5. **MLX must be installed** as it replaces NumPy as the core array library

---

## 📝 Summary of Changes

### Primary Change: Package Rename

The entire package was renamed from `scipy` to `scipy-mlx`:

- **Package distribution name:** scipy → scipy-mlx
- **Python module name:** scipy → scipy_mlx  
- **Directory structure:** scipy/ → scipy_mlx/
- **All internal imports:** Updated throughout codebase
- **Core array library:** NumPy → MLX

### Purpose

Enable side-by-side installation with standard scipy for:
- Numeric stability testing
- GPU acceleration comparison
- Independent MLX-based implementation

### Files Affected

- ~2,260 files total
- All Python (.py) files
- All Cython (.pyx, .pxd) files
- All stub (.pyi) files
- Configuration files (pyproject.toml, meson.build, pytest.ini, mypy.ini)
- Documentation files

---

## 🎓 Understanding the Documents

### Document Relationships

```
DIFFERENCES_SUMMARY.md (Quick overview)
           ↓
    CODE_EXAMPLES.md (See the changes)
           ↓
BRANCH_DIFFERENCES.md (Full details)
```

### Which Document to Read?

**"I just need the basics"**
→ DIFFERENCES_SUMMARY.md

**"Show me the actual code changes"**
→ CODE_EXAMPLES.md

**"I need to understand everything"**
→ BRANCH_DIFFERENCES.md

**"I'm migrating code"**
→ CODE_EXAMPLES.md + BRANCH_DIFFERENCES.md

**"I'm making decisions about the project"**
→ All three documents

---

## 📞 Questions?

If you need clarification on any aspect of the differences:

1. Check the appropriate documentation file above
2. Review the git history: `git log --oneline --graph origin/main..HEAD`
3. Examine specific files: `git diff origin/main HEAD -- <filepath>`
4. Look at commit details: `git show <commit-hash>`

---

## ✅ Verification Commands

To verify this analysis yourself:

```bash
# View commits on this branch not in main
git log --oneline origin/main..HEAD

# View overall statistics
git diff --shortstat origin/main HEAD

# View changed files
git diff --stat origin/main HEAD

# View numeric statistics
git diff --numstat origin/main HEAD

# View specific file changes
git diff origin/main HEAD -- <filepath>
```

---

## 📅 Analysis Date

This analysis was generated on: **December 2, 2025**

Based on:
- Branch: `copilot/compare-branch-to-main`
- Compared to: `origin/main`
- Repository: `MLXPorts/mlx-scipy`

---

## 🎯 Bottom Line

**This branch contains a comprehensive package rename from scipy to scipy-mlx, changing ~2,260 files to enable side-by-side installation with standard scipy and establish scipy-mlx as an independent MLX-accelerated implementation.**

For the complete story, start with [DIFFERENCES_SUMMARY.md](DIFFERENCES_SUMMARY.md). 🚀
