# 🔍 Branch Comparison Analysis

**Quick Answer:** This branch differs from `main` by implementing a comprehensive package rename from `scipy` to `scipy-mlx`, affecting 2,260 files.

---

## 📖 Documentation Files

All differences have been thoroughly analyzed and documented. Choose your starting point:

### 🚀 Quick Start (2-3 minutes)
**[DIFFERENCES_SUMMARY.md](DIFFERENCES_SUMMARY.md)**
- TL;DR of all changes
- Quick statistics
- Visual comparisons
- High-level overview

### 💡 Entry Point & Navigation
**[BRANCH_ANALYSIS.md](BRANCH_ANALYSIS.md)**
- Complete navigation guide
- Document relationships
- How to use this analysis
- Quick reference tables

### 💻 See the Code Changes (5-10 minutes)
**[CODE_EXAMPLES.md](CODE_EXAMPLES.md)**
- Before/after code comparisons
- Import statement changes
- Configuration examples
- Migration guide with code

### 📚 Complete Details (10-15 minutes)
**[BRANCH_DIFFERENCES.md](BRANCH_DIFFERENCES.md)**
- Full commit history
- Detailed file-by-file analysis
- Impact analysis
- Technical specifications

---

## 🎯 Key Finding

This branch contains **2 commits ahead of main** that rename the entire package:

```
scipy → scipy-mlx (package name)
scipy → scipy_mlx (import name)
scipy/ → scipy_mlx/ (directory)
numpy → mlx (core array library)
```

**Files Changed:** 2,260  
**Purpose:** Enable side-by-side installation with standard scipy

---

## 📊 Quick Statistics

| Metric | Value |
|--------|-------|
| Commits ahead | 2 |
| Commits behind | 0 |
| Files changed | 2,260 |
| Lines added | 3,479 |
| Lines deleted | 3,813 |
| Net change | -334 |

---

## 🎓 Recommended Reading Order

**If you want...**

- **Quick overview** → [DIFFERENCES_SUMMARY.md](DIFFERENCES_SUMMARY.md)
- **Navigation help** → [BRANCH_ANALYSIS.md](BRANCH_ANALYSIS.md)
- **Code examples** → [CODE_EXAMPLES.md](CODE_EXAMPLES.md)
- **Complete details** → [BRANCH_DIFFERENCES.md](BRANCH_DIFFERENCES.md)
- **Everything** → Read in order: DIFFERENCES_SUMMARY → CODE_EXAMPLES → BRANCH_DIFFERENCES

---

## 🔄 Before & After

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

---

## ⚠️ Important Notes

- ⚠️ This branch is missing latest fixes from main
- ⚠️ Commit 3e448c205 is grafted (incomplete history)
- ✅ All imports systematically updated
- ✅ Configuration files updated
- ✅ Documentation updated

---

## �� Summary

This branch implements a **complete package rename** to enable:
- Side-by-side installation with standard scipy
- Numeric stability testing and comparison
- Clear separation of MLX implementation
- Independent MLX-accelerated scipy package

**Start with [BRANCH_ANALYSIS.md](BRANCH_ANALYSIS.md) for full navigation guidance.**

---

Generated: December 2, 2025
