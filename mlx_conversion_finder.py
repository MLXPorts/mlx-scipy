#!/usr/bin/env python
"""
mlx_conversion_finder.py - Find Python files that still import NumPy and suggest MLX replacements

This tool helps identify which files in scipy_mlx still need NumPy-to-MLX conversion
and provides suggested replacements.
"""

import os
import re
import sys
import importlib.util


# Mapping of common NumPy functions to MLX equivalents
NUMPY_TO_MLX = {
    # Math functions
    'abs': 'abs',
    'absolute': 'abs',
    'sqrt': 'sqrt',
    'sin': 'sin',
    'cos': 'cos',
    'tan': 'tan',
    'sinh': 'sinh',
    'cosh': 'cosh',
    'tanh': 'tanh',
    'exp': 'exp',
    'log': 'log',
    'log10': 'log10',
    'power': 'power',
    'exp2': 'exp2',
    'expm1': 'expm1',
    'log1p': 'log1p',
    'log2': 'log2',
    'pi': 'pi',
    'e': 'e',
    'inf': 'inf',
    'nan': 'nan',
    
    # Array creation
    'array': 'array',
    'asarray': 'array',
    'zeros': 'zeros',
    'ones': 'ones',
    'eye': 'eye',
    'identity': 'identity',
    'full': 'full',
    'empty': 'zeros',  # MLX doesn't have empty
    'arange': 'arange',
    'linspace': 'linspace',
    'meshgrid': 'meshgrid',
    
    # Array manipulation
    'reshape': 'reshape',
    'concatenate': 'concatenate',
    'stack': 'stack',
    'vstack': 'stack',  # Use stack with proper axis
    'hstack': 'concatenate',  # MLX doesn't have hstack
    'squeeze': 'squeeze',
    'transpose': 'transpose',
    'swapaxes': 'swapaxes',
    'moveaxis': 'moveaxis',
    'flatten': 'flatten',
    'ravel': 'reshape',  # Use reshape(-1)
    'expand_dims': 'expand_dims',
    'atleast_1d': 'atleast_1d',
    'atleast_2d': 'atleast_2d',
    'atleast_3d': 'atleast_3d',
    
    # Indexing/slicing
    'argmax': 'argmax',
    'argmin': 'argmin',
    'argsort': 'argsort',
    'where': 'where',
    'take': 'take',
    'diag': 'diag',
    'diagonal': 'diagonal',
    'trace': 'trace',
    'tri': 'tri',
    'tril': 'tril',
    'triu': 'triu',
    'pad': 'pad',
    
    # Math operations
    'dot': 'matmul',  # Or mx.multiply for element-wise
    'matmul': 'matmul',
    'inner': 'inner',
    'outer': 'outer',
    'kron': 'kron',
    'tensordot': 'tensordot',
    'einsum': 'einsum',
    'sum': 'sum',
    'prod': 'prod',
    'mean': 'mean',
    'std': 'std',
    'var': 'var',
    'max': 'max',
    'min': 'min',
    'amax': 'max',
    'amin': 'min',
    'maximum': 'maximum',
    'minimum': 'minimum',
    
    # Comparison
    'allclose': 'allclose',
    'array_equal': 'array_equal',
    'isclose': 'isclose',
    'isfinite': 'isfinite',
    'isinf': 'isinf',
    'isnan': 'isnan',
    'greater': 'greater',
    'greater_equal': 'greater_equal',
    'less': 'less',
    'less_equal': 'less_equal',
    'equal': 'equal',
    'not_equal': 'not_equal',
    'logical_and': 'logical_and',
    'logical_or': 'logical_or',
    'logical_not': 'logical_not',
    'logical_xor': 'logical_xor',
    
    # Rounding
    'round': 'round',
    'floor': 'floor',
    'ceil': 'ceil',
    
    # Type info
    'dtype': 'Dtype',  # MLX uses Dtype class
    'finfo': 'finfo',
    'iinfo': 'iinfo',
    'issubdtype': 'issubdtype',
    
    # Shape/Dtype
    'shape': 'shape attribute (.shape)',
    'ndim': 'ndim attribute (.ndim)',
    'size': 'size attribute (.size)',
    'astype': 'astype method',
}


def find_files_with_numpy(directory):
    """Find all Python files that import numpy with detailed information."""
    results = []
    
    for root, dirs, files in os.walk(directory):
        # Skip test files for now
        dirs[:] = [d for d in dirs if not d.startswith('test')]
        
        for file in files:
            if file.endswith('.py') and not file.startswith('test'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Find numpy imports with line numbers
                    imports_found = []
                    
                    for i, line in enumerate(content.split('\n'), 1):
                        # Match: import numpy as np
                        if re.match(r'^\s*import\s+numpy\s+as\s+(\w+)', line):
                            imports_found.append((i, line.strip()))
                        # Match: import numpy
                        elif re.match(r'^\s*import\s+numpy\b', line):
                            imports_found.append((i, line.strip()))
                        # Match: from numpy import ...
                        elif re.match(r'^\s*from\s+numpy\s+import\s+', line):
                            imports_found.append((i, line.strip()))
                    
                    if imports_found:
                        results.append({
                            'filepath': filepath,
                            'imports': imports_found
                        })
                        
                except Exception:
                    continue
    
    return results


def suggest_replacement(import_line):
    """Suggest MLX replacement for a NumPy import line."""
    line = import_line.strip()
    
    # Case 1: import numpy as np
    match = re.match(r'import\s+numpy\s+as\s+(\w+)', line)
    if match:
        alias = match.group(1)
        return f"import mlx.core as {alias}  # WARNING: Need manual review of all {alias}. usages"
    
    # Case 2: import numpy
    if re.match(r'import\s+numpy\b', line):
        return "import mlx.core as mx  # WARNING: Need manual replacement of all numpy. usages"
    
    # Case 3: from numpy import ...
    match = re.match(r'from\s+numpy\s+import\s+(.*)', line)
    if match:
        imports = match.group(1)
        imported_items = [item.strip() for item in imports.split(',')]
        
        # Check for common replacements
        mlx_items = []
        unknown_items = []
        
        for item in imported_items:
            # Handle 'as' aliases: e.g., absolute as abs
            parts = [p.strip() for p in item.split(' as ')]
            base_item = parts[0]
            alias = parts[1] if len(parts) > 1 else None
            
            if base_item in NUMPY_TO_MLX:
                mlx_name = NUMPY_TO_MLX[base_item]
                if alias:
                    mlx_items.append(f"{mlx_name} as {alias}")
                else:
                    mlx_items.append(mlx_name)
            else:
                unknown_items.append(item)
        
        if mlx_items and not unknown_items:
            return f"from mlx.core import {', '.join(mlx_items)}"
        elif mlx_items and unknown_items:
            return f"from mlx.core import {', '.join(mlx_items)}  # UNKNOWN: {', '.join(unknown_items)}"
        else:
            return f"# MANUAL REVIEW NEEDED: {line}"
    
    return f"# UNRECOGNIZED: {line}"


def main():
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = 'scipy_mlx'
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)
    
    print(f"Searching for NumPy imports in {directory}...")
    numpy_files = find_files_with_numpy(directory)
    
    if not numpy_files:
        print("\n✅ No files with NumPy imports found!")
        return
    
    print(f"\nFound {len(numpy_files)} files with NumPy imports:\n")
    
    files_with_replacements = 0
    files_needing_review = 0
    
    for result in sorted(numpy_files, key=lambda x: x['filepath']):
        filepath = result['filepath']
        imports = result['imports']
        
        # Show relative path
        rel_path = os.path.relpath(filepath, start=directory)
        print(f"\n📄 scipy_mlx/{rel_path}")
        
        has_auto_replacement = False
        has_manual_review = False
        
        for line_num, line in imports:
            suggestion = suggest_replacement(line)
            print(f"   Line {line_num}: {line}")
            print(f"   → {suggestion}")
            
            if "from mlx.core import" in suggestion and "UNKNOWN" not in suggestion:
                has_auto_replacement = True
            elif "MANUAL REVIEW" in suggestion or "WARNING" in suggestion or "UNKNOWN" in suggestion:
                has_manual_review = True
        
        if has_auto_replacement:
            files_with_replacements += 1
        if has_manual_review:
            files_needing_review += 1
    
    print(f"\n{'='*70}")
    print(f"Summary: {len(numpy_files)} files need conversion")
    print(f"  🟢 {files_with_replacements} files have auto-suggestable replacements")
    print(f"  🟡 {files_needing_review} files need manual review")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
