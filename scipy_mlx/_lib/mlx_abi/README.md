# MLX C++ ABI header API vendored for local bindings

This directory vendors the MLX C++ API headers copied from:

- Source repository: `/Volumes/stuff/Projects/mlx-precise`
- Source path: `python/mlx/include/`
- Imported on: 2026-02-16

Contents:
- `include/mlx/*` — MLX C++ headers
- `include/metal_cpp/*` — Metal C++ interoperability headers

This is header-only vendored API surface intended for compiling native extensions that
need to bind C/C++ code directly against MLX.
