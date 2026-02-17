py3.extension_module('_hierarchy',
  cython_gen_cpp.process('_hierarchy.pyx'),
  cpp_args: cython_cpp_args,
  dependencies: np_mlx_dep,
  link_args: version_link_args,
  install: true,
  subdir: 'scipy/cluster'
)

py3.extension_module('_optimal_leaf_ordering',
  cython_gen_cpp.process('_optimal_leaf_ordering.pyx'),
  cpp_args: cython_cpp_args,
  dependencies: np_mlx_dep,
  link_args: version_link_args,
  install: true,
  subdir: 'scipy/cluster'
)

py3.extension_module('_vq',
  cython_gen_cpp.process('_vq.pyx'),
  cpp_args: cython_cpp_args,
  dependencies: np_mlx_dep,
  link_args: version_link_args,
  install: true,
  subdir: 'scipy/cluster'
)


py3.install_sources([
    '__init__.py',
    'hierarchy.py',
    'vq.py'
  ],
  subdir: 'scipy/cluster'
)

subdir('tests')
