add_languages('cython')

cy = meson.get_compiler('cython')
py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

module_path = 'pylifted'

py.install_sources(
    '__init__.py',
    subdir: module_path
)

# NumPy include directory
numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_22_API_VERSION']
np_dep = dependency('numpy')

cython_cpp_args = [numpy_nodepr_api]

if 'msvc' in cpp_id
	cython_cpp_args = [cython_cpp_args, '/bigobj', '/arch:AVX2']

elif 'intel' in cpp_id
	cython_cpp_args = [cython_cpp_args, '-march=native']

endif

cython_args = []
if cy.version().version_compare('>=3.1.0')
  cython_args += ['-Xfreethreading_compatible=True']
endif

instantiation_sources = [
    'for.cpp',
    'for_adj.cpp',
    'inv.cpp',
    'inv_adj.cpp',
    'for_sep.cpp',
    'for_sep_adj.cpp',
    'inv_sep.cpp',
    'inv_sep_adj.cpp'
]

inst_inc = include_directories('.')

py.extension_module(
    '_wavelets',
    ['_wavelets.pyx'] + instantiation_sources,
    cython_args: cython_args,
    cpp_args: cython_cpp_args,
    install: true,
    subdir: module_path,
    dependencies : [py_dep, np_dep, wavelets_dep],
    include_directories: inst_inc,
    override_options : ['cython_language=cpp'],
)
