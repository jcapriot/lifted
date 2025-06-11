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
endif

cython_args = []
if cy.version().version_compare('>=3.1.0')
  cython_args += ['-Xfreethreading_compatible=True']
endif

py.extension_module(
    '_wavelets',
    ['_wavelets.pyx'],
    cython_args: cython_args,
    cpp_args: cython_cpp_args,
    install: true,
    subdir: module_path,
    dependencies : [py_dep, np_dep, wavelets_dep],
    override_options : ['cython_language=cpp'],
)
