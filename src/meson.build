
wvlt_inc = include_directories('.')

wvlt_hdr_sources = files(
	'ndarray.hpp',
	'wavelets.hpp'
)

wavelets_dep = declare_dependency(
	include_directories: wvlt_inc
)

lifted_sources = files(
	'lifted-double.cpp',
	'lifted-float.cpp',
	'lifted-sep-float.cpp',
	'lifted-sep-double.cpp',
)

lib_defines = []
if not_static_build
	lib_defines += ['-DLIFTED_SHARED_DEFINE', '-DLIFTED_EXPORTS']
endif

lifted_lib = library(
	'lifted_lib',
	lifted_sources,
	dependencies: hwy_dep,
	cpp_args: lib_defines,
)

complie_with_args = []
if not_static_build
	complie_with_args += ['-DLIFTED_SHARED_DEFINE']
endif

lifted_lib_dep = declare_dependency(
	include_directories: wvlt_inc,
	link_with: lifted_lib,
	compile_args: complie_with_args
)

if get_option('python_install')
    subdir('pylifted')
endif