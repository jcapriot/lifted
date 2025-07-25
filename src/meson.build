
lifted_inc = include_directories('.')

lifted_hdr_sources = files(
	'lifted.hpp',
	'lifted-common.hpp',
	'lifted-wavelets.hpp',
	'lifted-export.hpp',
	'lifted-inl.hpp',
	'lifted-drivers-inl.hpp',
	'lifted-ops-inl.hpp',
	'lifted-steps-inl.hpp',
)

lifted_hdr_dep = declare_dependency(
	include_directories: lifted_inc
)

lifted_sources = files(
	'lifted-double.cpp',
	'lifted-float.cpp',
	'lifted-sep-float.cpp',
	'lifted-sep-double.cpp',
)

lifted_lib = library(
	'lifted_lib',
	lifted_sources,
	dependencies: [hwy_dep, thread_dep],
	cpp_shared_args: ['-DLIFTED_SHARED_DEFINE', '-DLIFTED_EXPORTS'],
	install: true
)

complie_with_args = []
if not_static_build
	complie_with_args += ['-DLIFTED_SHARED_DEFINE']
endif

lifted_lib_dep = declare_dependency(
	include_directories: lifted_inc,
	link_with: lifted_lib,
	compile_args: complie_with_args
)

if get_option('python_install')
    subdir('pylifted')
endif