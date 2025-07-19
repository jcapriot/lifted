project(
	'pocket_wavelets',
	'cpp',
	version: '0.0.1_alpha',
	default_options :[
		'buildtype=release',
		'b_ndebug=if-release',
		'cpp_std=c++20',
	],
	meson_version: '>=1.2',
)

cpp = meson.get_compiler('cpp')
cpp_id = cpp.get_id()

if 'msvc' in cpp_id
	add_project_arguments('/bigobj', language: 'cpp')
endif

hwy = subproject('highway')
hwy_dep = hwy.get_variable('hwy_dep')

not_static_build = get_option('default_library') != 'static'

export_defines = []
if not_static_build
	export_defines += '-DLIFTED_SHARED_DEFINE'
endif

subdir('src')

if get_option('build_tests')
	gtest_proj = subproject('gtest')
	subdir('tests')
endif

if get_option('build_benchmarks')
	gtest_proj = subproject('google-benchmark', default_options:{'default-library':'static'})
	subdir('benchmarks')
endif