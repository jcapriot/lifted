project(
	'pocket_wavelets',
	'cpp',
	version: '0.0.1_alpha',
	default_options :[
		'buildtype=release',
		'b_ndebug=if-release',
		'cpp_std=c++20',
	]
)

cpp = meson.get_compiler('cpp')

cpp_id = cpp.get_id()
if 'msvc' in cpp_id
	add_project_arguments('/arch:AVX2', language:'cpp')
endif

subdir('src')

if get_option('build_tests')
	gtest_proj = subproject('gtest')
	subdir('tests')
endif