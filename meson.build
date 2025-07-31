project(
	'pocket_wavelets',
	'cpp',
	version: '0.0.1_alpha',
	default_options :[
		'buildtype=release',
		'b_ndebug=if-release',
		'cpp_std=c++20',
		'cpp_rtti=false',
	],
	meson_version: '>=1.3',
)

cpp = meson.get_compiler('cpp')
cpp_id = cpp.get_id()

thread_dep = dependency('threads')
if cpp.has_header('pthread.h')
	add_project_arguments('-DLIFTED_PTHREADS', language:'cpp')
endif

if cpp.get_argument_syntax() == 'msvc'
	add_project_arguments(
		'/bigobj',
        # Dissable run-time type information.
        '/GR-',
		language: 'cpp'
	)
endif

hwy = subproject(
	'highway', 
	default_options: [
		'default_library=static',
		'contrib=disabled',
		'test_standalone=false',
	]
)
hwy_dep = hwy.get_variable('hwy_dep')

not_static_build = get_option('default_library') != 'static'

export_defines = []
if not_static_build
	export_defines += '-DLIFTED_SHARED_DEFINE'
endif

do_py_install = get_option('python_install')

subdir('src')

enable_tests = get_option('tests').disable_auto_if(do_py_install).allowed()
enable_bench = get_option('benchmarks').disable_auto_if(do_py_install).allowed()

if enable_tests
	subdir('tests')
endif

if enable_bench
	gtest_proj = subproject('google-benchmark', default_options:{'default-library':'static'})
	subdir('benchmarks')
endif