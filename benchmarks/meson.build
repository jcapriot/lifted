
gbench_dep = gtest_proj.get_variable('google_benchmark_dep')
gbench_main_dep = gtest_proj.get_variable('google_benchmark_main_dep')

extra_cpp_args = []
autovec_args = []
if 'intel' in cpp_id
	extra_cpp_args += ['-Qopt-report=max']
	autovec_args += ['-march=native', '-fp-model=fast']
elif 'clang' in cpp_id
	extra_cpp_args += ['-Xclang', '-Rpass=loop-unroll', '-Xclang', '-Rpass-missed=loop-unroll']
	autovec_args += ['-march=native']
elif 'msvc' in cpp_id
	autovec_args += ['/mAVX2']
endif

autovec_bench = executable(
	'wavelet_autovec', 
	'wavelet_autovec.cpp',
	dependencies : [gbench_dep, gbench_main_dep, wavelets_dep],
    cpp_args : extra_cpp_args + autovec_args
)

highway_bench = executable(
	'wavelet_highway', 
	'wavelet_highway.cpp',
	dependencies : [gbench_dep, gbench_main_dep, wavelets_dep, hwy_dep],
    cpp_args: extra_cpp_args
)

executable(
	'highway_tests', 
	'highway_tests.cpp',
	dependencies : [gbench_dep, gbench_main_dep, hwy_dep],
    cpp_args: extra_cpp_args
)

benchmark('wavelet_autovec', autovec_bench)
benchmark('wavelet_highway', highway_bench)