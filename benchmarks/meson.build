
gbench_dep = gtest_proj.get_variable('google_benchmark_dep')
gbench_main_dep = gtest_proj.get_variable('google_benchmark_main_dep')

extra_cpp_args = []
if 'intel' in cpp_id
	extra_cpp_args += ['-Qopt-report=max']
elif 'clang' in cpp_id
	extra_cpp_args += ['-Xclang', '-Rpass=loop-unroll', '-Xclang', '-Rpass-missed=loop-unroll']
endif

highway_bench = executable(
	'wavelet_highway', 
	'wavelet_highway.cpp',
	dependencies : [gbench_dep, gbench_main_dep, lifted_hdr_dep, hwy_dep],
    cpp_args: extra_cpp_args,
)

benchmark('wavelet_highway', highway_bench)