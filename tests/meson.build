
gtest_dep = gtest_proj.get_variable('gtest_dep')
gtest_main_dep = gtest_proj.get_variable('gtest_main_dep')


test_inc = include_directories('.')


test_single_levels = executable(
	'test_single_levels', 
	'test_single_levels.cpp',
	dependencies : [gtest_dep, gtest_main_dep, wavelets_dep],
	include_directories : test_inc,
)

test('Single level forward, inverse, and adjoint validations', test_single_levels)


test_dwt = executable(
	'test_dwt', 
	'test_dwt.cpp',
	dependencies : [gtest_dep, gtest_main_dep, wavelets_dep],
	include_directories : test_inc,
)

test('Driver validations', test_dwt)