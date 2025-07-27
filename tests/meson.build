
gtest_main_dep = dependency('gtest', main:true, fallback:['gtest', 'gtest_main_dep'])

hwy_test_dep = hwy.get_variable('hwy_test_dep')


test_inc = include_directories('.')


test_single_levels = executable(
	'test_single_levels', 
	'test_single_levels.cpp',
	cpp_args:['-DHWY_TEST_STANDALONE=0'],
	dependencies : [gtest_main_dep, lifted_hdr_dep, hwy_dep, hwy_test_dep],
	include_directories : test_inc,
)

test('single_level', test_single_levels, protocol:'gtest', suite:'mock')


test_dwt = executable(
	'test_dwt', 
	'test_dwt.cpp',
	cpp_args:['-DHWY_TEST_STANDALONE=0'],
	dependencies : [gtest_main_dep, lifted_lib_dep, hwy_dep],
	include_directories : test_inc,
)

test('driver', test_dwt, protocol:'gtest', suite: 'lwt')