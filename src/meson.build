
wvlt_inc = include_directories('.')

wvlt_hdr_sources = files(
	'ndarray.hpp',
	'wavelets.hpp'
)

wavelets_dep = declare_dependency(
	include_directories: wvlt_inc
)

if get_option('python_install')
    subdir('pylifted')
endif