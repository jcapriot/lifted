#include "wavelets.hpp"

using namespace wavelets;

namespace lifted {

	void lwt_sep_float(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
		const float* data_in, float* data_out, size_t n_threads = 1
	) {
		lwt(wvlt, bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
	}

	void lwt_sep_double(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
		const double* data_in, double* data_out, size_t n_threads = 1
	) {
		lwt(wvlt, bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
	}
}