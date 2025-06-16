#include "wavelets.hpp"

using namespace wavelets;

namespace lifted {

	void ilwt_adjoint_float(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
		const float* data_in, float* data_out, size_t n_threads = 1
	) {
		ilwt_adjoint(wvlt, bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
	}

	void ilwt_adjoint_double(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
		const double* data_in, double* data_out, size_t n_threads = 1
	) {
		ilwt_adjoint(wvlt, bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
	}
}