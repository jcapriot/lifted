#include "wavelets.hpp"

using namespace wavelets;

namespace lifted {

	// forward
	void lwt_float(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
		const float* data_in, float* data_out, size_t n_threads = 1
	);

	void lwt_double(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
		const double* data_in, double* data_out, size_t n_threads = 1
	);

	void lwt_sep_float(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
		const float* data_in, float* data_out, size_t n_threads = 1
	);

	void lwt_sep_double(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
		const double* data_in, double* data_out, size_t n_threads = 1
	);

	// inverse
	void ilwt_float(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
		const float* data_in, float* data_out, size_t n_threads = 1
	);

	void ilwt_double(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
		const double* data_in, double* data_out, size_t n_threads = 1
	);

	void ilwt_sep_float(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
		const float* data_in, float* data_out, size_t n_threads = 1
	);

	void ilwt_sep_double(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
		const double* data_in, double* data_out, size_t n_threads = 1
	);

	// forward adjoint
	void lwt_adjoint_float(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
		const float* data_in, float* data_out, size_t n_threads = 1
	);

	void lwt_adjoint_double(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
		const double* data_in, double* data_out, size_t n_threads = 1
	);

	void lwt_sep_adjoint_float(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
		const float* data_in, float* data_out, size_t n_threads = 1
	);

	void lwt_sep_adjoint_double(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
		const double* data_in, double* data_out, size_t n_threads = 1
	);

	// inverse adjoint
	void ilwt_adjoint_float(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
		const float* data_in, float* data_out, size_t n_threads = 1
	);

	void ilwt_adjoint_double(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
		const double* data_in, double* data_out, size_t n_threads = 1
	);

	void ilwt_sep_adjoint_float(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
		const float* data_in, float* data_out, size_t n_threads = 1
	);

	void ilwt_sep_adjoint_double(
		const Wavelet wvlt, const BoundaryCondition bc,
		const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
		const double* data_in, double* data_out, size_t n_threads = 1
	);
}