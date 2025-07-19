#ifndef LIFTED_H_
#define LIFTED_H_

#include "lifted-common.hpp"
#include "lifted-export.hpp"

namespace lifted{

    LIFTED_DLLEXPORT void lwt(
            Wavelet wvlt, Transform op, BoundaryCondition bc,
            const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
            const size_v& axes, const size_t levels,
            const float* data_in, float* data_out, const size_t n_threads
        );

    LIFTED_DLLEXPORT void lwt(
            Wavelet wvlt, Transform op, BoundaryCondition bc,
            const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
            const size_v& axes, const size_t levels,
            const double* data_in, double* data_out, const size_t n_threads
        );

    LIFTED_DLLEXPORT void lwt(
            Wavelet wvlt, Transform op, BoundaryCondition bc,
            const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
            const size_v& axes, const size_v& levels,
            const float* data_in, float* data_out, const size_t n_threads
        );

    LIFTED_DLLEXPORT void lwt(
            Wavelet wvlt, Transform op, BoundaryCondition bc,
            const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
            const size_v& axes, const size_v& levels,
            const double* data_in, double* data_out, const size_t n_threads
        );
};
#endif