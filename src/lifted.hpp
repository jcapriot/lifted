#ifndef LIFTED_H_
#define LIFTED_H_

#if !(__cplusplus >= 202002L || _MSVC_LANG+0L >= 202002L)
#error This file requires at least C++20 support.
#endif

#include "lifted-common.hpp"
#include "lifted-export.hpp"

namespace lifted{

    LIFTED_DLLEXPORT void lwt(
            const Wavelet wvlt, const Transform op, const BoundaryCondition bc,
            const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
            const size_v& axes, const size_t levels,
            const float* data_in, float* data_out, const size_t n_threads=1
        );

    LIFTED_DLLEXPORT void lwt(
            const Wavelet wvlt, const Transform op, const BoundaryCondition bc,
            const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
            const size_v& axes, const size_t levels,
            const double* data_in, double* data_out, const size_t n_threads=1
        );

    LIFTED_DLLEXPORT void lwt(
            const Wavelet wvlt, const Transform op, const BoundaryCondition bc,
            const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
            const size_v& axes, const size_v& levels,
            const float* data_in, float* data_out, const size_t n_threads=1
        );

    LIFTED_DLLEXPORT void lwt(
            const Wavelet wvlt, const Transform op, const BoundaryCondition bc,
            const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
            const size_v& axes, const size_v& levels,
            const double* data_in, double* data_out, const size_t n_threads=1
        );
};
#endif