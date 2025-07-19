#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lifted-float.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/highway.h"
#include "lifted.hpp"
#include "lifted-inl.hpp"

HWY_BEFORE_NAMESPACE();
namespace lifted{
namespace HWY_NAMESPACE{
    auto& lwt_float = lifting_transform<float>;
}
}
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace lifted{
    HWY_EXPORT(lwt_float);

    void lwt(
        Wavelet wvlt, Transform op, BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
        const size_v& axes, const size_t levels,
        const float* data_in, float* data_out, const size_t n_threads
    ){
        HWY_DYNAMIC_DISPATCH(lwt_float)(
            wvlt, op, bc, shape, stride_in, stride_out, axes,
            levels, data_in, data_out, n_threads
        );
    }
}

#endif