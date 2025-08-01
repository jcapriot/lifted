#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lifted-sep-float.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/highway.h"
#include "lifted.hpp"
#include "lifted-inl.hpp"

namespace lifted{
namespace HWY_NAMESPACE{
    auto& lwt_sep_float = seperable_lifting_transform<float>;
}
}

#if HWY_ONCE
namespace lifted{
    HWY_EXPORT(lwt_sep_float);

    void lwt(
        const Wavelet wvlt, const Transform op, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
        const size_v& axes, const size_v& levels,
        const float* data_in, float* data_out, const size_t n_threads
    ){
        HWY_DYNAMIC_DISPATCH(lwt_sep_float)(
            wvlt, op, bc, shape, stride_in, stride_out, axes,
            levels, data_in, data_out, n_threads
        );
    }
}

#endif