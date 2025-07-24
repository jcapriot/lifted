#if defined(LIFTED_OPS_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIFTED_OPS_INL_H_
#undef LIFTED_OPS_INL_H_
#else
#define LIFTED_OPS_INL_H_
#endif

#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "lifted-common.hpp"

#include "lifted-steps-inl.hpp"

HWY_BEFORE_NAMESPACE();
namespace lifted {
namespace detail {
namespace HWY_NAMESPACE {
    namespace hn = hwy::HWY_NAMESPACE;

    // Allocating a good sized Array:
    template<typename T> auto alloc_tmp(
        const size_t axsize, const size_t n_levels
    )
    {
        auto sz = axsize;
        // for each lvl of the transform, need to add an extra element if the resulting ns shape is odd
        // so there is working space for the diver to deinterleave s onto d at each level.
        int ns = axsize;
        for (size_t lvl = 0; lvl < n_levels; ++lvl) {
            if (ns % 2 == 1) sz += 1;
            ns = ns - ns / 2;
        }
        return hwy::AllocateAligned<T>(sz);
    }

    template<typename T> auto alloc_tmp(
        const size_v& shape, const size_t axsize, const size_t n_levels)
    {
        HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(hn::ScalableTag<T>());

        auto othersize = prod(shape) / axsize;
        auto sz = axsize;
        // for each lvl of the transform, need to add an extra element if the resulting ns shape is odd
        // so there is working space for the diver to deinterleave s onto d at each level.
        int ns = axsize;
        for (size_t lvl = 0; lvl < n_levels; ++lvl) {
            if (ns % 2 == 1) sz += 1;
            ns = ns - ns / 2;
        }
        auto tmpsize = sz * ((othersize >= lanes) ? lanes : 1);
        return hwy::AllocateAligned<T>(tmpsize);
    }

    template<typename T> auto alloc_tmp(
        const size_v& shape, const size_v& axes, const size_t n_levels)
    {
        HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(hn::ScalableTag<T>());
        size_t fullsize = prod(shape);
        size_t tmpsize = 0;
        for (size_t i = 0; i < axes.size(); ++i)
        {
            auto axsize = shape[axes[i]];
            auto othersize = fullsize / axsize;
            
            // for each lvl of the transform, need to add an extra element if the resulting ns shape is odd
            // so there is working space for the diver to deinterleave s onto d at each level.
            int ns = axsize;
            for (size_t lvl = 0; lvl < n_levels; ++lvl) {
                if (ns % 2 == 1) axsize += 1;
                ns = ns - ns / 2;
            }
            auto sz = axsize * ((othersize >= lanes) ? lanes : 1);
            if (sz > tmpsize) tmpsize = sz;
        }
        return hwy::AllocateAligned<T>(tmpsize);
    }

    
    // buffer interleaves and de-interleaves
    template <typename T>
    HWY_INLINE void deinterleave(
        const Across axis, const T* HWY_RESTRICT src, T* HWY_RESTRICT dst, const size_t len
    ) {
        size_t nd = len / 2;
        size_t ns = len - nd;
        const hn::ScalableTag<T> dtag;
        HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);

        T* HWY_RESTRICT lower = dst;
        T* HWY_RESTRICT upper = dst + ns * lanes;

        for (size_t i = 0, ii = 0; i < nd; ++i, ii += 2){
            const auto even = hn::Load(dtag, src + ii * lanes);
            const auto odd = hn::Load(dtag, src + (ii + 1) * lanes);
            hn::Store(even, dtag, lower + i * lanes);
            hn::Store(odd, dtag, upper + i * lanes);
        }
        if(ns > nd){
            const auto even = hn::Load(dtag, src + (len - 1) * lanes);
            hn::Store(even, dtag, lower + (ns - 1) * lanes);
        }
    }

    template <typename T>
    HWY_INLINE void deinterleave(
        const Along axis, const T* HWY_RESTRICT src, T* HWY_RESTRICT dst, const size_t len
    ) {
        size_t nd = len / 2;
        size_t ns = len - nd;

        const hn::ScalableTag<T> dtag;
        HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);

        T* HWY_RESTRICT lower = dst;
        T* HWY_RESTRICT upper = dst + ns;

        size_t ii = 0;
        if (nd >= lanes){
            const size_t N = nd - lanes;
            for (size_t i = 0; i <= N; i+=lanes, ii += lanes) {
                auto evens = hn::Undefined(dtag);
                auto odds = hn::Undefined(dtag);
                hn::LoadInterleaved2(dtag, src + 2 * i, evens, odds);
                hn::StoreU(evens, dtag, lower + i);
                hn::StoreU(odds, dtag, upper + i);
            }
        }
        for(; ii<nd; ++ii){
            lower[ii] = src[ii * 2];
            upper[ii] = src[ii * 2 + 1];
        }
        if(ns > nd){
            lower[ns - 1] = src[len - 1];
        }
    }

    template <typename T>
    HWY_INLINE void interleave(
        const Across axis, const T* HWY_RESTRICT src, T* HWY_RESTRICT dst, const size_t len
    ){
        size_t nd = len / 2;
        size_t ns = len - nd;

        const hn::ScalableTag<T> dtag;
        HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);

        for(size_t i = 0, ii = 0; i < nd; ++i, ii += 2){
            const auto first = hn::Load(dtag, src + i * lanes);
            const auto second = hn::Load(dtag, src + (i + ns) * lanes);
            hn::Store(first, dtag, dst + ii * lanes);
            hn::Store(second, dtag, dst + (ii + 1) * lanes);
        }
        if(ns > nd){
            const auto first = hn::Load(dtag, src + (ns - 1) * lanes);
            hn::Store(first, dtag, dst + (len - 1) * lanes);
        }
    }

    template <typename T>
    HWY_INLINE void interleave(
        const Along axis, const T* HWY_RESTRICT src, T* HWY_RESTRICT dst, const size_t len)
    {
        size_t nd = len / 2;
        size_t ns = len - nd;

        const hn::ScalableTag<T> dtag;
        HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);

        const T* HWY_RESTRICT s = src;
        const T* HWY_RESTRICT d = src + ns;

        size_t ii = 0;
        if (nd >= lanes){
            size_t N = nd - lanes;
            for(size_t i=0; i <= N; i += lanes, ii += lanes){
                const auto evens = hn::LoadU(dtag, s + i);
                const auto odds = hn::LoadU(dtag, d + i);
                hn::StoreInterleaved2(evens, odds, dtag, dst + i * 2);
            }
        }
        for(; ii < nd; ++ii){
            dst[ii * 2] = s[ii];
            dst[ii * 2 + 1] = d[ii];
        }
        if(ns > nd){
            dst[len - 1] = s[ns - 1];
        }
    }

    // copy in operations (copy, copy & interleave, copy & deinterleave)
    template <typename T, size_t vlen>
    HWY_INLINE void copy_input(
        const Across axes, const multi_iter<vlen>& it, const cndarr<T>& src, T* HWY_RESTRICT dst, const size_t from, const size_t to)
    {	
        const hn::ScalableTag<T> dtag;
        HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);

        // check if the array is contiguous Across the axes
        if (it.i_across_contiguous()){
            const T* HWY_RESTRICT sp = &src[it.iofs(0, from)];

            const size_t count = (to > from)? to - from : 0;
            const size_t stride_i = it.stride_in();

            for(size_t i = 0; i < count; ++i){
                const auto si = hn::LoadU(dtag, sp + i * stride_i);
                hn::Store(si, dtag, dst + i * lanes);
            }
        }else{
            for (size_t i = from, ii = 0; i < to; ++i, ++ii)
                for (size_t j = 0; j < lanes; ++j)
                    dst[ii * lanes + j] = src[it.iofs(j, i)];
        }
    }
    
    template <typename T, size_t vlen>
    HWY_INLINE void copy_input(
        const Along axes, const multi_iter<vlen>& it, const cndarr<T>& src, T* HWY_RESTRICT dst, const size_t from, const size_t to)
    {
        if (dst == &src[it.iofs(from)]) return;  // in-place

        if (it.i_along_contiguous()){
            const T* HWY_RESTRICT sp = &src[it.iofs(from)];
            const size_t count = (to > from)? to - from : 0;
            const auto dtag = hn::ScalableTag<T>();
            HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);
            size_t k = 0;
            if(count >= lanes){
                size_t n_full_load = count - lanes;
                for(size_t i = 0; i < n_full_load; i += lanes, k += lanes){
                    const auto si = hn::LoadU(dtag, sp + k);
                    hn::StoreU(si, dtag, dst + k);
                }
            }
            if(k < count){
                size_t rem = count - k;
                const auto si = hn::LoadN(dtag, sp + k, rem);
                hn::StoreN(si, dtag, dst + k, rem);
            }
        }else{
            for (size_t i = from, ii = 0; i < to; ++i, ++ii)
                dst[ii] = src[it.iofs(i)];
        }
    }

    template <typename T, size_t vlen>
    HWY_INLINE void copy_inout(
        const multi_iter<vlen>& it, const cndarr<T>& src, ndarr<T>& dst) {
        if (&dst[it.oofs(0)] == &src[it.iofs(0)]) return; //in-place
        if(it.i_along_contiguous() && it.o_along_contiguous()){

            const T* sp = &src[it.iofs(0)];
            T* dp = &dst[it.oofs(0)];

            const size_t count = it.length_in();
            const auto dtag = hn::ScalableTag<T>();

            HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);
            size_t k = 0;
            if(count >= lanes){
                size_t n_full_load = count - lanes;
                for(size_t i = 0; i < n_full_load; i += lanes, k += lanes){
                    const auto si = hn::LoadU(dtag, sp + k);
                    hn::StoreU(si, dtag, dp + k);
                }
            }
            if(k < count){
                size_t rem = count - k;
                const auto si = hn::LoadN(dtag, sp + k, rem);
                hn::StoreN(si, dtag, dp + k, rem);
            }
        }else{
            for (size_t i = 0; i < it.length_in(); ++i)
                dst[it.oofs(i)] = src[it.iofs(i)];
        }
    }

    template <typename T, size_t vlen>
    HWY_INLINE void deinterleave_input(
        const Across axes, const multi_iter<vlen>& it, const cndarr<T>& src, T* HWY_RESTRICT dst, const size_t len)
    {
        size_t nd = len / 2;
        size_t ns = len - nd;
        
        const auto dtag = hn::ScalableTag<T>();
        HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);
        // check if the array is contiguous Across the axes
        if (it.i_across_contiguous()){
            const T* HWY_RESTRICT sp = &src[it.iofs(0, 0)];
            const size_t stride_ii = it.stride_in();

            T* HWY_RESTRICT lower = dst;
            T* HWY_RESTRICT upper = dst + ns * lanes;

            for (size_t i = 0, ii = 0; i < nd; ++i, ii += 2){
                const auto even = hn::LoadU(dtag, sp + ii * stride_ii);
                const auto odd = hn::LoadU(dtag, sp + (ii + 1) * stride_ii);
                hn::Store(even, dtag, lower + i * lanes);
                hn::Store(odd, dtag, upper + i * lanes);
            }
            if(ns > nd){
                const auto even = hn::LoadU(dtag, sp + (len - 1) * stride_ii);
                hn::Store(even, dtag, lower + (ns - 1) * lanes);
            }
        }else{
            for (size_t i=0, ii=0; i < nd; ++i, ii += 2)
                for (size_t j = 0; j < lanes; ++j) {
                    dst[i * lanes + j] = src[it.iofs(j, ii)];
                    dst[(i + ns) * lanes + j] = src[it.iofs(j, ii + 1)];
                }
            if(ns > nd)
                for (size_t j = 0; j < lanes; ++j)
                    dst[(ns - 1) * lanes + j] = src[it.iofs(j, len - 1)];
        }
    }

    template <typename T, size_t vlen>
    HWY_INLINE void deinterleave_input(
        const Along axes, const multi_iter<vlen>& it, const cndarr<T>& src, T* HWY_RESTRICT dst, const size_t len)
    {

        if(it.i_along_contiguous()){
            deinterleave(axes, &src[it.iofs(0)], dst, len);
        }else{
            size_t nd = len / 2;
            size_t ns = len - nd;

            for (size_t i = 0, ii=0; i < nd; ++i, ii+=2) {
                dst[i] = src[it.iofs(ii)];
                dst[i + ns] = src[it.iofs(ii + 1)];
            }
            if(ns > nd) dst[ns - 1] = src[it.iofs(len - 1)];

        }
    }

    // copy out operations (copy, copy & interleave, copy & deinterleave)
    template <typename T, size_t vlen>
    HWY_INLINE void copy_output(
        const Across axes, const multi_iter<vlen>& it, const T* HWY_RESTRICT src, ndarr<T>& dst, const size_t from, const size_t to)
    {
        const auto dtag = hn::ScalableTag<T>();
        HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);

        if(it.o_across_contiguous()){
            T* HWY_RESTRICT dp = &dst[it.oofs(0, from)];
            const size_t count = (to > from)? to - from : 0;
            const size_t stride_i = it.stride_out();

            for(size_t i = 0, ii = 0; i < count; ++i, ++ii){
                const auto si = hn::Load(dtag, src + ii * lanes);
                hn::StoreU(si, dtag, dp + i * stride_i);
            }
        }else{
            for (size_t i = from, ii = 0; i < to; ++i, ++ii)
                for (size_t j = 0; j < lanes; ++j)
                    dst[it.oofs(j, i)] = src[ii * lanes + j];
        }
    }

    template <typename T, size_t vlen>
    HWY_INLINE void copy_output(
        const Along axes, const multi_iter<vlen>& it, const T* HWY_RESTRICT src, ndarr<T>& dst, const size_t from, const size_t to)
    {
        if (src == &dst[it.oofs(from)]) return;  // in-place
        if (it.o_along_contiguous()){
            T* HWY_RESTRICT dp = &dst[it.oofs(from)];
            const size_t count = (to > from)? to - from : 0;
            const auto dtag = hn::ScalableTag<T>();
            HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);
            size_t k = 0;
            if(count >= lanes){
                size_t n_full_load = count - lanes;
                for(size_t i = 0; i < n_full_load; i += lanes, k += lanes){
                    const auto si = hn::LoadU(dtag, src + k);
                    hn::StoreU(si, dtag, dp + k);
                }
            }
            if(k < count){
                size_t rem = count - k;
                const auto si = hn::LoadN(dtag, src + k, rem);
                hn::StoreN(si, dtag, dp + k, rem);
            }
        }else{
            for (size_t i = from, ii = 0; i < to; ++i, ++ii)
                dst[it.oofs(i)] = src[ii];
        }
    }

    template <typename T, size_t vlen>
    HWY_INLINE void interleave_output(
        const Across axes, const multi_iter<vlen>& it, const T* HWY_RESTRICT src, ndarr<T>& dst, const size_t len)
    {
        size_t nd = len / 2;
        size_t ns = len - nd;
        
        const auto dtag = hn::ScalableTag<T>();
        HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);

        const T* HWY_RESTRICT lower = src;
        const T* HWY_RESTRICT upper = src + ns * lanes;

        if(it.o_across_contiguous()){
            T* HWY_RESTRICT dp = &dst[it.oofs(0, 0)];
            
            const size_t stride_i = it.stride_out();

            for (size_t i=0, ii=0; i < nd; ++i, ii += 2){
                const auto lo = hn::Load(dtag, lower + i * lanes);
                const auto up = hn::Load(dtag, upper + i * lanes);
                hn::StoreU(lo, dtag, dp + ii * stride_i);
                hn::StoreU(up, dtag, dp + (ii + 1) * stride_i);
            }
            if(ns > nd){
                const auto lo = hn::Load(dtag, lower + (ns - 1) * lanes);
                hn::StoreU(lo, dtag, dp + (len - 1) * stride_i);
            }
        }else{
            for (size_t i=0, ii=0; i < nd; ++i, ii += 2)
                for (size_t j = 0; j < lanes; ++j) {
                    dst[it.oofs(j, ii)] = lower[i * lanes + j];
                    dst[it.oofs(j, ii + 1)] = upper[i * lanes + j];
                }
            if(ns > nd)
                for (size_t j = 0; j < lanes; ++j)
                    dst[it.oofs(j, len - 1)] = lower[(ns - 1) * lanes + j];

        }
    }

    template <typename T, size_t vlen>
    HWY_INLINE void interleave_output(
        const Along axes, const multi_iter<vlen>& it, const T* HWY_RESTRICT src, ndarr<T>& dst, const size_t len)
    {

        if (it.o_along_contiguous()){
            interleave(axes, src, &dst[it.oofs(0)], len);
        }else{
            size_t nd = len / 2;
            size_t ns = len - nd;
            for (size_t i=0, ii=0; i < nd; ++i, ii += 2) {
                dst[it.oofs(ii)] = src[i];
                dst[it.oofs(ii + 1)] = src[i + ns];
            }
            if(ns > nd) dst[it.oofs(len - 1)] = src[ns - 1];
        }
    }
}
}
}
HWY_AFTER_NAMESPACE();
#endif
