#if defined(LIFTED_DRIVERS_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIFTED_DRIVERS_INL_H_
#undef LIFTED_DRIVERS_INL_H_
#else
#define LIFTED_DRIVERS_INL_H_
#endif

#include <concepts>
#include <type_traits>
#include <numeric>

#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "lifted-common.hpp"
#include "lifted-wavelets.hpp"
#include "lifted-ops-inl.hpp"

HWY_BEFORE_NAMESPACE();
namespace lifted {
namespace detail {
namespace HWY_NAMESPACE {
    namespace hn = hwy::HWY_NAMESPACE;

    template<typename T>
    constexpr size_t vlen(){
        #if HWY_HAVE_CONSTEXPR_LANES
        return hn::Lanes(VecTag<T>());
        #else
        return dynamic_size;
        #endif
    }

    template<typename T>
    using multi_iter_type = multi_iter<vlen<T>()>;


    template<typename WVLT>
	struct FixedTransform {
    public:
        constexpr static size_t n_steps = std::tuple_size_v<decltype(WVLT::steps)>;
    private:
        using T = typename WVLT::type;
        constexpr static auto steps = tuple_generate<n_steps>(
            []<size_t k>() HWY_ATTR {
                return Step(std::get<k>(WVLT::steps));
            }
        );

    public:
        
        template<ForwardStepLoop TF, VecDir AX>
        static void apply(const TF tf, const AX axis, const BoundaryCondition bc, T* HWY_RESTRICT s, T* HWY_RESTRICT d, const size_t ns, const size_t nd){
            static_for<n_steps>([&]<size_t k>() HWY_ATTR {
                std::get<k>(steps)(tf, axis, bc, s, d, ns, nd);
            });
        }

        template<ReverseStepLoop TF, VecDir AX>
        static void apply(const TF tf, const AX axis, const BoundaryCondition bc, T* HWY_RESTRICT s, T* HWY_RESTRICT d, const size_t ns, const size_t nd){
            static_for<n_steps>([&]<size_t k>() HWY_ATTR {
                    std::get<n_steps - 1 - k>(steps)(tf, axis, bc, s, d, ns, nd);
            });
        }
	};

    template<typename WVLT>
    struct Driver{
    private:

    public:

        using transform = FixedTransform<WVLT>;
        using T = typename WVLT::type;
        // an n-level transform
        template<ForwardStepLoop TF, VecDir AX>
        static void apply(
            const TF tf,
            const AX axis,
            const BoundaryCondition bc,
            const multi_iter_type<T>& it,
            const cndarr<T>& ain,
            ndarr<T>& aout,
            T* HWY_RESTRICT buf,
            const size_t n,
            const size_t n_levels
        ) {
            if (n_levels == 0) return;
            
            size_t nd = n / 2;
            size_t ns = n - nd;

            // const auto bc = BC();  // boundary condition object
            HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(hn::ScalableTag<T>());

            // perform first level transform

            // deinterleave input into buffer
            deinterleave_input(axis, it, ain, buf, n);

            // get pointers to s and d
            T* s = buf;
            T* d = [&]{
                if constexpr (std::is_same_v<AX, Along>){
                    return s + ns;
                }
                else{
                    return s + lanes * ns;
                };
            }();

            transform::apply(tf, axis, bc, s, d, ns, nd);

            // copy d into output
            copy_output(axis, it, d, aout, ns, n);
            
            // If I have more levels... keep going
            for (size_t lvl = 1; lvl < n_levels; ++lvl) {
                // use "d" as the new buffer to deinterleave into (which should have enough space),
                // alloc_temp function adds an extra buffer element for odd transform lengths at each level)
                deinterleave(axis, s, d, ns);
                nd = ns / 2;
                ns = ns - nd;
                
                // get pointers to the starts of s and d;
                s = d;
                d = [&]{
                    if constexpr (std::is_same_v<AX, Along>){
                        return s + ns;
                    }
                    else{
                        return s + lanes * ns;
                    };
                }();
            
                transform::apply(tf, axis, bc, s, d, ns, nd);
                
                // copy d into output;
                copy_output(axis, it, d, aout, ns, ns + nd);
            }
            // copy last s into output;
            copy_output(axis, it, s, aout, 0, ns);
        }

        template<ForwardStepLoop TF, VecDir AX>
        static void apply(
            const TF tf,
            const AX axis,
            const BoundaryCondition bc,
            const multi_iter_type<T>& it,
            const cndarr<T>& ain,
            ndarr<T>& aout,
            T* HWY_RESTRICT buf,
            const size_t n
        ) {            
            size_t nd = n / 2;
            size_t ns = n - nd;

            //const BC bc;  // boundary condition object
            HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(hn::ScalableTag<T>());

            // deinterleave input into buffer
            deinterleave_input(axis, it, ain, buf, n);

            // get pointers to s and d
            T* s = buf;
            T* d = [&]{
                if constexpr (std::is_same_v<AX, Along>){
                    return s + ns;
                }
                else{
                    return s + lanes * ns;
                };
            }();

            transform::apply(tf, axis, bc, s, d, ns, nd);

            // copy into output
            copy_output(axis, it, s, aout, 0, n);
        }
    
        // an n-level transform
        template<ReverseStepLoop TF, VecDir AX>
        static void apply(
            const TF tf,
            const AX axis,
            const BoundaryCondition bc,
            const multi_iter_type<T>& it,
            const cndarr<T>& ain,
            ndarr<T>& aout,
            T* HWY_RESTRICT buf,
            const size_t n,
            const size_t n_levels
        ) {
            if (n_levels == 0) return;
            HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(hn::ScalableTag<T>());

            auto ns_s = std::vector<size_t>(n_levels);
            auto nd_s = std::vector<size_t>(n_levels);
            auto buffer_offsets = std::vector<size_t>(n_levels);
            size_t nd = n / 2;
            size_t ns = n - nd;
            size_t offsets = 0;
            for (size_t lvl = 0; lvl < n_levels; ++lvl) {
                buffer_offsets[lvl] = offsets;
                ns_s[lvl] = ns;
                nd_s[lvl] = nd;
                offsets += ns;
                nd = ns / 2;
                ns = ns - nd;
            }

            //const auto bc = BC();

            T* s = [&]{
                if constexpr (std::is_same_v<AX, Along>){
                    return buf + buffer_offsets[n_levels - 1];
                }
                else{
                    return buf + lanes * buffer_offsets[n_levels - 1];
                };
            }();
            T* d;
            ns = ns_s[n_levels - 1];
            // copy last level's s into the s_buffer;
            copy_input(axis, it, ain, s, 0, ns);
            for (size_t lvl = n_levels; lvl-- > 1;) {  // loops to the second to last level
                ns = ns_s[lvl];
                nd = nd_s[lvl];

                d = [&]{
                    if constexpr (std::is_same_v<AX, Along>){
                        return s + ns;
                    }
                    else{
                        return s + lanes * ns;
                    };
                }();
                // copy this level's d into the d_buffer;
                copy_input(axis, it, ain, d, ns, ns + nd);

                // transform this level in place
                transform::apply(tf, axis, bc, s, d, ns, nd);
                
                // interleave s and d into the next s_buffer;
                T* next_s = [&]{
                    if constexpr (std::is_same_v<AX, Along>){
                        return buf + buffer_offsets[lvl - 1];
                    }
                    else{
                        return buf + lanes * buffer_offsets[lvl - 1];
                    };
                }();
                interleave(axis, s, next_s, ns + nd);
                s = next_s;
            }
            ns = ns_s[0];
            nd = nd_s[0];
            d = [&]{
                    if constexpr (std::is_same_v<AX, Along>){
                        return s + ns;
                    }
                    else{
                        return s + lanes * ns;
                    };
                }();

            // copy the last levels input into the d buffer;
            copy_input(axis, it, ain, d, ns, ns + nd);

            transform::apply(tf, axis, bc, s, d, ns, nd);

            // interleave the buffer into the output;
            interleave_output(axis, it, buf, aout, n);
        }

        // a one level transform
        template<ReverseStepLoop TF, VecDir AX>
        static void apply(
            const TF tf,
            const AX axis,
            const BoundaryCondition bc,
            const multi_iter_type<T>& it,
            const cndarr<T>& ain,
            ndarr<T>& aout,
            T* HWY_RESTRICT buf,
            const size_t n
        ) {
            size_t nd = n / 2;
            size_t ns = n - nd;

            // const BC bc;
            HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(hn::ScalableTag<T>());

            // copy input into the buffer
            copy_input(axis, it, ain, buf, 0, n);

            T* s = buf;
            T* d = [&]{
                if constexpr (std::is_same_v<AX, Along>){
                    return s + ns;
                }
                else{
                    return s + lanes * ns;
                };
            }();

            transform::apply(tf, axis, bc, s, d, ns, nd);

            // interleave the buffer into the output;
            interleave_output(axis, it, buf, aout, n);
        }
    };

    template<typename T>
    using FuncType = std::function<void(const BoundaryCondition, const multi_iter_type<T>&, const cndarr<T>&, ndarr<T>&, T* HWY_RESTRICT, size_t)>;
    
    template<typename T>
    using SepFuncType = std::function<void(const BoundaryCondition, const multi_iter_type<T>&, const cndarr<T>&, ndarr<T>&, T* HWY_RESTRICT, size_t, size_t)>;

    template<DimSeperability SEP, typename T>
    struct WaveletOperations{
        using ftype = std::conditional_t<
            std::same_as<SEP, NonSeperableTransform>,
            FuncType<T>,
            SepFuncType<T>
        >;
        const bool forward_stepping;
        const size_t wvlt_width;
        const ftype along;
        const ftype across;

        WaveletOperations(
            const bool forward_stepping_,
            const size_t wvlt_width_,
            const ftype& along_,
            const ftype& across_
        ) : 
            along(along_),
            across(across_),
            forward_stepping(forward_stepping_),
            wvlt_width(wvlt_width_)
        {}
    };

    template<typename ftype>
    using func_info = std::tuple<ftype, ftype, bool, size_t>;

    template <typename T, VecDir AX, typename FuncType>
    static auto make_jump_table() {

        constexpr size_t n_wvlts = std::size(wavelet_enum_array);

        array< array<FuncType, LIFTED_N_TRANSFORM_TYPES>, n_wvlts> jump_table;

        static_for<n_wvlts>([&]<size_t IW>(){
            using WVLT = wavelet_from_enum_t<wavelet_enum_array[IW], T>;
            static_for<LIFTED_N_TRANSFORM_TYPES>([&]<size_t IT>(){
                using TF = transform_from_enum_t<Transform(IT)>;
                jump_table[IW][IT] = FuncType{[&](auto&&... args) {
                        Driver<WVLT>::apply(
                            TF(),
                            AX(),
                            std::forward<decltype(args)>(args)...
                        );
                    }
                };
            });
        });
        return jump_table;
    }

    template<DimSeperability SEP, typename T>
    static auto make_wvlt_table(){
        using op_type = WaveletOperations<SEP, T>;
        using ftype = typename op_type::ftype;
        using fi = func_info<ftype>;

        constexpr size_t n_wvlts = std::size(wavelet_enum_array);

        array< array<fi, LIFTED_N_TRANSFORM_TYPES>, n_wvlts> wvlt_table;
        
        static_for<n_wvlts>([&]<size_t IW>(){
            using WVLT = wavelet_from_enum_t<wavelet_enum_array[IW], T>;
            using driver = Driver<WVLT>;
            static_for<LIFTED_N_TRANSFORM_TYPES>([&]<size_t IT>(){
                using TF = transform_from_enum_t<Transform(IT)>;
                constexpr bool forward_stepping = ForwardStepLoop<TF>;
                constexpr size_t wvlt_width = WVLT::width;

                wvlt_table[IW][IT] = fi{
                    ftype{
                        [&](auto&&... args) {
                            driver::apply(
                                TF(),
                                Along(),
                                std::forward<decltype(args)>(args)...
                            );
                        }
                    },
                    ftype{
                        [&](auto&&... args) {
                            driver::apply(
                                TF(),
                                Across(),
                                std::forward<decltype(args)>(args)...
                            );
                        }
                    },
                    forward_stepping,
                    wvlt_width, 
                };
            });
        });
        return wvlt_table;
    }

    template<typename T>
    // seperable ND transform
    static void general_nd(
        const func_info<SepFuncType<T>>& fi, const BoundaryCondition bc, 
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
        const size_v& axes, const size_v& levels,
        const T* data_in, T* data_out, const size_t n_threads
    ){
        const auto& along = std::get<0>(fi);
        const auto& across = std::get<1>(fi);
        const auto wvlt_width = std::get<3>(fi);
        HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(hn::ScalableTag<T>());

        const size_t ndim = shape.size();

        // sort shape, stride_in, and stride_out by stride_out
        // to make memory access as efficient as possible.
        auto stride_order = size_v(ndim);
        std::iota(stride_order.begin(), stride_order.end(), 0);
        std::sort(stride_order.begin(), stride_order.end(),
            [&](const size_t i1, const size_t i2){ return std::abs(stride_out[i1]) < std::abs(stride_out[i2]);}
        );

        auto shape_ = size_v(ndim);
        auto stride_in_ = stride_v(ndim);
        auto stride_out_ = stride_v(ndim);
        for(auto i = 0; i < ndim; ++i){
            const size_t i_ax = stride_order[i];
            shape_[i] = shape[i_ax];
            stride_in_[i] = stride_in[i_ax];
            stride_out_[i] = stride_out[i_ax];
        }
        
        const size_t n_ax = axes.size();
        auto axes_ = size_v(n_ax);
        for(auto i = 0; i < n_ax; ++i){
            axes_[i] = stride_order[axes[i]];
        }

        auto ain = cndarr<T>(data_in, shape_, stride_in_);
        auto aout = ndarr<T>(data_out, shape_, stride_out_);

        for (size_t iax = 0; iax < n_ax; ++iax) {
            size_t ax = axes_[iax];
            size_t len = ain.shape(ax);
            size_t level = levels[iax];
            if (level == 0) level = max_level(wvlt_width, len);

            threading::thread_map(
                threading::thread_count(n_threads, ain.shape(), ax, lanes),
                [&] {
                    auto storage = alloc_tmp<T>(ain.shape(), len, level);
                    const auto& tin(iax == 0 ? ain : aout);
                    #if HWY_HAVE_CONSTEXPR_LANES
                    multi_iter_type<T> it(tin, aout, ax);
                    #else
                    multi_iter_type<T> it(tin, aout, ax, lanes);
                    #endif
                    while (it.remaining() >= lanes) {
                        it.advance(lanes);
                        across(bc, it, tin, aout, storage.get(), len, level);
                    }
                    while (it.remaining() > 0) {
                        it.advance(1);
                        along(bc, it, tin, aout, storage.get(), len, level);
                    }
                }
            );  // end of parallel region
        }
    }

    template<typename T>
    // non-seperable ND transform
    static void general_nd(
        const func_info<FuncType<T>>& fi, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
        const size_v& axes, const size_t level,
        const T* data_in, T* data_out, const size_t n_threads
    ){
        const auto& along = std::get<0>(fi);
        const auto& across = std::get<1>(fi);
        const auto forward_stepping = std::get<2>(fi);
        const auto wvlt_width = std::get<3>(fi);

        HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(hn::ScalableTag<T>());

        const size_t ndim = shape.size();

        // sort shape, stride_in, and stride_out by stride_out
        // to make memory access as efficient as possible.
        auto stride_order = size_v(ndim);
        std::iota(stride_order.begin(), stride_order.end(), 0);
        std::sort(stride_order.begin(), stride_order.end(),
            [&](const size_t i1, const size_t i2){ return std::abs(stride_out[i1]) < std::abs(stride_out[i2]);}
        );

        auto shape_ = size_v(ndim);
        auto stride_in_ = stride_v(ndim);
        auto stride_out_ = stride_v(ndim);
        for(size_t i = 0; i < ndim; ++i){
            const size_t i_ax = stride_order[i];
            shape_[i] = shape[i_ax];
            stride_in_[i] = stride_in[i_ax];
            stride_out_[i] = stride_out[i_ax];
        }
        
        const size_t n_ax = axes.size();
        auto axes_ = size_v(n_ax);
        for(size_t i = 0; i < n_ax; ++i){
            axes_[i] = stride_order[axes[i]];
        }

        size_t lvl = level;
        if (lvl == 0) lvl = max_level(wvlt_width, shape_, axes_);

        // A list of shapes for each level:
        auto level_shapes = std::vector<size_v>(lvl);
        level_shapes[0] = size_v(shape_);
        for (size_t ilvl = 1; ilvl < lvl; ++ilvl){
            auto& shape_im1 = level_shapes[ilvl - 1];
            auto shape_i = size_v(shape_im1);
            for (auto ax : axes_) shape_i[ax] = shape_im1[ax] - shape_im1[ax] / 2;
            level_shapes[ilvl] = shape_i;
        }

        if (!forward_stepping) {
            if (data_in != data_out) {
                // shape_, stride_in_, and stride_out_ have been sorted by stride_out_
                auto ain = cndarr<T>(data_in, shape_, stride_in_);
                auto aout = ndarr<T>(data_out, shape_, stride_out_);

                threading::thread_map(
                    threading::thread_count(n_threads, ain.shape(), 0, lanes),
                    [&] {
                        #if HWY_HAVE_CONSTEXPR_LANES
                        multi_iter_type<T> it(ain, aout, 0);
                        #else
                        multi_iter_type<T> it(ain, aout, 0, lanes);
                        #endif

                        while (it.remaining() > 0) {
                            it.advance(1);
                            copy_inout(it, ain, aout);
                        }
                    }
                );  // end of parallel region

                data_in = data_out;
                stride_in_ = stride_out_;
            }
        }

        for (size_t ilvl = 0; ilvl < lvl; ++ilvl) {

            const auto& shape_lvl = (forward_stepping)? level_shapes[ilvl] : level_shapes[lvl - 1 - ilvl];

            auto ain = cndarr<T>(data_in, shape_lvl, stride_in_);
            auto aout = ndarr<T>(data_out, shape_lvl, stride_out_);

            for (size_t iax = 0; iax < n_ax; ++iax) {
                size_t ax = axes_[iax];
                size_t len = ain.shape(ax);

                bool out_dim_contiguous = (stride_out_[ax] == 1);

                threading::thread_map(
                    threading::thread_count(n_threads, ain.shape(), ax, lanes),
                    [&] {
                        auto storage = (out_dim_contiguous)?
                            alloc_tmp<T>(len, level) :
                            alloc_tmp<T>(ain.shape(), len, level);
                        const auto& tin(iax == 0 ? ain : aout);
                        #if HWY_HAVE_CONSTEXPR_LANES
                        multi_iter_type<T> it(tin, aout, ax);
                        #else
                        multi_iter_type<T> it(tin, aout, ax, lanes);
                        #endif

                        while (!out_dim_contiguous && (it.remaining() >= lanes)) {
                            it.advance(lanes);
                            across(bc, it, tin, aout, storage.get(), len);
                        }
                        while (it.remaining() > 0) {
                            it.advance(1);
                            along(bc, it, tin, aout, storage.get(), len);
                        }
                    }
                );  // end of parallel region
            }

            data_in = data_out;
            stride_in_ = stride_out_;
        }
    }


    template<DimSeperability SEP, typename T, typename...Ts>
    static void lifting_transform_dispatcher(
        const Wavelet wvlt, const Transform op,
        Ts&&... args
    ){
        const static auto ops_table = make_wvlt_table<SEP, T>();

        const size_t i_wvlt = wavelet_enum_to_index[wvlt];
        const size_t i_tt = static_cast<size_t>(op);
    
        general_nd(ops_table[i_wvlt][i_tt], std::forward<Ts>(args)...);
    }
}
}
namespace HWY_NAMESPACE {
    namespace dh = detail::HWY_NAMESPACE;

    template<typename T>
    static void lifting_transform(
        const Wavelet wvlt, const Transform op, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
        const size_v& axes, const size_t level,
        const T* data_in, T* data_out, const size_t n_threads
    ){
        dh::lifting_transform_dispatcher<NonSeperableTransform, T>(
            wvlt, op, bc,
            shape, stride_in, stride_out,
            axes, level, data_in, data_out, n_threads
        );
    }

    template<typename T>
    static void seperable_lifting_transform(
        const Wavelet wvlt, const Transform op, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
        const size_v& axes, const size_v& levels,
        const T* data_in, T* data_out, const size_t n_threads
    ){
        dh::lifting_transform_dispatcher<SeperableTransform, T>(
            wvlt, op, bc,
            shape, stride_in, stride_out,
            axes, levels, data_in, data_out, n_threads
        );
    }
}
}
HWY_AFTER_NAMESPACE();
#endif