#if defined(LIFTED_STEPS_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIFTED_STEPS_INL_H_
#undef LIFTED_STEPS_INL_H_
#else
#define LIFTED_STEPS_INL_H_
#endif

// It is fine to #include normal or *-inl headers.
#include "hwy/highway.h"

#include <cstddef>
#include <array>
#include <utility>
#include <concepts>
#include <type_traits>
#include <algorithm>
#include <stdexcept>

#include "lifted-common.hpp"

#include <iostream>

HWY_BEFORE_NAMESPACE();
namespace lifted {
namespace detail {
namespace HWY_NAMESPACE {
    namespace hn = hwy::HWY_NAMESPACE;

    template<typename T>
    using VecTag = decltype(hn::ScalableTag<T>());

    template<typename T>
    using VecType = hn::VFromD<VecTag<T>>;

    struct ISub;

    struct IAdd{
        using Reverse = ISub;

        template<typename T>
        static HWY_INLINE void iop(T& v1, const T& v2){
            v1 += v2;
        }

        template<typename VT>
        static HWY_INLINE void iop_v(VT& v1, const VT& v2){
            v1 = hn::Add(v1, v2);
        }

        template<typename VT>
        static HWY_INLINE VT fused_iop(const VT& v1, const VT& v2, const VT& v3){
            return hn::MulAdd(v1, v2, v3);
        }
    };

    struct ISub{
        using Reverse = IAdd;

        template<typename T>
        static HWY_INLINE void iop(T& v1, const T& v2){
            v1 -= v2;
        }

        template<typename VT>
        static HWY_INLINE void iop_v(VT& v1, const VT& v2){
            v1 = hn::Sub(v1, v2);
        }

        template<typename VT>
        static HWY_INLINE VT fused_iop(const VT& v1, const VT& v2, const VT& v3){
            return hn::NegMulAdd(v1, v2, v3);
        }
    };

    template<TransformType TF> struct op_from_tf{};
    template<ForwardTransform TF> struct op_from_tf<TF>{using op = IAdd;};
    template<InverseTransform TF> struct op_from_tf<TF>{using op = ISub;};

    template<TransformType TF>
    using op_from_tf_t = op_from_tf<TF>::op;

    template<typename T>
    concept UpdateOp = std::same_as<T, IAdd> || std::same_as<T, ISub>;
    
    namespace ip{

        template<size_t N, UpdateOp OP, typename T>
        static HWY_INLINE void inner_product(
            const OP op,
            const VecTag<T> d,
            VecType<T>& inout,
            const array<T, N>& vals,
            const ptrdiff_t offset,
            const T* HWY_RESTRICT x,
            const size_t i
        ) {
            size_t io = i + offset;
            static_for<N>([&]<size_t k>() HWY_ATTR {
                const auto vk = hn::Set(d, vals[k]);
                const auto xi = hn::LoadU(d, x + io + k);
                inout = OP::fused_iop(vk, xi, inout);
            });
        }

        template<size_t N, UpdateOp OP, typename T>
        static HWY_INLINE void inner_product(
            const OP op,
            const VecTag<T> d,
            VecType<T>& inout,
            const VecType<T>& v,
            const ptrdiff_t offset,
            const T* HWY_RESTRICT x,
            const size_t i
        ) {
            size_t io = i + offset;
            auto x0 = hn::LoadU(d, x + io);
            static_for<N - 1>([&]<size_t k>() HWY_ATTR {
                const auto xi = hn::LoadU(d, x + io + k + 1);
                x0 = hn::Add(x0, xi);
            });
            inout = OP::fused_iop(v, x0, inout);
        }

        template<size_t N, UpdateOp OP, typename T>
        static HWY_INLINE void inner_product(
            const OP op,
            const VecTag<T> d,
            VecType<T>& inout,
            const array<T, N>& vals,
            const ptrdiff_t offset,
            const T* HWY_RESTRICT x,
            const size_t i,
            const size_t lanes
        ) {
            size_t io = i + offset;
            static_for<N>([&]<size_t k>() HWY_ATTR {
                const auto vk = hn::Set(d, vals[k]);
                const auto xi = hn::Load(d, x + (io + k) * lanes);
                inout = OP::fused_iop(vk, xi, inout);
            });
        }

        template<size_t N, UpdateOp OP, typename T>
        static HWY_INLINE void inner_product(
            const OP op,
            const VecTag<T> d,
            VecType<T>& inout,
            const VecType<T>& v,
            const ptrdiff_t offset,
            const T* HWY_RESTRICT x,
            const size_t i,
            const size_t lanes
        ) {
            size_t io = i + offset;
            auto x0 = hn::Load(d, x + io * lanes);
            static_for<N - 1>([&]<size_t k>() HWY_ATTR {
                const auto xi = hn::Load(d, x + (io + k + 1) * lanes);
                x0 = hn::Add(x0, xi);
            });
            inout = OP::fused_iop(v, x0, inout);
        }

        template<size_t N, UpdateOp OP, BoundCond BC, typename T>
        static HWY_INLINE void inner_product(
            const OP op,
            const BC bc,
            const VecTag<T> d,
            VecType<T>& inout,
            const array<T, N>& vals,
            const ptrdiff_t offset,
            const T* HWY_RESTRICT x,
            const size_t i,
            const size_t n,
            const size_t lanes
        ) {
            size_t io = i + offset;
            static_for<N>([&]<size_t k>() HWY_ATTR {
                size_t i_bc = BC::index_of(io + ptrdiff_t(k), n);
                if (i_bc < n){
                    const auto vk = hn::Set(d, vals[k]);
                    const auto xi = hn::Load(d, x + i_bc*lanes);
                    inout = OP::fused_iop(vk, xi, inout);
                }
            });
        }

        template<size_t N, UpdateOp OP, BoundCond BC, typename T>
        static HWY_INLINE void inner_product(
            const OP op,
            const BC bc,
            const VecTag<T> d,
            VecType<T>& inout,
            const VecType<T>& v,
            const ptrdiff_t offset,
            const T* HWY_RESTRICT x,
            const size_t i,
            const size_t n,
            const size_t lanes
        ) {
            size_t io = i + offset;
            size_t i_bc = BC::index_of(io, n);
            auto x0 = (i_bc < n)? hn::Load(d, x + i_bc * lanes) : hn::Zero(d);
            static_for<N - 1>([&]<size_t k>() HWY_ATTR {
                i_bc = BC::index_of(io + ptrdiff_t(k) + 1, n);
                if (i_bc < n){
                    const auto xi = hn::Load(d, x + i_bc*lanes);
                    x0 = hn::Add(x0, xi);
                }
            });
            inout = OP::fused_iop(v, x0, inout);
        }

        template<size_t N, UpdateOp OP, BoundCond BC, typename T>
        static HWY_INLINE void inner_product(
            const OP op,
            const BC bc,
            T& HWY_RESTRICT inout,
            const array<T, N>& vals,
            const ptrdiff_t offset,
            const T* HWY_RESTRICT x,
            const size_t i,
            const size_t n
        ) {
            const size_t io = i + offset;
            static_for<N>([&]<size_t k>() {
                size_t i_bc = BC::index_of(io + ptrdiff_t(k), n);
                if (i_bc < n){
                    OP::iop(inout, vals[k] * x[i_bc]);
                }
            });
        }

        template<size_t N, UpdateOp OP, BoundCond BC, typename T>
        static HWY_INLINE void inner_product(
            const OP op,
            const BC bc,
            T& HWY_RESTRICT inout,
            const T v,
            const ptrdiff_t offset,
            const T* HWY_RESTRICT x,
            const size_t i,
            const size_t n
        ) {
            const size_t io = i + offset;
            size_t i_bc = BC::index_of(io, n);
            T x0 = (i_bc < n)? x[i_bc] : T(0.0);
            static_for<N - 1>([&]<size_t k>() {
                size_t i_bc = BC::index_of(io + ptrdiff_t(k) + 1, n);
                if (i_bc < n){
                    x0 += x[i_bc];
                }
            });
            
            OP::iop(inout, v * x0);
        }

        template<size_t N, UpdateOp OP, typename T>
        static HWY_INLINE void inner_product_n(
            const OP op,
            const VecTag<T> d,
            VecType<T>& inout,
            const array<T, N>& vals,
            const ptrdiff_t offset,
            const T* HWY_RESTRICT x,
            const size_t i,
            const size_t rem
        ) {
            size_t io = i + offset;
            static_for<N>([&]<size_t k>() HWY_ATTR {
                const auto vk = hn::Set(d, vals[k]);
                const auto xi = hn::LoadN(d, x + io + k, rem);
                inout = OP::fused_iop(vk, xi, inout);
            });
        }

        template<size_t N, UpdateOp OP, typename T>
        static HWY_INLINE void inner_product_n(
            const OP op,
            const VecTag<T> d,
            VecType<T>& inout,
            const VecType<T>& v,
            const ptrdiff_t offset,
            const T* HWY_RESTRICT x,
            const size_t i,
            const size_t rem
        ) {
            size_t io = i + offset;
            auto x0 = hn::LoadN(d, x + io, rem);
            static_for<N - 1>([&]<size_t k>() HWY_ATTR {
                const auto xi = hn::LoadN(d, x + io + k + 1, rem);
                x0 = hn::Add(x0, xi);
            });
            inout = OP::fused_iop(v, x0, inout);
        }

        template<size_t N, UpdateOp OP, typename T>
        static HWY_INLINE void inner_product_safe(
            const OP op,
            T& HWY_RESTRICT inout,
            const array<T, N>& vals,
            const T* HWY_RESTRICT x,
            const ptrdiff_t i,
            const size_t n
        ) {
            static_for<N>([&]<size_t k>() {
                ptrdiff_t ii = i + ptrdiff_t(k);
                if((ii >= 0 ) && (ii < ptrdiff_t(n))) OP::iop(inout, vals[k] * x[ii]);
            });
        }

        template<size_t N, UpdateOp OP, typename T>
        static HWY_INLINE void inner_product_safe(
            const OP op,
            T& HWY_RESTRICT inout,
            const T v,
            const T* HWY_RESTRICT x,
            const ptrdiff_t i,
            const size_t n
        ) {
            
            T x0 = ((i >= 0 ) && (i < ptrdiff_t(n)))? x[i] : T(0);
            static_for<N - 1>([&]<size_t k>() {
                ptrdiff_t ii = i + ptrdiff_t(k) + 1;
                if((ii >= 0 ) && (ii < ptrdiff_t(n)))
                    x0 += x[ii];
            });
            OP::iop(inout, v * x0);
        }
        
        template<size_t N, UpdateOp OP, typename T>
        static HWY_INLINE void inner_product_safe(
            const OP op,
            const VecTag<T> d,
            VecType<T>& inout,
            const array<T, N>& vals,
            const T* HWY_RESTRICT x,
            const ptrdiff_t i,
            const size_t n,
            const size_t lanes
        ) {
            static_for<N>([&]<size_t k>() HWY_ATTR {
                ptrdiff_t ii = i + ptrdiff_t(k);
                if((ii >= 0 ) && (ii < ptrdiff_t(n))){
                    const auto vk = hn::Set(d, vals[k]);
                    const auto xi = hn::Load(d, x + ii*lanes);
                    inout = OP::fused_iop(vk, xi, inout);
                }
            });
        }
        
        template<size_t N, UpdateOp OP, typename T>
        static HWY_INLINE void inner_product_safe(
            const OP op,
            const VecTag<T> d,
            VecType<T>& inout,
            const VecType<T>& v,
            const T* HWY_RESTRICT x,
            const ptrdiff_t i,
            const size_t n,
            const size_t lanes
        ) {
            auto x0 = ((i >= 0 ) && (i < ptrdiff_t(n)))? hn::Load(d, x + i * lanes) : hn::Zero(d);
            static_for<N - 1>([&]<size_t k>() HWY_ATTR {
                ptrdiff_t ii = i + ptrdiff_t(k) + 1;
                if((ii >= 0 ) && (ii < ptrdiff_t(n))){
                    const auto xi = hn::Load(d, x + ii*lanes);
                    x0 = hn::Add(x0, xi);
                }
            });
            inout = OP::fused_iop(v, x0, inout);
        }
    }

    template<typename STEP> class Step{};
    
    template<typename T, size_t N>
    class Step<GenericUpdateStep<T, N>>{
    private:
        const GenericUpdateStep<T, N>& step_dat;    
    public:
        constexpr Step(const GenericUpdateStep<T, N>& step_dat_) : step_dat(step_dat_) {};
 
        template<TransformType TF>
        void operator()(
            const TF tf, const Along axis, const BoundaryCondition bc,
            T* HWY_RESTRICT s, T* HWY_RESTRICT d, const size_t ns, const size_t nd
        ) const {
            using OP = op_from_tf_t<TF>;
            const auto op = OP();
            
            constexpr auto is_normal = NormalTransform<TF>;
            const auto target_update_type = [&]{
                if constexpr (is_normal) return UpdateType::UpdateOdds; else return UpdateType::UpdateEvens;
            }();
            const size_t nf = [&]{if constexpr (is_normal) return step_dat.n_front; else return step_dat.n_front_r;}();
            const size_t nb = [&]{if constexpr (is_normal) return step_dat.n_back; else return step_dat.n_back_r;}();
            const ptrdiff_t off = [&]{if constexpr (is_normal) return step_dat.offset; else return step_dat.offset_r;}();
            const auto& vs = [&]{if constexpr (is_normal) return step_dat.vals; else return step_dat.vals_r;}();

            // using BCT = std::conditional_t<is_normal, BC, ZeroBoundary>;
            // const auto bcond = BCT();

			const size_t nx = (step_dat.update_type == target_update_type) ? ns : nd;
			const T* HWY_RESTRICT x = (step_dat.update_type == target_update_type) ? s : d;

			const size_t ny = (step_dat.update_type == target_update_type) ? nd : ns;
			T* HWY_RESTRICT y = (step_dat.update_type == target_update_type) ? d : s;

            const size_t n1 = std::min(nf, ny);
            const size_t n2 = (nd > nb)? nd - nb : 0;

            const VecTag<T> dtag;
            HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);

            
            if constexpr(!is_normal){
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = step_dat.offset, j = 1 - ptrdiff_t(N); i < 0; ++i, ++j) { \
                        size_t io = name##Boundary::index_of(i, ny); \
                        if ((io != ny) && (ptrdiff_t(io) != i)) \
                            ip::inner_product_safe(op, y[io], vs, x, j, nx); \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
                for (ptrdiff_t i = 0; i < ptrdiff_t(n1); ++i)
                    ip::inner_product(op, ZeroBoundary(), y[i], vs, off, x, i, nx);
            }else{
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = 0; i < ptrdiff_t(n1); ++i) \
                        ip::inner_product(op, name##Boundary(), y[i], vs, off, x, i, nx); \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
            }

            size_t k = n1;
            const size_t count = (n2 > n1)? n2 - n1: 0;
            if (count >= lanes){
                const size_t n_safe_load = count - lanes;
                for(size_t i = 0; i <= n_safe_load; k += lanes, i += lanes){
                    auto yi = hn::LoadU(dtag, y + k);
                    ip::inner_product(op, dtag, yi, vs, off, x, k);
                    hn::StoreU(yi, dtag, y + k);
                }
            }
            if (k < n2 ){
                // Handle the last vector
                const size_t rem = n2 - k;
                auto yi = hn::LoadN(dtag, y + k, rem);
                ip::inner_product_n(op, dtag, yi, vs, off, x, k, rem);
                hn::StoreN(yi, dtag, y + k, rem);
            }

            
            if constexpr(!is_normal){

                for (ptrdiff_t i = n2; i < ptrdiff_t(ny); ++i)
                    ip::inner_product(op, ZeroBoundary(), y[i], vs, off, x, i, nx);

                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = nd, j = ptrdiff_t(nd) + off; i < ptrdiff_t(nx) - off; ++i, ++j) { \
                        size_t io = name##Boundary::index_of(i, ny); \
                        if ((io != ny) && (ptrdiff_t(io) != i)) \
                            ip::inner_product_safe(op, y[io], vs, x, j, nx); \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
            }else{
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = n2; i < ptrdiff_t(ny); ++i) \
                        ip::inner_product(op, name##Boundary(), y[i], vs, off, x, i, nx); \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
            }
		}

        template<TransformType TF>
		void operator()(   
            const TF tf, const Across axis, const BoundaryCondition bc,
            T* HWY_RESTRICT s, T* HWY_RESTRICT d, const size_t ns, const size_t nd
        ) const {
            using OP = op_from_tf_t<TF>;
            const auto op = OP();

            constexpr auto is_normal = NormalTransform<TF>;
            const auto target_update_type = [&]{
                if constexpr (is_normal) return UpdateType::UpdateOdds; else return UpdateType::UpdateEvens;
            }();
            const size_t nf = [&]{if constexpr (is_normal) return step_dat.n_front; else return step_dat.n_front_r;}();
            const size_t nb = [&]{if constexpr (is_normal) return step_dat.n_back; else return step_dat.n_back_r;}();
            const ptrdiff_t off = [&]{if constexpr (is_normal) return step_dat.offset; else return step_dat.offset_r;}();
            const auto& vs = [&]{if constexpr (is_normal) return step_dat.vals; else return step_dat.vals_r;}();

			const size_t nx = (step_dat.update_type == target_update_type) ? ns : nd;
			const T* HWY_RESTRICT x = (step_dat.update_type == target_update_type) ? s : d;

			const size_t ny = (step_dat.update_type == target_update_type) ? nd : ns;
			T* HWY_RESTRICT y = (step_dat.update_type == target_update_type) ? d : s;

            const size_t n1 = std::min(nf, ny);
            const size_t n2 = (nd > nb)? nd - nb : 0;

            const VecTag<T> dtag;
            HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);

            if constexpr(!is_normal){
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = step_dat.offset, j = 1 - ptrdiff_t(N); i < 0; ++i, ++j) { \
                        size_t io = name##Boundary::index_of(i, ny); \
                        if ((io != ny) && (ptrdiff_t(io) != i)){ \
                            auto yi = hn::Load(dtag, y + io * lanes); \
                            ip::inner_product_safe(op, dtag, yi, vs, x, j, nx, lanes); \
                            hn::Store(yi, dtag, y + io * lanes); \
                        } \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
                for (ptrdiff_t i = 0; i < ptrdiff_t(n1); ++i){
                    auto yi = hn::Load(dtag, y + i * lanes);
                    ip::inner_product(op, ZeroBoundary(), dtag, yi, vs, off, x, i, nx, lanes);
                    hn::Store(yi, dtag, y + i * lanes);
                }
            }else{
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = 0; i < ptrdiff_t(n1); ++i){ \
                        auto yi = hn::Load(dtag, y + i * lanes); \
                        ip::inner_product(op, name##Boundary(), dtag, yi, vs, off, x, i, nx, lanes); \
                        hn::Store(yi, dtag, y + i * lanes); \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
            }

            for (size_t i = n1; i < n2; ++i){
                auto yi = hn::Load(dtag, y + i * lanes);
                ip::inner_product(op, dtag, yi, vs, off, x, i, lanes);
                hn::Store(yi, dtag, y + i * lanes);
            }

            //back bc loop
            if constexpr(!is_normal){
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = nd, j = ptrdiff_t(nd) + off; i < ptrdiff_t(nx) - off; ++i, ++j) { \
                        size_t io = name##Boundary::index_of(i, ny); \
                        if ((io != ny) && (ptrdiff_t(io) != i)){ \
                            auto yi = hn::Load(dtag, y + io * lanes); \
                            ip::inner_product_safe(op, dtag, yi, vs, x, j, nx, lanes); \
                            hn::Store(yi, dtag, y + io * lanes); \
                        } \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
                for (ptrdiff_t i = n2; i < ptrdiff_t(ny); ++i){
                    auto yi = hn::Load(dtag, y + i * lanes);
                    ip::inner_product(op, ZeroBoundary(), dtag, yi, vs, off, x, i, nx, lanes);
                    hn::Store(yi, dtag, y + i * lanes);
                }
            }else{
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = n2; i < ptrdiff_t(ny); ++i){ \
                        auto yi = hn::Load(dtag, y + i * lanes); \
                        ip::inner_product(op, name##Boundary(), dtag, yi, vs, off, x, i, nx, lanes); \
                        hn::Store(yi, dtag, y + i * lanes); \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
            }
		}
    };

    template<typename T, UpdateOperation PM>
    class Step<UnitUpdateStep<T, PM>>{
    private:
        const UnitUpdateStep<T, PM>& step_dat;

        using unit_op = std::conditional_t<PM == UpdateOperation::add, IAdd, ISub>;

        template<TransformType TF>
        using unit_op_from_tf_t = std::conditional_t<ForwardTransform<TF>, unit_op, typename unit_op::Reverse>;

    public:

        constexpr Step(const UnitUpdateStep<T, PM>& step_dat_) : step_dat(step_dat_){}
        
        template<TransformType TF>
        void operator()(
            const TF tf, const Along axis, const BoundaryCondition bc,
            T* HWY_RESTRICT s, T* HWY_RESTRICT d, const size_t ns, const size_t nd
        ) const {
            using OP = unit_op_from_tf_t<TF>;
            
            constexpr auto is_normal = NormalTransform<TF>;
            const auto target_update_type = [&]{
                if constexpr (is_normal) return UpdateType::UpdateOdds; else return UpdateType::UpdateEvens;
            }();
            const size_t nf = [&]{if constexpr (is_normal) return step_dat.n_front; else return step_dat.n_front_r;}();
            const size_t nb = [&]{if constexpr (is_normal) return step_dat.n_back; else return step_dat.n_back_r;}();
            const ptrdiff_t off = [&]{if constexpr (is_normal) return step_dat.offset; else return step_dat.offset_r;}();

			const size_t nx = (step_dat.update_type == target_update_type) ? ns : nd;
			const T* HWY_RESTRICT x = (step_dat.update_type == target_update_type) ? s : d;

			const size_t ny = (step_dat.update_type == target_update_type) ? nd : ns;
			T* HWY_RESTRICT y = (step_dat.update_type == target_update_type) ? d : s;

            const size_t n1 = std::min(nf, ny);
            const size_t n2 = (nd > nb)? nd - nb : 0;

            const VecTag<T> dtag;
            HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);

            if constexpr(!is_normal){
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = step_dat.offset, j = 0; (i < 0) && (j < ptrdiff_t(nx)); ++i, ++j) { \
                        size_t io = name##Boundary::index_of(i, ny); \
                        if ((io != ny) && (ptrdiff_t(io) != i)) \
                            OP::iop(y[io], x[j]); \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
                for (ptrdiff_t i = 0; i < ptrdiff_t(n1); ++i){
                    const size_t i_bc = ZeroBoundary::index_of(i + off, nx);
                    if(i_bc < nx) OP::iop(y[i], x[i_bc]);
                }
            }else{
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = 0; i < ptrdiff_t(n1); ++i){ \
                        const size_t i_bc = name##Boundary::index_of(i + off, nx); \
                        if(i_bc < nx) OP::iop(y[i], x[i_bc]); \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
            }

            size_t k = n1;
            const size_t count = (n2 > n1)? n2 - n1: 0;
            if (count >= lanes){
                const size_t n_safe_load = count - lanes;
                for(size_t i = 0; i <= n_safe_load; k += lanes, i += lanes){
                    auto yi = hn::LoadU(dtag, y + k);
                    auto xi = hn::LoadU(dtag, x + k + off);
                    OP::iop_v(yi, xi);
                    hn::StoreU(yi, dtag, y + k);
                }
            }

            if (k < n2 ){
                // Handle the last vector
                const size_t rem = n2 - k;
                auto yi = hn::LoadN(dtag, y + k, rem);
                auto xi = hn::LoadN(dtag, x + k + off, rem);
                OP::iop_v(yi, xi);
                hn::StoreN(yi, dtag, y + k, rem);
            }

            if constexpr(!is_normal){

                for (ptrdiff_t i = n2; i < ptrdiff_t(ny); ++i){
                    const size_t i_bc = ZeroBoundary::index_of(i + off, nx);
                    if(i_bc < nx) OP::iop(y[i], x[i_bc]);
                }

                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = nd, j = ptrdiff_t(nd) + off; i < ptrdiff_t(nx) - off; ++i, ++j)  { \
                        size_t io = name##Boundary::index_of(i, ny); \
                        if ((io != ny) && (ptrdiff_t(io) != i)) \
                            OP::iop(y[io], x[j]); \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
            }else{
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = n2; i < ptrdiff_t(ny); ++i){ \
                        const size_t i_bc = name##Boundary::index_of(i + off, nx); \
                        if(i_bc < nx) OP::iop(y[i], x[i_bc]); \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
            }
		}


        template<TransformType TF>
        void operator()(
            const TF tf, const Across axis, const BoundaryCondition bc,
            T* HWY_RESTRICT s, T* HWY_RESTRICT d, const size_t ns, const size_t nd
        ) const {
            using OP = unit_op_from_tf_t<TF>;

            constexpr auto is_normal = NormalTransform<TF>;
            const auto target_update_type = [&]{
                if constexpr (is_normal) return UpdateType::UpdateOdds; else return UpdateType::UpdateEvens;
            }();
            const auto nf = [&]{if constexpr (is_normal) return step_dat.n_front; else return step_dat.n_front_r;}();
            const auto nb = [&]{if constexpr (is_normal) return step_dat.n_back; else return step_dat.n_back_r;}();
            const auto off = [&]{if constexpr (is_normal) return step_dat.offset; else return step_dat.offset_r;}();

			const size_t nx = (step_dat.update_type == target_update_type) ? ns : nd;
			const T* HWY_RESTRICT x = (step_dat.update_type == target_update_type) ? s : d;

			const size_t ny = (step_dat.update_type == target_update_type) ? nd : ns;
			T* HWY_RESTRICT y = (step_dat.update_type == target_update_type) ? d : s;

            const size_t n1 = std::min(nf, ny);
            const size_t n2 = (nd > nb)? nd - nb : 0;

            const VecTag<T> dtag;
            HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);

            if constexpr(!is_normal){
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = step_dat.offset, j = 0; (i < 0) && (j < ptrdiff_t(nx)); ++i, ++j)  { \
                        size_t io = name##Boundary::index_of(i, ny); \
                        if ((io != ny) && (ptrdiff_t(io) != i)){ \
                            auto yi = hn::Load(dtag, y + io * lanes); \
                            const auto xi = hn::Load(dtag, x + j * lanes); \
                            OP::iop_v(yi, xi); \
                            hn::Store(yi, dtag, y + io * lanes); \
                        } \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
                for (ptrdiff_t i = 0; i < ptrdiff_t(n1); ++i){
                    const size_t i_bc = ZeroBoundary::index_of(i + off, nx);
                    if(i_bc < nx){
                        auto yi = hn::Load(dtag, y + i * lanes);
                        auto xi = hn::Load(dtag, x + i_bc * lanes);
                        OP::iop_v(yi, xi);
                        hn::Store(yi, dtag, y + i * lanes);
                    }
                }
            }else{
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = 0; i < ptrdiff_t(n1); ++i){ \
                        const size_t i_bc = name##Boundary::index_of(i + off, nx); \
                        if(i_bc < nx){ \
                            auto yi = hn::Load(dtag, y + i * lanes); \
                            auto xi = hn::Load(dtag, x + i_bc * lanes); \
                            OP::iop_v(yi, xi); \
                            hn::Store(yi, dtag, y + i * lanes); \
                        } \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
            }

            for (size_t i = n1; i < n2; ++i){
                auto yi = hn::Load(dtag, y + i * lanes);
                auto xi = hn::Load(dtag, x + (i + off) * lanes);
                OP::iop_v(yi, xi);
                hn::Store(yi, dtag, y + i * lanes);
            }

            if constexpr(!is_normal){
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = nd, j = ptrdiff_t(nd) + off; i < ptrdiff_t(nx) - off; ++i, ++j) { \
                        size_t io = name##Boundary::index_of(i, ny); \
                        if ((io != ny) && (ptrdiff_t(io) != i)){ \
                            auto yi = hn::Load(dtag, y + io * lanes); \
                            const auto xi = hn::Load(dtag, x + j * lanes); \
                            OP::iop_v(yi, xi); \
                            hn::Store(yi, dtag, y + io * lanes); \
                        } \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
                for (ptrdiff_t i = n2; i < ptrdiff_t(ny); ++i){
                    const size_t i_bc = ZeroBoundary::index_of(i + off, nx);
                    if(i_bc < nx){
                        auto yi = hn::Load(dtag, y + i * lanes);
                        auto xi = hn::Load(dtag, x + i_bc * lanes);
                        OP::iop_v(yi, xi);
                        hn::Store(yi, dtag, y + i * lanes);
                    }
                }
            }else{
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = n2; i < ptrdiff_t(ny); ++i){ \
                        const size_t i_bc = name##Boundary::index_of(i + off, nx); \
                        if(i_bc < nx){ \
                            auto yi = hn::Load(dtag, y + i * lanes); \
                            auto xi = hn::Load(dtag, x + i_bc * lanes); \
                            OP::iop_v(yi, xi); \
                            hn::Store(yi, dtag, y + i * lanes); \
                        } \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
            }
		}
    };

    template<typename T, size_t N>
	class Step<RepeatUpdateStep<T, N>>{
    private:
        const RepeatUpdateStep<T, N>& step_dat;

    public:

        constexpr Step(const RepeatUpdateStep<T, N>& step_dat_) : step_dat(step_dat_){};
        
        template<TransformType TF>
        void operator()(
            const TF tf, const Along axis, const BoundaryCondition bc,
            T* HWY_RESTRICT s, T* HWY_RESTRICT d, const size_t ns, const size_t nd
        ) const {
            using OP = op_from_tf_t<TF>;
            const auto op = OP();

            constexpr auto is_normal = NormalTransform<TF>;
            const auto target_update_type = [&]{
                if constexpr (is_normal) return UpdateType::UpdateOdds; else return UpdateType::UpdateEvens;
            }();
            const auto nf = [&]{if constexpr (is_normal) return step_dat.n_front; else return step_dat.n_front_r;}();
            const auto nb = [&]{if constexpr (is_normal) return step_dat.n_back; else return step_dat.n_back_r;}();
            const auto off = [&]{if constexpr (is_normal) return step_dat.offset; else return step_dat.offset_r;}();

			const size_t nx = (step_dat.update_type == target_update_type) ? ns : nd;
			const T* HWY_RESTRICT x = (step_dat.update_type == target_update_type) ? s : d;

			const size_t ny = (step_dat.update_type == target_update_type) ? nd : ns;
			T* HWY_RESTRICT y = (step_dat.update_type == target_update_type) ? d : s;

            const size_t n1 = std::min(nf, ny);
            const size_t n2 = (nd > nb)? nd - nb : 0;

            const VecTag<T> dtag;
            HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);
        
            if constexpr(!is_normal){
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = step_dat.offset, j = 1 - ptrdiff_t(N); i < 0; ++i, ++j) { \
                        size_t io = name##Boundary::index_of(i, ny); \
                        if ((io != ny) && (ptrdiff_t(io) != i)) \
                            ip::inner_product_safe<N>(op, y[io], step_dat.val, x, j, nx); \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
                for (ptrdiff_t i = 0; i < ptrdiff_t(n1); ++i)
                    ip::inner_product<N>(op, ZeroBoundary(), y[i], step_dat.val, off, x, i, nx);
            }else{
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = 0; i < ptrdiff_t(n1); ++i) \
                        ip::inner_product<N>(op, name##Boundary(), y[i], step_dat.val, off, x, i, nx); \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
            }

            const auto vv = hn::Set(dtag, step_dat.val);

            size_t k = n1;
            const size_t count = (n2 > n1)? n2 - n1: 0;
            if (count >= lanes){
                const size_t n_safe_load = count - lanes;
                for(size_t i = 0; i <= n_safe_load; k += lanes, i += lanes){
                    auto yi = hn::LoadU(dtag, y + k);
                    ip::inner_product<N>(op, dtag, yi, vv, off, x, k);
                    hn::StoreU(yi, dtag, y + k);
                }
            }

            if (k < n2 ){
                // Handle the last vector
                const size_t rem = n2 - k;
                auto yi = hn::LoadN(dtag, y + k, rem);
                ip::inner_product_n<N>(op, dtag, yi, vv, off, x, k, rem);
                hn::StoreN(yi, dtag, y + k, rem);
            }

            if constexpr(!is_normal){

                for (ptrdiff_t i = n2; i < ptrdiff_t(ny); ++i)
                    ip::inner_product<N>(op, ZeroBoundary(), y[i], step_dat.val, off, x, i, nx);

                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = nd, j = ptrdiff_t(nd) + off; i < ptrdiff_t(nx) - off; ++i, ++j) { \
                        size_t io = name##Boundary::index_of(i, ny); \
                        if ((io != ny) && (ptrdiff_t(io) != i)) \
                            ip::inner_product_safe<N>(op, y[io], step_dat.val, x, j, nx); \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
            }else{
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = n2; i < ptrdiff_t(ny); ++i) \
                        ip::inner_product<N>(op, name##Boundary(), y[i], step_dat.val, off, x, i, nx); \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
            }
		}

        template<TransformType TF>
        void operator()(
            const TF tf, const Across axis, const BoundaryCondition bc,
            T* HWY_RESTRICT s, T* HWY_RESTRICT d, const size_t ns, const size_t nd
        ) const {
            using OP = op_from_tf_t<TF>;
            const auto op = OP();

            constexpr auto is_normal = NormalTransform<TF>;
            const auto target_update_type = [&]{
                if constexpr (is_normal) return UpdateType::UpdateOdds; else return UpdateType::UpdateEvens;
            }();
            const auto nf = [&]{if constexpr (is_normal) return step_dat.n_front; else return step_dat.n_front_r;}();
            const auto nb = [&]{if constexpr (is_normal) return step_dat.n_back; else return step_dat.n_back_r;}();
            const auto off = [&]{if constexpr (is_normal) return step_dat.offset; else return step_dat.offset_r;}();

			const size_t nx = (step_dat.update_type == target_update_type) ? ns : nd;
			const T* HWY_RESTRICT x = (step_dat.update_type == target_update_type) ? s : d;

			const size_t ny = (step_dat.update_type == target_update_type) ? nd : ns;
			T* HWY_RESTRICT y = (step_dat.update_type == target_update_type) ? d : s;

            const size_t n1 = std::min(nf, ny);
            const size_t n2 = (nd > nb)? nd - nb : 0;

            const VecTag<T> dtag;
            HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);

            const auto v = hn::Set(dtag, step_dat.val);

            if constexpr(!is_normal){
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = step_dat.offset, j = 1 - ptrdiff_t(N); i < 0; ++i, ++j) { \
                        size_t io = name##Boundary::index_of(i, ny); \
                        if ((io != ny) && (ptrdiff_t(io) != i)){ \
                            auto yi = hn::Load(dtag, y + io * lanes); \
                            ip::inner_product_safe<N>(op, dtag, yi, v, x, j, nx, lanes); \
                            hn::Store(yi, dtag, y + io * lanes); \
                        } \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
                for (ptrdiff_t i = 0; i < ptrdiff_t(n1); ++i){
                    auto yi = hn::Load(dtag, y + i * lanes);
                    ip::inner_product<N>(op, ZeroBoundary(), dtag, yi, v, off, x, i, nx, lanes);
                    hn::Store(yi, dtag, y + i * lanes);
                }
            }else{
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = 0; i < ptrdiff_t(n1); ++i){ \
                        auto yi = hn::Load(dtag, y + i * lanes); \
                        ip::inner_product<N>(op, name##Boundary(), dtag, yi, v, off, x, i, nx, lanes); \
                        hn::Store(yi, dtag, y + i * lanes); \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
            }

            for (size_t i = n1; i < n2; ++i){
                auto yi = hn::Load(dtag, y + i * lanes);
                ip::inner_product<N>(op, dtag, yi, v, off, x, i, lanes);
                hn::Store(yi, dtag, y + i * lanes);
            }

            if constexpr(!is_normal){
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = nd, j = ptrdiff_t(nd) + off; i < ptrdiff_t(nx) -off; ++i, ++j) { \
                        size_t io = name##Boundary::index_of(i, ny); \
                        if ((io != ny) && (ptrdiff_t(io) != i)){ \
                            auto yi = hn::Load(dtag, y + io * lanes); \
                            ip::inner_product_safe<N>(op, dtag, yi, v, x, j, nx, lanes); \
                            hn::Store(yi, dtag, y + io * lanes); \
                        } \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
                for (ptrdiff_t i = n2; i < ptrdiff_t(ny); ++i){
                    auto yi = hn::Load(dtag, y + i * lanes);
                    ip::inner_product<N>(op, ZeroBoundary(), dtag, yi, v, off, x, i, nx, lanes);
                    hn::Store(yi, dtag, y + i * lanes);
                }
            }else{
                switch(bc){
                    #define X(name, value) \
                    case BoundaryCondition::name : \
                    for (ptrdiff_t i = n2; i < ptrdiff_t(ny); ++i){ \
                        auto yi = hn::Load(dtag, y + i * lanes); \
                        ip::inner_product<N>(op, name##Boundary(), dtag, yi, v, off, x, i, nx, lanes); \
                        hn::Store(yi, dtag, y + i * lanes); \
                    } \
                    break;
                    LIFTED_BOUNDARY_CONDITIONS
                    #undef X
                    default:
                    throw std::invalid_argument("Unknown Boundary Condition");
                }
            }
		}
    };

    template<typename T>
    class Step<ScaleStep<T>>{
    private:
        const ScaleStep<T>& step_dat;

    public:
        constexpr Step(const ScaleStep<T>& step_dat_) : step_dat(step_dat_){}

        template<TransformType TF>
        void operator()(
            const TF tf, const Along axis, const BoundaryCondition bc,
            T* HWY_RESTRICT s, T* HWY_RESTRICT d, const size_t ns, const size_t nd
        ) const {
            
            constexpr auto is_forward = ForwardTransform<TF>;
            const T dm = [&]{if constexpr(is_forward) return step_dat.d_mul; else return step_dat.d_div;}();
            const T sm = [&]{if constexpr(is_forward) return step_dat.s_mul; else return step_dat.s_div;}();

            const VecTag<T> dtag;
            HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);

            const auto d_mul_v = hn::Set(dtag, dm);
            const auto s_mul_v = hn::Set(dtag, sm);

            size_t i_s = 0;
            size_t i_d = 0;
            if (nd >= lanes) {
                const size_t count = nd - lanes;
                for (size_t ii=0; ii <= count; ii += lanes, i_s += lanes, i_d += lanes) {
                    const auto di = hn::LoadU(dtag, d + ii);
                    const auto si = hn::LoadU(dtag, s + ii);
                    hn::StoreU(hn::Mul(di, d_mul_v), dtag, d + ii);
                    hn::StoreU(hn::Mul(si, s_mul_v), dtag, s + ii);
                }
            }
            if (i_d < nd) {
                // Handle the last d vector
                const size_t rem = nd - i_d;
                const auto di = hn::LoadN(dtag, d + i_d, rem);
                hn::StoreN(hn::Mul(di, d_mul_v), dtag, d + i_d, rem);
            }
            if (i_s < ns) {
                // Handle the last s vector
                const size_t rem = ns - i_s;
                const auto si = hn::LoadN(dtag, s + i_s, rem);
                hn::StoreN(hn::Mul(si, s_mul_v), dtag, s + i_s, rem);
            }
        }

        template<TransformType TF>
        void operator()(
            const TF tf, const Across axis, const BoundaryCondition bc,
            T* HWY_RESTRICT s, T* HWY_RESTRICT d, const size_t ns, const size_t nd
        ) const {
            constexpr auto is_forward = ForwardTransform<TF>;
            const T dm = [&]{if constexpr(is_forward) return step_dat.d_mul; else return step_dat.d_div;}();
            const T sm = [&]{if constexpr(is_forward) return step_dat.s_mul; else return step_dat.s_div;}();

            const VecTag<T> dtag;
            HWY_LANES_CONSTEXPR size_t lanes = hn::Lanes(dtag);

            const auto d_mul_v = hn::Set(dtag, dm);
            const auto s_mul_v = hn::Set(dtag, sm);

            for(size_t i = 0; i < nd; ++i){
                const auto di = hn::Load(dtag, d + i * lanes);
                const auto si = hn::Load(dtag, s + i * lanes);
                hn::Store(hn::Mul(di, d_mul_v), dtag, d + i * lanes);
                hn::Store(hn::Mul(si, s_mul_v), dtag, s + i * lanes);
            }
            if(ns > nd) {
                const auto si = hn::Load(dtag, s + (ns - 1) * lanes);
                hn::Store(hn::Mul(si, s_mul_v), dtag, s + (ns - 1) * lanes);
            }
        }
    };

    // Compiler Deduction Guides:
    // Deduction guides (needed to deduce template arguments)
    template<typename T, size_t N>
    Step(GenericUpdateStep<T, N>) -> Step<GenericUpdateStep<T, N>>;

    template<typename T, UpdateOperation PM>
    Step(UnitUpdateStep<T, PM>) -> Step<UnitUpdateStep<T, PM>>;

    template<typename T, size_t N>
	Step(RepeatUpdateStep<T, N>) -> Step<RepeatUpdateStep<T, N>>;

    template<typename T>
    Step(ScaleStep<T>) -> Step<ScaleStep<T>>;

}
}
}  // namespace lifted
HWY_AFTER_NAMESPACE();

#endif  // include guard