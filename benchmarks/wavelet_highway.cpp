#include <vector>
#include <numeric>
#include <string>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "wavelet_highway.cpp"
#include "hwy/foreach_target.h"
#include "lifted-inl.hpp"

// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"

#include <benchmark/benchmark.h>

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
    using lifted::detail::HWY_NAMESPACE::FixedTransform;

    template<typename T>
    static void BM_Daubechies3Forward(benchmark::State& state) {
        using WVLT = lifted::Daubechies3<T>;
        using FOR = lifted::detail::Forward;
        using INV = lifted::detail::Inverse;
        using AXIS = lifted::detail::Along;

        constexpr static auto transform = FixedTransform<WVLT>();
        const static auto bc = lifted::BoundaryCondition::Zero;

        const size_t n = 100000 * 16;

        const size_t nd = n/2;
        const size_t ns = n - nd;

        auto s(AllocateAligned<T>(ns));
        auto d(AllocateAligned<T>(nd));

        std::iota(s.get(), s.get() + ns, T(0.0));
        std::iota(d.get(), d.get() + nd, T(0.0));

        for (auto _ : state){
            transform.apply(FOR(), AXIS(), bc, s.get(), d.get(), ns, nd);
            transform.apply(INV(), AXIS(), bc, s.get(), d.get(), ns, nd);
        }
    }

    template<typename T>
    static void BM_Daubechies3ForwardVec(benchmark::State& state) {
        using WVLT = lifted::Daubechies3<T>;
        using FOR = lifted::detail::Forward;
        using INV = lifted::detail::Inverse;
        using AXIS = lifted::detail::Across;

        constexpr static auto transform = FixedTransform<WVLT>();
        const static auto bc = lifted::BoundaryCondition::Zero;

        const ScalableTag<T> dtag;

        const size_t n = 100000 * 16;
        HWY_LANES_CONSTEXPR size_t lanes = Lanes(dtag);

        const size_t n_sub = n / lanes;
        const size_t nd = n_sub/2;
        const size_t ns = n_sub - nd;

        auto s(AllocateAligned<T>(ns * lanes));
        auto d(AllocateAligned<T>(nd * lanes));

        std::iota(s.get(), s.get() + ns * lanes, T(0.0));
        std::iota(d.get(), d.get() + nd * lanes, T(0.0));

        for (auto _ : state){
            transform.apply(FOR(), AXIS(), bc, s.get(), d.get(), ns, nd);
            transform.apply(INV(), AXIS(), bc, s.get(), d.get(), ns, nd);
        }
    }

    template<typename T>
    static auto get_forward_name(const bool vec){
        std::string func_name;
        if (vec){
            func_name = "Daubechies3::Forward::Vec";
        } else {
            func_name = "Daubechies3::Forward";
        }
        std::string target_name = TargetName(HWY_TARGET);
        std::string type_name;
        if constexpr(std::is_same_v(T, float)){
            type_name = "float";
        }else{
            type_name = "double";
        }

        auto name = target_name + "::" + func_name + "<" + type_name + ">";

        HWY_LANES_CONSTEXPR size_t lanes = Lanes(ScalableTag<T>());
        return name + " with " + std::to_string(lanes) + " lanes";
    }

BENCHMARK(BM_Daubechies3Forward<float>)->Name(get_forward_name<float>(false));
BENCHMARK(BM_Daubechies3ForwardVec<float>)->Name(get_forward_name<float>(true));

BENCHMARK(BM_Daubechies3Forward<double>)->Name(get_forward_name<double>(false));
BENCHMARK(BM_Daubechies3ForwardVec<double>)->Name(get_forward_name<double>(true));
}
}
HWY_AFTER_NAMESPACE();

// Register the function as a benchmark

#if HWY_ONCE
BENCHMARK_MAIN();
#endif