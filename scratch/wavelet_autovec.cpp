#include <vector>
#include <numeric>

#include <benchmark/benchmark.h>
#include "wavelets.hpp"

template<typename T>
static void BM_Daubechies3Forward(benchmark::State& state) {
    using WVLT = wavelets::Daubechies3<T>;

    const size_t n = 100000 * 16;

    const size_t nd = n/2;
    const size_t ns = n - nd;

    std::vector<T> s(ns);
    std::vector<T> d(nd);

    std::iota(s.begin(), s.end(), T(0.0));
    std::iota(d.begin(), d.end(), T(0.0));

    for (auto _ : state){
      wavelets::LiftingTransform<WVLT>::forward(s.data(), d.data(), ns, nd);
      wavelets::LiftingTransform<WVLT>::inverse(s.data(), d.data(), ns, nd);
    }
}
// Register the function as a benchmark
BENCHMARK(BM_Daubechies3Forward<float>);
BENCHMARK(BM_Daubechies3Forward<double>);


template<typename T>
static void BM_Daubechies3ForwardVec(benchmark::State& state) {
    using WVLT = wavelets::Daubechies3<T>;

    const size_t n = 100000 * 16;
    const size_t lanes = wavelets::detail::VLEN<T>::val;

    const size_t n_sub = n / lanes;
    const size_t nd = n_sub/2;
    const size_t ns = n_sub - nd;

    std::vector<T> s(ns * lanes);
    std::vector<T> d(nd * lanes);

    std::iota(s.begin(), s.end(), 0.0f);
    std::iota(d.begin(), d.end(), 0.0f);

    for (auto _ : state){
        wavelets::LiftingTransform<WVLT>::forward(s.data(), d.data(), ns, nd, lanes, lanes, lanes);
        wavelets::LiftingTransform<WVLT>::inverse(s.data(), d.data(), ns, nd, lanes, lanes, lanes);
    }
}
// Register the function as a benchmark
BENCHMARK(BM_Daubechies3ForwardVec<float>);
BENCHMARK(BM_Daubechies3ForwardVec<double>);

BENCHMARK_MAIN();