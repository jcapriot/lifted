// Copyright 2019 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>  // abort
#include <iostream>

#include <cmath>  // std::abs
#include <memory>
#include <numeric>  // std::iota, std::inner_product
#include <array>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "highway_tests.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"
#include "hwy/nanobenchmark.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

namespace {

template<typename T>
class TwoArray {
 public:
  static size_t NumItems() { return 412331; }

  TwoArray() :
      inp_(AllocateAligned<T>(NumItems())),
      out_(AllocateAligned<T>(NumItems()))
  {
    // = 1, but compiler doesn't know
    const T init = static_cast<T>(Unpredictable1());
    std::iota(inp_.get(), inp_.get() + NumItems(), init);
  }

 protected:
  AlignedFreeUniquePtr<T[]> inp_;
  AlignedFreeUniquePtr<T[]> out_;
};

// Measures durations, verifies results, prints timings.
template <class Benchmark>
void RunBenchmark(const char* caption) {
  printf("%10s: ", caption);
  const size_t kNumInputs = 1;
  const size_t num_items = Benchmark::NumItems() * size_t(Unpredictable1());
  const FuncInput inputs[kNumInputs] = {num_items};
  Result results[kNumInputs];

  Benchmark benchmark;

  Params p;
  p.verbose = false;
  p.max_evals = 7;
  p.target_rel_mad = 0.002;
  const size_t num_results = MeasureClosure(
      [&benchmark](const FuncInput input) { return benchmark(input); }, inputs,
      kNumInputs, results, p);
  if (num_results != kNumInputs) {
    HWY_WARN("MeasureClosure failed.\n");
  }

  benchmark.Verify(num_items);

  for (size_t i = 0; i < num_results; ++i) {
    const double cycles_per_item =
        results[i].ticks / static_cast<double>(results[i].input);
    const double mad = results[i].variability * cycles_per_item;
    printf("%6d: %6.3f (+/- %5.3f)\n", static_cast<int>(results[i].input),
           cycles_per_item, mad);
  }
}

template<typename T=float>
class BenchmarkInterleaveVector : public TwoArray<T> {
 public:
  BenchmarkInterleaveVector() : TwoArray<T>() {}

  FuncOutput operator()(const size_t num_items) {

    const size_t nd = num_items / 2;
    const size_t ns = num_items - nd;

    const T* HWY_RESTRICT s = TwoArray<T>::inp_.get();
    const T* HWY_RESTRICT d = s + ns;

    T* HWY_RESTRICT dst = TwoArray<T>::out_.get();

    const ScalableTag<T> dtag;
    HWY_LANES_CONSTEXPR size_t lanes = Lanes(dtag);

    size_t ii = 0;
    if (nd >= lanes){
        size_t N = nd - lanes;
        for(size_t i=0; i <= N; i += lanes, ii += lanes){
          const auto evens = Load(dtag, s + i);
          const auto odds = LoadU(dtag, d + i);
          StoreInterleaved2(evens, odds, dtag, dst + i * 2);
        }
    }
    for(; ii < nd; ++ii){
      dst[ii * 2] = s[ii];
      dst[ii * 2 + 1] = d[ii];
    }
    if(ns > nd){
        dst[num_items - 1] = s[ns - 1];
    }
    return static_cast<FuncOutput>(dst[0]);
  }

  void Verify(size_t num_items) {
    const size_t nd = num_items / 2;
    const size_t ns = num_items - nd;

    auto ref = std::vector<T>(num_items);

    const T* HWY_RESTRICT s = TwoArray<T>::inp_.get();
    const T* HWY_RESTRICT d = s + ns;

    for(size_t i=0, ii=0; i<nd; ++i, ii += 2){
        ref[ii] = s[i];
        ref[ii+1] = d[i];
    }
    if(ns > nd){
      ref[num_items - 1] = s[ns - 1];
    }
    

    for(size_t i = 0; i < num_items; ++i) {
        T rel_err = std::abs(ref[i] - TwoArray<T>::out_[i]) / std::abs(ref[i]);
        
        if (rel_err > 1.1E-6f) {
            std::cout << "i=" << i << ": " << ref[i] << " =? " << TwoArray<T>::out_[i] << std::endl;
            HWY_ABORT("Interleave: expected %e actual %e (%e)\n", ref[i], TwoArray<T>::out_[i], rel_err);
         }
    }
  }

 private:
  std::vector<T> out_;  // for Verify
};

template<typename T=float>
class BenchmarkInterleaveScalar : public TwoArray<T> {
 public:
  BenchmarkInterleaveScalar() : TwoArray<T>() {}

  FuncOutput operator()(const size_t num_items) {

    const size_t nd = num_items / 2;
    const size_t ns = num_items - nd;

    const T* HWY_RESTRICT s = TwoArray<T>::inp_.get();
    const T* HWY_RESTRICT d = s + ns;

    T* HWY_RESTRICT dst = TwoArray<T>::out_.get();

    const ScalableTag<T> dtag;
    HWY_LANES_CONSTEXPR size_t lanes = Lanes(dtag);

    for(size_t i=0, ii=0; i<nd; ++i, ii += 2){
        dst[ii] = s[i];
        dst[ii+1] = d[i];
    }
    if(ns > nd){
      dst[num_items - 1] = s[ns - 1];
    }

    return static_cast<FuncOutput>(dst[0]);
  }

  void Verify(size_t num_items) {
    const size_t nd = num_items / 2;
    const size_t ns = num_items - nd;

    auto ref = std::vector<T>(num_items);

    const T* HWY_RESTRICT s = TwoArray<T>::inp_.get();
    const T* HWY_RESTRICT d = s + ns;

    for(size_t i=0, ii=0; i<nd; ++i, ii += 2){
        ref[ii] = s[i];
        ref[ii+1] = d[i];
    }
    if(ns > nd){
      ref[num_items - 1] = s[ns - 1];
    }
    

    for(size_t i = 0; i < num_items; ++i) {
        T rel_err = std::abs(ref[i] - TwoArray<T>::out_[i]) / std::abs(ref[i]);
        
        if (rel_err > 1.1E-6f) {
            std::cout << "i=" << i << ": " << ref[i] << " =? " << TwoArray<T>::out_[i] << std::endl;
            HWY_ABORT("Interleave: expected %e actual %e (%e)\n", ref[i], TwoArray<T>::out_[i], rel_err);
         }
    }
  }

 private:
  std::vector<T> out_;  // for Verify
};


void RunBenchmarks() {
  printf("------------------------ %s\n", TargetName(HWY_TARGET));
  RunBenchmark<BenchmarkInterleaveVector<float>>("Vector float");
  RunBenchmark<BenchmarkInterleaveVector<double>>("Vector double");
  RunBenchmark<BenchmarkInterleaveScalar<float>>("Scalar float");
  RunBenchmark<BenchmarkInterleaveScalar<double>>("Scalar double");
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
namespace {
HWY_EXPORT(RunBenchmarks);

void Run() {
  for (int64_t target : SupportedAndGeneratedTargets()) {
    SetSupportedTargetsForTest(target);
    HWY_DYNAMIC_DISPATCH(RunBenchmarks)();
  }
  SetSupportedTargetsForTest(0);  // Reset the mask afterwards.
}

}  // namespace
}  // namespace hwy

int main(int /*argc*/, char** /*argv*/) {
  hwy::Run();
  return 0;
}
#endif  // HWY_ONCE