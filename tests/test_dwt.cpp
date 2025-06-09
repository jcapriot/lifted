
#include <tuple>
#include <vector>
#include <span>

#include "wavelets.hpp"
#include "ndarray.hpp"
#include "test_helpers.h"

#include <gtest/gtest.h>

using namespace wavelets;
using ndarray::prod;
using namespace test_helpers;

class TestShapesAndAxes : public ::testing::TestWithParam<std::tuple<size_v, size_v>> {};


TEST_P(TestShapesAndAxes, ValidateDWTDriver) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    using trans = LiftingTransform<WVLT, BC>;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    size_t n_last = shape[ndim - 1];
    size_t n_other = sz / n_last;

    // initialize inputs and outputs
    std::vector<prec> input(sz);
    std::vector<prec> output(sz);
    std::vector<prec> output_ref(sz);

    auto strides = stride_v(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = 2; i <= ndim; ++i){
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }

    for (size_t i = 0; i < n_other; ++i) {
        auto in_span = std::span(input.data() + i * n_last, n_last);
        fill_sin(in_span, -10.0 * (i + 1), 10.0 * (i + 1));
    }

    prec* ain = input.data();
    prec* aout = output_ref.data();
    for (auto ax : axes) {
        size_t len = shape[ax];
        size_t nd = len / 2;
        size_t ns = len - nd;

        auto in_slices = all_arrays_along_axis(ain, shape, strides, ax);
        auto out_slices = all_arrays_along_axis(aout, shape, strides, ax);

        EXPECT_EQ(in_slices.size(), sz / len);

        std::vector<prec> s(ns);
        std::vector<prec> d(nd);
        
        for (size_t i = 0; i < in_slices.size(); ++i) {
            deinterleave(in_slices[i], s, d);
            trans::forward(s, d);
            stack(s, d, out_slices[i]);
        }
        ain = aout;
    }

    dwt<WVLT, BC>(shape, strides, strides, axes, input.data(), output.data(), 1);

    for (size_t i = 0; i < sz; ++i) {
        EXPECT_NE(output[i], output_ref[i]);
    }
}


INSTANTIATE_TEST_SUITE_P(
    DWTDriverTests,  // Instance name
    TestShapesAndAxes,
    ::testing::Values(
        std::make_tuple(size_v{31}, size_v{1}),
        std::make_tuple(size_v{32}, size_v{1}),
    )
);