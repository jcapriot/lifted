
#include <tuple>
#include <vector>
#include <span>
#include <numeric>

#include "wavelets.hpp"
#include "ndarray.hpp"
#include "test_helpers.h"

#include <gtest/gtest.h>

using namespace wavelets;
using ndarray::prod;
using namespace test_helpers;

class TestShapesAndAxes : public ::testing::TestWithParam<std::tuple<size_v, size_v>> {};


TEST_P(TestShapesAndAxes, ValidateDWTSingleLevel) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    using trans = LiftingTransform<WVLT, BC>;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();
    
    auto levels = size_v(shape.size(), 1);

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
;
    fill_sin(input, -1000.0, 1000.0);

    prec* ain = input.data();
    prec* aout = output_ref.data();
    for (auto ax : axes) {
        size_t len = shape[ax];
        size_t nd = len / 2;
        size_t ns = len - nd;

        auto in_slices = all_arrays_along_axis(ain, shape, strides, ax);
        auto out_slices = all_arrays_along_axis(aout, shape, strides, ax);

        ASSERT_EQ(in_slices.size(), sz / len);

        std::vector<prec> s(ns);
        std::vector<prec> d(nd);
        
        for (size_t i = 0; i < in_slices.size(); ++i) {
            deinterleave(in_slices[i], s, d);
            trans::forward(s, d);
            stack(s, d, out_slices[i]);
        }
        ain = aout;
    }

    dwt<WVLT, BC>(shape, strides, strides, axes, levels, input.data(), output.data());

    for (size_t i = 0; i < sz; ++i) {
        EXPECT_EQ(output[i], output_ref[i]);
    }
}


TEST_P(TestShapesAndAxes, ValidateDWTSeperableMultipleLevel) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    using trans = LiftingTransform<WVLT, BC>;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    auto levels = size_v(shape.size(), 3);

    // initialize inputs and outputs
    std::vector<prec> input(sz);
    std::vector<prec> output(sz);
    std::vector<prec> output_ref(sz);

    auto strides = stride_v(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }
    ;
    fill_sin(input, -1000.0, 1000.0);

    prec* ain = input.data();
    prec* aout = output_ref.data();
    for (size_t iax = 0; iax < axes.size(); ++iax) {
        size_t ax = axes[iax];
        size_t len = shape[ax];
        size_t nd = len / 2;
        size_t ns = len - nd;

        auto in_slices = test_helpers::all_arrays_along_axis(ain, shape, strides, ax);
        auto out_slices = test_helpers::all_arrays_along_axis(aout, shape, strides, ax);

        std::vector<prec> x(len);
        std::vector<prec> s(ns);
        std::vector<prec> d(nd);

        for (size_t i = 0; i < in_slices.size(); ++i) {
            // copy in
            for (size_t j = 0; j < len; ++j) x[j] = in_slices[i][j];
            nd = len / 2;
            ns = len - nd;

            for (size_t lvl = 0; lvl < levels[iax]; ++lvl) {
                auto x_t = std::span(x.data(), ns + nd);
                auto s_t = std::span(s.data(), ns);
                auto d_t = std::span(d.data(), nd);

                test_helpers::deinterleave(x_t, s_t, d_t);
                trans::forward(s_t, d_t);
                test_helpers::stack(s_t, d_t, x_t);

                nd = ns / 2;
                ns = ns - nd;
            }

            //copy out
            for (size_t j = 0; j < len; ++j) out_slices[i][j] = x[j];
        }
        ain = aout;
    }

    dwt<WVLT, BC>(shape, strides, strides, axes, levels, input.data(), output.data());

    for (size_t i = 0; i < sz; ++i) {
        EXPECT_EQ(output[i], output_ref[i]);
    }
}


TEST_P(TestShapesAndAxes, ValidateDWTMultipleLevel) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    using trans = LiftingTransform<WVLT, BC>;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    size_t level = 3;

    // initialize inputs and outputs
    std::vector<prec> input(sz);
    std::vector<prec> output(sz);
    std::vector<prec> output_ref(sz);

    auto strides = stride_v(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }
    ;
    fill_sin(input, -1000.0, 1001.0);

    auto shape_ = size_v(shape);
    prec* ain = input.data();
    prec* aout = output_ref.data();
    for (size_t lvl = 0; lvl < level; ++lvl) {

        for (auto ax : axes) {
            size_t len = shape_[ax];
            size_t nd = len / 2;
            size_t ns = len - nd;

            auto in_slices = test_helpers::all_arrays_along_axis(ain, shape_, strides, ax);
            auto out_slices = test_helpers::all_arrays_along_axis(aout, shape_, strides, ax);

            std::vector<prec> s(ns);
            std::vector<prec> d(nd);

            for (size_t i = 0; i < in_slices.size(); ++i) {
                test_helpers::deinterleave(in_slices[i], s, d);
                trans::forward(s, d);
                test_helpers::stack(s, d, out_slices[i]);
            }
            ain = aout;
        }

        for (auto ax : axes) shape_[ax] = shape_[ax] - shape_[ax] / 2;
    }

    dwt<WVLT, BC>(shape, strides, strides, axes, level, input.data(), output.data());

    for (size_t i = 0; i < sz; ++i) {
        EXPECT_EQ(output[i], output_ref[i]);
    }
}


TEST_P(TestShapesAndAxes, ValidateFwdInvDWTSingleLevel) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    auto levels = size_v(shape.size(), 1);

    // initialize inputs and outputs
    std::vector<prec> x_0(sz);
    std::vector<prec> x_w(sz);
    std::vector<prec> x_1(sz);

    auto strides = stride_v(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }
    fill_sin(x_0, -1000.0, 1001.0);

    dwt<WVLT, BC>(shape, strides, strides, axes, levels, x_0.data(), x_w.data());
    idwt<WVLT, BC>(shape, strides, strides, axes, levels, x_w.data(), x_1.data());

    prec rtol = 1E-7;
    prec atol = 0.0;

    for (size_t i = 0; i < sz; ++i) {
        prec thresh = atol + rtol * std::abs(x_0[i]);
        EXPECT_NEAR(x_0[i], x_1[i], thresh);
    }
}


TEST_P(TestShapesAndAxes, ValidateFwdInvDWTSeperableMultipleLevel) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    auto levels = size_v(shape.size(), 3);

    // initialize inputs and outputs
    std::vector<prec> x_0(sz);
    std::vector<prec> x_w(sz);
    std::vector<prec> x_1(sz);

    auto strides = stride_v(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }
    fill_sin(x_0, -1000.0, 1001.0);

    dwt<WVLT, BC>(shape, strides, strides, axes, levels, x_0.data(), x_w.data());
    idwt<WVLT, BC>(shape, strides, strides, axes, levels, x_w.data(), x_1.data());

    prec rtol = 1E-7;
    prec atol = 0.0;

    for (size_t i = 0; i < sz; ++i) {
        prec thresh = atol + rtol * std::abs(x_0[i]);
        EXPECT_NEAR(x_0[i], x_1[i], thresh);
    }
}


TEST_P(TestShapesAndAxes, ValidateFwdInvDWTMultipleLevel) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    size_t level = 3;

    // initialize inputs and outputs
    std::vector<prec> x_0(sz);
    std::vector<prec> x_w(sz);
    std::vector<prec> x_1(sz);

    auto strides = stride_v(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }
    fill_sin(x_0, -1000.0, 1001.0);

    dwt<WVLT, BC>(shape, strides, strides, axes, level, x_0.data(), x_w.data());
    idwt<WVLT, BC>(shape, strides, strides, axes, level, x_w.data(), x_1.data());

    prec rtol = 1E-7;
    prec atol = 0.0;

    for (size_t i = 0; i < sz; ++i) {
        prec thresh = atol + rtol * std::abs(x_0[i]);
        EXPECT_NEAR(x_0[i], x_1[i], thresh);
    }
}


TEST_P(TestShapesAndAxes, ValidateFwdInvDWTAdjointSingleLevel) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    auto levels = size_v(shape.size(), 1);

    // initialize inputs and outputs
    std::vector<prec> x_0(sz);
    std::vector<prec> x_w(sz);
    std::vector<prec> x_1(sz);

    auto strides = stride_v(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }
    fill_sin(x_0, -1000.0, 1001.0);

    dwt_adjoint<WVLT, BC>(shape, strides, strides, axes, levels, x_0.data(), x_w.data());
    idwt_adjoint<WVLT, BC>(shape, strides, strides, axes, levels, x_w.data(), x_1.data());

    prec rtol = 1E-7;
    prec atol = 0.0;

    for (size_t i = 0; i < sz; ++i) {
        prec thresh = atol + rtol * std::abs(x_0[i]);
        EXPECT_NEAR(x_0[i], x_1[i], thresh);
    }
}


TEST_P(TestShapesAndAxes, ValidateFwdInvDWTAdjointSeperableMultipleLevel) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    auto levels = size_v(shape.size(), 3);

    // initialize inputs and outputs
    std::vector<prec> x_0(sz);
    std::vector<prec> x_w(sz);
    std::vector<prec> x_1(sz);

    auto strides = stride_v(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }
    fill_sin(x_0, -1000.0, 1001.0);

    dwt_adjoint<WVLT, BC>(shape, strides, strides, axes, levels, x_0.data(), x_w.data());
    idwt_adjoint<WVLT, BC>(shape, strides, strides, axes, levels, x_w.data(), x_1.data());

    prec rtol = 1E-7;
    prec atol = 0.0;

    for (size_t i = 0; i < sz; ++i) {
        prec thresh = atol + rtol * std::abs(x_0[i]);
        EXPECT_NEAR(x_0[i], x_1[i], thresh);
    }
}


TEST_P(TestShapesAndAxes, ValidateFwdInvDWTAdjointMultipleLevel) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    size_t level = 3;

    // initialize inputs and outputs
    std::vector<prec> x_0(sz);
    std::vector<prec> x_w(sz);
    std::vector<prec> x_1(sz);

    auto strides = stride_v(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }
    fill_sin(x_0, -1000.0, 1001.0);

    dwt_adjoint<WVLT, BC>(shape, strides, strides, axes, level, x_0.data(), x_w.data());
    idwt_adjoint<WVLT, BC>(shape, strides, strides, axes, level, x_w.data(), x_1.data());

    prec rtol = 1E-7;
    prec atol = 0.0;

    for (size_t i = 0; i < sz; ++i) {
        prec thresh = atol + rtol * std::abs(x_0[i]);
        EXPECT_NEAR(x_0[i], x_1[i], thresh);
    }
}


TEST_P(TestShapesAndAxes, ValidateDWTAdjointSingleLevel) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    auto levels = size_v(shape.size(), 1);

    // initialize inputs and outputs
    std::vector<prec> u(sz);
    std::vector<prec> f_u(sz);
    std::vector<prec> v(sz);
    std::vector<prec> ft_v(sz);

    auto strides = stride_v(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }
    fill_sin(u, -11024.0, 1123.321);
    fill_sin(v, -5698234.42, 2615.53);

    dwt<WVLT, BC>(shape, strides, strides, axes, levels, u.data(), f_u.data());
    dwt_adjoint<WVLT, BC>(shape, strides, strides, axes, levels, v.data(), ft_v.data());

    prec v1 = std::inner_product(v.begin(), v.end(), f_u.begin(), prec(0));
    prec v2 = std::inner_product(ft_v.begin(), ft_v.end(), u.begin(), prec(0));

    prec rtol = 1E-7;
    prec atol = 0.0;
    prec thresh = atol + rtol * std::abs(v2);
    EXPECT_NEAR(v1, v2, thresh);
}


TEST_P(TestShapesAndAxes, ValidateDWTAdjointSeperableMultipleLevel) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    auto levels = size_v(shape.size(), 3);

    // initialize inputs and outputs
    std::vector<prec> u(sz);
    std::vector<prec> f_u(sz);
    std::vector<prec> v(sz);
    std::vector<prec> ft_v(sz);

    auto strides = stride_v(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }
    fill_sin(u, -11024.0, 1123.321);
    fill_sin(v, -5698234.42, 2615.53);

    dwt<WVLT, BC>(shape, strides, strides, axes, levels, u.data(), f_u.data());
    dwt_adjoint<WVLT, BC>(shape, strides, strides, axes, levels, v.data(), ft_v.data());

    prec v1 = std::inner_product(v.begin(), v.end(), f_u.begin(), prec(0));
    prec v2 = std::inner_product(ft_v.begin(), ft_v.end(), u.begin(), prec(0));

    prec rtol = 1E-7;
    prec atol = 0.0;
    prec thresh = atol + rtol * std::abs(v2);
    EXPECT_NEAR(v1, v2, thresh);
}


TEST_P(TestShapesAndAxes, ValidateDWTAdjointMultipleLevel) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    size_t level = 2;

    // initialize inputs and outputs
    std::vector<prec> u(sz);
    std::vector<prec> f_u(sz);
    std::vector<prec> v(sz);
    std::vector<prec> ft_v(sz);

    auto strides = stride_v(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }
    fill_sin(u, -11024.0, 1123.321);
    fill_sin(v, -5698234.42, 2615.53);

    dwt<WVLT, BC>(shape, strides, strides, axes, level, u.data(), f_u.data());
    dwt_adjoint<WVLT, BC>(shape, strides, strides, axes, level, v.data(), ft_v.data());

    prec v1 = std::inner_product(v.begin(), v.end(), f_u.begin(), prec(0));
    prec v2 = std::inner_product(ft_v.begin(), ft_v.end(), u.begin(), prec(0));

    prec rtol = 1E-7;
    prec atol = 0.0;
    prec thresh = atol + rtol * std::abs(v2);
    EXPECT_NEAR(v1, v2, thresh);
}


TEST_P(TestShapesAndAxes, ValidateIDWTAdjointSingleLevel) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    auto levels = size_v(shape.size(), 1);

    // initialize inputs and outputs
    std::vector<prec> u(sz);
    std::vector<prec> f_u(sz);
    std::vector<prec> v(sz);
    std::vector<prec> ft_v(sz);

    auto strides = stride_v(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }
    fill_sin(u, -11024.0, 1123.321);
    fill_sin(v, -5698234.42, 2615.53);

    idwt<WVLT, BC>(shape, strides, strides, axes, levels, u.data(), f_u.data());
    idwt_adjoint<WVLT, BC>(shape, strides, strides, axes, levels, v.data(), ft_v.data());

    prec v1 = std::inner_product(v.begin(), v.end(), f_u.begin(), prec(0));
    prec v2 = std::inner_product(ft_v.begin(), ft_v.end(), u.begin(), prec(0));

    prec rtol = 1E-7;
    prec atol = 0.0;
    prec thresh = atol + rtol * std::abs(v2);
    EXPECT_NEAR(v1, v2, thresh);
}


TEST_P(TestShapesAndAxes, ValidateIDWTAdjointSeperableMultipleLevel) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    auto levels = size_v(shape.size(), 3);

    // initialize inputs and outputs
    std::vector<prec> u(sz);
    std::vector<prec> f_u(sz);
    std::vector<prec> v(sz);
    std::vector<prec> ft_v(sz);

    auto strides = stride_v(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }
    fill_sin(u, -11024.0, 1123.321);
    fill_sin(v, -5698234.42, 2615.53);

    idwt<WVLT, BC>(shape, strides, strides, axes, levels, u.data(), f_u.data());
    idwt_adjoint<WVLT, BC>(shape, strides, strides, axes, levels, v.data(), ft_v.data());

    prec v1 = std::inner_product(v.begin(), v.end(), f_u.begin(), prec(0));
    prec v2 = std::inner_product(ft_v.begin(), ft_v.end(), u.begin(), prec(0));

    prec rtol = 1E-7;
    prec atol = 0.0;
    prec thresh = atol + rtol * std::abs(v2);
    EXPECT_NEAR(v1, v2, thresh);
}


TEST_P(TestShapesAndAxes, ValidateIDWTAdjointMultipleLevel) {

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    const auto& [shape, axes] = GetParam();

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    size_t level = 2;

    // initialize inputs and outputs
    std::vector<prec> u(sz);
    std::vector<prec> f_u(sz);
    std::vector<prec> v(sz);
    std::vector<prec> ft_v(sz);

    auto strides = stride_v(ndim);
    strides[ndim - 1] = 1;
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }
    fill_sin(u, -11024.0, 1123.321);
    fill_sin(v, -5698234.42, 2615.53);

    idwt<WVLT, BC>(shape, strides, strides, axes, level, u.data(), f_u.data());
    idwt_adjoint<WVLT, BC>(shape, strides, strides, axes, level, v.data(), ft_v.data());

    prec v1 = std::inner_product(v.begin(), v.end(), f_u.begin(), prec(0));
    prec v2 = std::inner_product(ft_v.begin(), ft_v.end(), u.begin(), prec(0));

    prec rtol = 1E-7;
    prec atol = 0.0;
    prec thresh = atol + rtol * std::abs(v2);
    EXPECT_NEAR(v1, v2, thresh);
}


INSTANTIATE_TEST_SUITE_P(
    DWTDriverTests,  // Instance name
    TestShapesAndAxes,
    ::testing::Values(
        std::make_tuple(size_v{ 31 }, size_v{ 0 }),
        std::make_tuple(size_v{ 32 }, size_v{ 0 }),
        std::make_tuple(size_v{ 32, 1 }, size_v{ 0 }),
        std::make_tuple(size_v{ 1, 32}, size_v{ 1 }),
        std::make_tuple(size_v{ 14, 31 }, size_v{ 1 }),
        std::make_tuple(size_v{ 23, 41 }, size_v{ 0 }),
        std::make_tuple(size_v{ 14, 31 }, size_v{ 1 , 0}),
        std::make_tuple(size_v{ 23, 41 }, size_v{ 0 , 1}),
        std::make_tuple(size_v{ 15, 14, 31 }, size_v{ 2 , }),
        std::make_tuple(size_v{ 14, 23, 41 }, size_v{ 1 }),
        std::make_tuple(size_v{ 14, 23, 41 }, size_v{ 0 }),
        std::make_tuple(size_v{ 15, 14, 31 }, size_v{ 2 , 1}),
        std::make_tuple(size_v{ 14, 23, 41 }, size_v{ 1 , 0}),
        std::make_tuple(size_v{ 14, 23, 41 }, size_v{ 0 , 2}),
        std::make_tuple(size_v{ 15, 14, 31 }, size_v{ 2 , 1, 0})
    )
);