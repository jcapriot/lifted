#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "test_dwt.cpp"
#include "hwy/foreach_target.h"  // IWYU pragma: keep

#include <vector>
// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "lifted.hpp"
#include "lifted-inl.hpp"
#include "hwy/tests/test_util-inl.h"
#include "test_helpers.h"

#include <gtest/gtest.h>

#ifndef TEST_WVLT
#define TEST_WVLT lifted::Wavelet::Daubechies2
#endif
#ifndef TEST_BC
#define TEST_BC lifted::BoundaryCondition::Zero
#endif

using prec = double;

HWY_BEFORE_NAMESPACE();
namespace lifted{
namespace HWY_NAMESPACE{
    
    namespace dh = detail::HWY_NAMESPACE;

    void trans_valid(const Transform tf, std::span<prec> s, std::span<prec> d){
        using WVLT = detail::wavelet_from_enum_t<TEST_WVLT, prec>;
        using transform = dh::FixedTransform<WVLT>;

        switch(tf){
            #define X(name, value)\
            case Transform::name: \
                transform::apply(\
                    detail::name(), detail::Along(), \
                    TEST_BC, &s[0], &d[0], s.size(), d.size() \
                ); \
                break;
            LIFTED_TRANSFORM_TYPES
            #undef X
        }
    }
}
}
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace lifted{
HWY_EXPORT(trans_valid);

void transform(const Transform tf, std::span<prec> s, std::span<prec> d){
    HWY_DYNAMIC_DISPATCH(trans_valid)(tf, s, d);
}

class TestShapesAndAxes : public ::testing::TestWithParam<std::tuple<size_t, std::tuple<size_v, size_v>>> {};

INSTANTIATE_TEST_SUITE_P(
    LWTDriverTests,  // Instance name
    TestShapesAndAxes,
    ::testing::Combine(
        ::testing::Values(1, 3),
        ::testing::Values(
            std::make_tuple(size_v{ 31 }, size_v{ 0 }),
            std::make_tuple(size_v{ 32 }, size_v{ 0 }),
            std::make_tuple(size_v{ 32, 1 }, size_v{ 0 }),
            std::make_tuple(size_v{ 1, 32}, size_v{ 1 }),
            std::make_tuple(size_v{ 34, 31 }, size_v{ 1 }),
            std::make_tuple(size_v{ 23, 41 }, size_v{ 0 }),
            std::make_tuple(size_v{ 34, 31 }, size_v{ 1 , 0}),
            std::make_tuple(size_v{ 23, 41 }, size_v{ 0 , 1}),
            std::make_tuple(size_v{ 25, 24, 31 }, size_v{ 2, }),
            std::make_tuple(size_v{ 24, 23, 41 }, size_v{ 1 }),
            std::make_tuple(size_v{ 24, 23, 41 }, size_v{ 0 }),
            std::make_tuple(size_v{ 25, 24, 31 }, size_v{ 2 , 1}),
            std::make_tuple(size_v{ 24, 23, 41 }, size_v{ 1 , 0}),
            std::make_tuple(size_v{ 24, 23, 41 }, size_v{ 0 , 2}),
            std::make_tuple(size_v{ 23, 24, 31 }, size_v{ 2 , 1, 0})
        )
    )
);

class TestSingleDim : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {};

INSTANTIATE_TEST_SUITE_P(
    LWT1DDriverTests,  // Instance name
    TestSingleDim,
    ::testing::Combine(
        ::testing::Values(1, 3),
        ::testing::Values(31, 32)
    )
);

//Validate the single dim transforms against the forward transform function
TEST_P(TestSingleDim, ValidateLWTSeperable){
    const auto [level, len] = GetParam();
    auto levels = size_v(1, level);
    auto shape = size_v(1, len);
    auto axes = size_v(1, 0);

    const auto tf = Transform::Forward;

    prec rtol = 1E-8;
    prec atol = 0;

    // initialize inputs and outputs
    std::vector<prec> input(len);
    std::vector<prec> output(len);
    std::vector<prec> output_ref(len);

    auto strides = stride_v(1, 1);
    detail::fill_sin(input, -1000.0, 1000.0);

    size_t nd = len / 2;
    size_t ns = len - nd;

    auto in_span = std::span(input.data(), ns + nd);
    auto out_span = std::span(output_ref.data(), ns + nd);
    for(size_t lvl=0; lvl < level; ++lvl){

        auto a_s = hwy::AllocateAligned<prec>(ns);
        auto a_d = hwy::AllocateAligned<prec>(nd);
        auto s = std::span(a_s.get(), ns);
        auto d = std::span(a_d.get(), nd);

        detail::deinterleave(in_span, s, d);

        // assert the deinterleave did it's job
        for(size_t i = 0; i < ns; ++i){
            ASSERT_EQ(in_span[2 * i], s[i]);
        }
        for(size_t i = 0; i < nd; ++i){
            ASSERT_EQ(in_span[2 * i  + 1], d[i]);
        }

        transform(tf, s, d);

        detail::stack(s, d, out_span);

        // assert the stack did it's job
        for(size_t i = 0; i < ns; ++i){
            ASSERT_EQ(out_span[i], s[i]);
        }
        for(size_t i = 0; i < nd; ++i){
            ASSERT_EQ(out_span[ns + i], d[i]);
        }

        nd = ns / 2;
        ns = ns - nd;
        in_span = std::span(out_span.data(), ns + nd);
        out_span = std::span(out_span.data(), ns + nd);
    }

    lwt(TEST_WVLT, tf, TEST_BC, shape, strides, strides, axes, levels, input.data(), output.data());

    for (size_t i = 0; i < len; ++i) {
        EXPECT_NEAR(output[i], output_ref[i], atol + rtol * std::abs(output_ref[i]));
    }
}

TEST_P(TestSingleDim, ValidateLWT){
    const auto [level, len] = GetParam();
    auto shape = size_v(1, len);
    auto axes = size_v(1, 0);

    const auto tf = Transform::Forward;

    prec rtol = 1E-8;
    prec atol = 0;

    // initialize inputs and outputs
    std::vector<prec> input(len);
    std::vector<prec> output(len);
    std::vector<prec> output_ref(len);

    auto strides = stride_v(1, 1);
    detail::fill_sin(input, -1000.0, 1000.0);

    size_t nd = len / 2;
    size_t ns = len - nd;

    auto in_span = std::span(input.data(), ns + nd);
    auto out_span = std::span(output_ref.data(), ns + nd);
    for(size_t lvl=0; lvl < level; ++lvl){

        auto a_s = hwy::AllocateAligned<prec>(ns);
        auto a_d = hwy::AllocateAligned<prec>(nd);
        auto s = std::span(a_s.get(), ns);
        auto d = std::span(a_d.get(), nd);

        detail::deinterleave(in_span, s, d);

        // assert the deinterleave did it's job
        for(size_t i = 0; i < ns; ++i){
            ASSERT_EQ(in_span[2 * i], s[i]);
        }
        for(size_t i = 0; i < nd; ++i){
            ASSERT_EQ(in_span[2 * i  + 1], d[i]);
        }

        transform(tf, s, d);
        detail::stack(s, d, out_span);

        // assert the stack did it's job
        for(size_t i = 0; i < ns; ++i){
            ASSERT_EQ(out_span[i], s[i]);
        }
        for(size_t i = 0; i < nd; ++i){
            ASSERT_EQ(out_span[ns + i], d[i]);
        }

        nd = ns / 2;
        ns = ns - nd;
        in_span = std::span(out_span.data(), ns + nd);
        out_span = std::span(out_span.data(), ns + nd);
    }

    lwt(TEST_WVLT, tf, TEST_BC, shape, strides, strides, axes, level, input.data(), output.data());

    for (size_t i = 0; i < len; ++i) {
        EXPECT_NEAR(output[i], output_ref[i], atol + rtol * std::abs(output_ref[i]));
    }
}

//Validate LWT against the 1D transforms

TEST_P(TestShapesAndAxes, ValidateLWTSeperable) {
    size_t level = std::get<0>(GetParam());
    const auto& [shape, axes] = std::get<1>(GetParam());
    auto levels = size_v(shape.size(), level);

    const auto tf = Transform::Forward;

    size_t sz = detail::prod(shape);
    size_t ndim = shape.size();

    prec rtol = 1E-8;
    prec atol = 0;

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
    detail::fill_sin(input, -1524.421, 1643.32);

    prec* d_in = input.data();
    prec* d_out = output_ref.data();
    auto ain = detail::cndarr<prec>(d_in, shape, strides);
    auto aout = detail::ndarr<prec>(d_out, shape, strides);

    for (size_t iax = 0; iax < axes.size(); ++iax) {
        size_t ax = axes[iax];
        size_t len = shape[ax];

        const auto& tin(iax == 0 ? ain : aout);
        detail::multi_iter<> it(tin, aout, ax, 1);

        auto x = std::vector<prec>(len);

        while(it.remaining() > 0){
            it.advance(1);
            // copy in
            for(size_t i=0; i < len; ++i)
                x[i] = tin[it.iofs(i)];

            size_t nd = len / 2;
            size_t ns = len - nd;

            for (size_t lvl = 0; lvl < levels[iax]; ++lvl) {
                auto x_span = std::span(x.data(), ns + nd);
                auto s = std::vector<prec>(ns);
                auto d = std::vector<prec>(nd);

                detail::deinterleave(x_span, s, d);
                transform(tf, s, d);
                detail::stack(s, d, x_span);

                nd = ns / 2;
                ns = ns - nd;
            }
            // copy out
            for(size_t i=0; i < len; ++i)
                aout[it.oofs(i)] = x[i];
        }
    }

    lwt(TEST_WVLT, tf, TEST_BC, shape, strides, strides, axes, levels, input.data(), output.data());

    for (size_t i = 0; i < sz; ++i) {
        EXPECT_NEAR(output[i], output_ref[i], atol + rtol * std::abs(output_ref[i]));
    }
}

TEST_P(TestShapesAndAxes, ValidateLWT) {
    size_t level = std::get<0>(GetParam());
    const auto& [shape, axes] = std::get<1>(GetParam());

    const auto tf = Transform::Forward;

    size_t sz = detail::prod(shape);
    size_t ndim = shape.size();

    auto shape_ = size_v(shape);
    for(size_t i = 0; i < ndim; ++i) ASSERT_EQ(shape_[i], shape[i]);

    prec rtol = 1E-8;
    prec atol = 0;

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
    detail::fill_sin(input, -1412.32, 1204.42);

    const prec* d_in = input.data();
    prec* d_out = output_ref.data();

    for (size_t lvl = 0; lvl < level; ++lvl) {
        for (auto ax : axes) {
            auto ain = detail::cndarr<prec>(d_in, shape_, strides);
            auto aout = detail::ndarr<prec>(d_out, shape_, strides);
            detail::multi_iter<> it(ain, aout, ax, 1);

            size_t len = shape_[ax];
            size_t nd = len / 2;
            size_t ns = len - nd;

            auto x = std::vector<prec>(len);
            auto s = std::vector<prec>(ns);
            auto d = std::vector<prec>(nd);

            while(it.remaining() > 0){
                it.advance(1);
                // copy in
                for(size_t i=0; i<len; ++i) x[i] = ain[it.iofs(i)];

                detail::deinterleave(x, s, d);
                transform(tf, s, d);
                detail::stack(s, d, x);

                // copy out
                for(size_t i=0; i<len; ++i) aout[it.oofs(i)] = x[i];
            }
            d_in = d_out;
        }
        for (auto ax : axes) shape_[ax] = shape_[ax] - shape_[ax] / 2;
    }

    lwt(TEST_WVLT, tf, TEST_BC, shape, strides, strides, axes, level, input.data(), output.data());
    for (size_t i = 0; i < sz; ++i) {
        EXPECT_NEAR(output[i], output_ref[i], atol + rtol * std::abs(output_ref[i]));
    }
}


//Validate LWT Forward and Inverse are actually inverses of each other

TEST_P(TestShapesAndAxes, ValidateFwdInvLWTSeperable) {
    size_t level = std::get<0>(GetParam());
    const auto& [shape, axes] = std::get<1>(GetParam());
    auto levels = size_v(shape.size(), level);

    const auto tf = Transform::Forward;
    const auto ti = Transform::Inverse;

    size_t sz = detail::prod(shape);
    size_t ndim = shape.size();

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
    detail::fill_sin(x_0, -1000.0, 1001.0);

    lwt(TEST_WVLT, tf, TEST_BC, shape, strides, strides, axes, levels, x_0.data(), x_w.data());
    lwt(TEST_WVLT, ti, TEST_BC, shape, strides, strides, axes, levels, x_w.data(), x_1.data());

    prec rtol = 1E-7;
    prec atol = 0.0;

    for (size_t i = 0; i < sz; ++i) {
        prec thresh = atol + rtol * std::abs(x_0[i]);
        EXPECT_NEAR(x_0[i], x_1[i], thresh);
    }
}

TEST_P(TestShapesAndAxes, ValidateFwdInvLWT) {
    size_t level = std::get<0>(GetParam());
    const auto& [shape, axes] = std::get<1>(GetParam());

    const auto tf = Transform::Forward;
    const auto ti = Transform::Inverse;

    size_t sz = detail::prod(shape);
    size_t ndim = shape.size();

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
    detail::fill_sin(x_0, -1000.0, 1001.0);

    lwt(TEST_WVLT, tf, TEST_BC, shape, strides, strides, axes, level, x_0.data(), x_w.data());
    lwt(TEST_WVLT, ti, TEST_BC, shape, strides, strides, axes, level, x_w.data(), x_1.data());

    prec rtol = 1E-7;
    prec atol = 0.0;

    for (size_t i = 0; i < sz; ++i) {
        prec thresh = atol + rtol * std::abs(x_0[i]);
        EXPECT_NEAR(x_0[i], x_1[i], thresh);
    }
}


//Validate LWT ForwardAdjoint and InverseAdjoint are actually inverses of each other

TEST_P(TestShapesAndAxes, ValidateFwdInvLWTAdjointSeperable) {
    size_t level = std::get<0>(GetParam());
    const auto& [shape, axes] = std::get<1>(GetParam());
    auto levels = size_v(shape.size(), level);

    const auto tfT = Transform::ForwardAdjoint;
    const auto tiT = Transform::InverseAdjoint;

    size_t sz = detail::prod(shape);
    size_t ndim = shape.size();

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
    detail::fill_sin(x_0, -1000.0, 1001.0);

    lwt(TEST_WVLT, tfT, TEST_BC, shape, strides, strides, axes, levels, x_0.data(), x_w.data());
    lwt(TEST_WVLT, tiT, TEST_BC, shape, strides, strides, axes, levels, x_w.data(), x_1.data());

    prec rtol = 1E-7;
    prec atol = 0.0;

    for (size_t i = 0; i < sz; ++i) {
        prec thresh = atol + rtol * std::abs(x_0[i]);
        EXPECT_NEAR(x_0[i], x_1[i], thresh);
    }
}

TEST_P(TestShapesAndAxes, ValidateFwdInvLWTAdjoint) {
    size_t lvl = std::get<0>(GetParam());
    const auto& [shape, axes] = std::get<1>(GetParam());

    const auto tfT = Transform::ForwardAdjoint;
    const auto tiT = Transform::InverseAdjoint;

    size_t sz = detail::prod(shape);
    size_t ndim = shape.size();

    size_t level = lvl;

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
    detail::fill_sin(x_0, -1000.0, 1001.0);

    lwt(TEST_WVLT, tfT, TEST_BC, shape, strides, strides, axes, level, x_0.data(), x_w.data());
    lwt(TEST_WVLT, tiT, TEST_BC, shape, strides, strides, axes, level, x_w.data(), x_1.data());

    prec rtol = 1E-7;
    prec atol = 0.0;

    for (size_t i = 0; i < sz; ++i) {
        prec thresh = atol + rtol * std::abs(x_0[i]);
        EXPECT_NEAR(x_0[i], x_1[i], thresh);
    }
}

//Validate LWT Forward and ForwardAdjoint are actually adjoints
TEST_P(TestShapesAndAxes, ValidateLWTAdjointSeperableMultipleLevel) {
    size_t level = std::get<0>(GetParam());
    const auto& [shape, axes] = std::get<1>(GetParam());
    auto levels = size_v(shape.size(), level);

    const auto tf = Transform::Forward;
    const auto tfT = Transform::ForwardAdjoint;

    size_t sz = detail::prod(shape);
    size_t ndim = shape.size();

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
    detail::fill_sin(u, -11024.0, 1123.321);
    detail::fill_sin(v, -5698234.42, 2615.53);

    lwt(TEST_WVLT, tf, TEST_BC, shape, strides, strides, axes, levels, u.data(), f_u.data());
    lwt(TEST_WVLT, tfT, TEST_BC, shape, strides, strides, axes, levels, v.data(), ft_v.data());

    prec v1 = std::inner_product(v.begin(), v.end(), f_u.begin(), prec(0));
    prec v2 = std::inner_product(ft_v.begin(), ft_v.end(), u.begin(), prec(0));

    prec rtol = 1E-7;
    prec atol = 0.0;
    prec thresh = atol + rtol * std::abs(v2);
    EXPECT_NEAR(v1, v2, thresh);
}

TEST_P(TestShapesAndAxes, ValidateLWTAdjointMultipleLevel) {
    size_t lvl = std::get<0>(GetParam());
    const auto& [shape, axes] = std::get<1>(GetParam());

    const auto tf = Transform::Forward;
    const auto tfT = Transform::ForwardAdjoint;

    size_t sz = detail::prod(shape);
    size_t ndim = shape.size();

    size_t level = lvl;

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
    detail::fill_sin(u, -11024.0, 1123.321);
    detail::fill_sin(v, -5698234.42, 2615.53);

    lwt(TEST_WVLT, tf, TEST_BC, shape, strides, strides, axes, level, u.data(), f_u.data());
    lwt(TEST_WVLT, tfT, TEST_BC, shape, strides, strides, axes, level, v.data(), ft_v.data());

    prec v1 = std::inner_product(v.begin(), v.end(), f_u.begin(), prec(0));
    prec v2 = std::inner_product(ft_v.begin(), ft_v.end(), u.begin(), prec(0));

    prec rtol = 1E-7;
    prec atol = 0.0;
    prec thresh = atol + rtol * std::abs(v2);
    EXPECT_NEAR(v1, v2, thresh);
}


//Validate LWT Inverse and InverseAdjoint are actually adjoints

TEST_P(TestShapesAndAxes, ValidateILWTAdjointSeperable) {
    size_t level = std::get<0>(GetParam());
    const auto& [shape, axes] = std::get<1>(GetParam());
    auto levels = size_v(shape.size(), level);

    const auto ti = Transform::Inverse;
    const auto tiT = Transform::InverseAdjoint;

    size_t sz = detail::prod(shape);
    size_t ndim = shape.size();

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
    detail::fill_sin(u, -11024.0, 1123.321);
    detail::fill_sin(v, -5698234.42, 2615.53);

    lwt(TEST_WVLT, ti, TEST_BC, shape, strides, strides, axes, levels, u.data(), f_u.data());
    lwt(TEST_WVLT, tiT, TEST_BC, shape, strides, strides, axes, levels, v.data(), ft_v.data());

    prec v1 = std::inner_product(v.begin(), v.end(), f_u.begin(), prec(0));
    prec v2 = std::inner_product(ft_v.begin(), ft_v.end(), u.begin(), prec(0));

    prec rtol = 1E-7;
    prec atol = 0.0;
    prec thresh = atol + rtol * std::abs(v2);
    EXPECT_NEAR(v1, v2, thresh);
}

TEST_P(TestShapesAndAxes, ValidateILWTAdjoint) {
    size_t lvl = std::get<0>(GetParam());
    const auto& [shape, axes] = std::get<1>(GetParam());

    const auto ti = Transform::Inverse;
    const auto tiT = Transform::InverseAdjoint;

    size_t sz = detail::prod(shape);
    size_t ndim = shape.size();

    size_t level = lvl;

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
    detail::fill_sin(u, -11024.0, 1123.321);
    detail::fill_sin(v, -5698234.42, 2615.53);

    lwt(TEST_WVLT, ti, TEST_BC, shape, strides, strides, axes, level, u.data(), f_u.data());
    lwt(TEST_WVLT, tiT, TEST_BC, shape, strides, strides, axes, level, v.data(), ft_v.data());

    prec v1 = std::inner_product(v.begin(), v.end(), f_u.begin(), prec(0));
    prec v2 = std::inner_product(ft_v.begin(), ft_v.end(), u.begin(), prec(0));

    prec rtol = 1E-7;
    prec atol = 0.0;
    prec thresh = atol + rtol * std::abs(v2);
    EXPECT_NEAR(v1, v2, thresh);
}

}

#endif