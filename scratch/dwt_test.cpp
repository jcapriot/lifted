
#include <vector>
#include <iostream>
#include <span>
#include "wavelets.hpp"
#include "test_helpers.h"

using namespace wavelets;
using namespace ndarray;

using std::cout;
using std::endl;

static void run_dwt_test(size_v& shape, size_v& axes){

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    using trans = LiftingTransform<WVLT, BC>;

    size_t sz = prod(shape);
    size_t ndim = shape.size();

    auto levels = size_v(shape.size(), 1);

    size_t n_last = shape[ndim - 1];
    size_t n_other = sz / n_last;

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

    for (size_t i = 0; i < n_other; ++i) {
        auto in_span = std::span(input.data() + i * n_last, n_last);
        test_helpers::fill_sin(in_span, -10.0, 10.0, i + 1.0);
    }

    prec* ain = input.data();
    prec* aout = output_ref.data();
    for (auto ax : axes) {
        size_t len = shape[ax];
        size_t nd = len / 2;
        size_t ns = len - nd;

        auto in_slices = test_helpers::all_arrays_along_axis(ain, shape, strides, ax);
        auto out_slices = test_helpers::all_arrays_along_axis(aout, shape, strides, ax);

        std::vector<prec> s(ns);
        std::vector<prec> d(nd);

        for (size_t i = 0; i < in_slices.size(); ++i) {
            test_helpers::deinterleave(in_slices[i], s, d);
            trans::forward(s, d);
            test_helpers::stack(s, d, out_slices[i]);
        }
        ain = aout;
    }

    dwt<WVLT, BC>(shape, strides, strides, axes, levels, input.data(), output.data());

    prec rtol = 1E-7;
    prec atol = 0.0;
    for (size_t i = 0; i < sz; ++i) {
        prec err = std::abs(output[i] - output_ref[i]);
        prec thresh = atol + rtol * std::abs(output_ref[i]);

        if (err > thresh)
            cout << i << ": " << output[i] << "!=" << output_ref[i] << endl;
    }
}

static void run_dwt_test(size_v& shape, size_v& axes, size_v& levels) {

    cout << "seperable multiple level transform" << endl;

    cout << shape[0] << endl;
    cout << axes[0] << endl;

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    using trans = LiftingTransform<WVLT, BC>;

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
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }

    for (size_t i = 0; i < n_other; ++i) {
        auto in_span = std::span(input.data() + i * n_last, n_last);
        test_helpers::fill_sin(in_span, -10.0, 10.0, i + 1.0);
    }

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

    prec rtol = 1E-7;
    prec atol = 0.0;
    for (size_t i = 0; i < sz; ++i) {
        prec err = std::abs(output[i] - output_ref[i]);
        prec thresh = atol + rtol * std::abs(output_ref[i]);

        if (err > thresh)
            cout << i << ": " << output[i] << "!=" << output_ref[i] << endl;
    }
}

static void run_dwt_test(size_v& shape, size_v& axes, size_t level) {

    cout << "multiple level transform" << endl;

    using prec = double;
    using WVLT = Daubechies2<prec>;
    using BC = ZeroBoundary;
    using trans = LiftingTransform<WVLT, BC>;

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
    for (size_t i = 2; i <= ndim; ++i) {
        size_t ir = ndim - i;
        strides[ir] = shape[ir + 1] * strides[ir + 1];
    }

    for (size_t i = 0; i < n_other; ++i) {
        auto in_span = std::span(input.data() + i * n_last, n_last);
        test_helpers::fill_sin(in_span, -10.0, 11.0, i + 1.0);
    }

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

    prec rtol = 1E-7;
    prec atol = 0.0;
    for (size_t i = 0; i < sz; ++i) {
        prec err = std::abs(output[i] - output_ref[i]);
        prec thresh = atol + rtol * std::abs(output_ref[i]);

        if (err > thresh) {
            cout << i << ": " << output[i] << "!=" << output_ref[i] << endl;
        }
    }
}

int main(int argc, char* argv[]) {

    using T = double;

    auto shape = size_v {15, 31 };
    auto axes = size_v{ 0, 1 };
    // single level
    run_dwt_test(shape, axes);

    // seperable transform
    auto levels = size_v{ 2 , 2};
    run_dwt_test(shape, axes, levels);

    // non-seperable transform
    run_dwt_test(shape, axes, 2);
}