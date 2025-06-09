
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

    //for (size_t i = 0; i < n_other; ++i) {
    //    auto in_span = std::span(input.data() + i * n_last, n_last);
    //    test_helpers::fill_sin(in_span, -10.0, 10.0, i + 1.0);
    //}
    for (size_t i = 0; i < sz; ++i)
        input[i] = i;

    cout << "Input:" << endl;
    for (size_t i = 0; i < sz; ++i)
        cout << "i:" << input[i] << endl;

    prec* ain = input.data();
    prec* aout = output_ref.data();
    for (auto ax : axes) {
        size_t len = shape[ax];
        size_t nd = len / 2;
        size_t ns = len - nd;

        auto in_slices = test_helpers::all_arrays_along_axis(ain, shape, strides, ax);
        auto out_slices = test_helpers::all_arrays_along_axis(aout, shape, strides, ax);

        cout << "ax: " << ax << endl;
        cout << "n_along_ax=" << in_slices.size() << endl;

        std::vector<prec> s(ns);
        std::vector<prec> d(nd);

        for (size_t i = 0; i < in_slices.size(); ++i) {
            test_helpers::deinterleave(in_slices[i], s, d);
            for (size_t j = 0; j < nd; ++j) {
                cout << "s[" << j << "], d[" << j << "] = " << s[j] << ", " << d[j] << endl;
            }
            if (nd < ns)
                cout << "s[" << nd << "] = " << s[nd] << endl;
            //trans::forward(s, d);
            test_helpers::stack(s, d, out_slices[i]);
        }
        ain = aout;
    }

    dwt<WVLT, BC>(shape, strides, strides, axes, input.data(), output.data(), 1);

    for (size_t i = 0; i < sz; ++i) {
        cout << i << ": " << output[i] << "=?" << output_ref[i] << endl;
    }
}

int main(int argc, char* argv[]) {

    using T = double;

    auto shape = size_v{10, 8 };
    auto axes = size_v{ 1 };
    run_dwt_test(shape, axes);
}