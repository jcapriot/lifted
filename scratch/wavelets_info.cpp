#include <iostream>
#include <tuple>
#include <array>

#include "wavelets.hpp"
#include "ndarray.hpp"

#define TYPE_INFO(x) cout << #x << ": " << typeid(x).name() << endl

using namespace wavelets;
using namespace ndarray;
using namespace std;

template<typename Wavelet>
static void print_info() {

	using precision = typename Wavelet::type;
	using steps = typename Wavelet::steps;

	TYPE_INFO(Wavelet);
	TYPE_INFO(precision);
	TYPE_INFO(steps) << endl;

	using step0 = tuple_element_t<0, steps>;
	using step1 = tuple_element_t<1, steps>;
	using step2 = tuple_element_t<2, steps>;
	TYPE_INFO(typename step0::lifter);
	TYPE_INFO(typename step1::lifter);
	TYPE_INFO(typename step2::lifter) << endl;

	TYPE_INFO(typename step0::lifter_r);
	TYPE_INFO(typename step1::lifter_r);
	TYPE_INFO(typename step2::lifter_r);

	const size_t N = 32;
	constexpr size_t Ne = N / 2;
	constexpr size_t No = N - Ne;

	auto x = aligned_array<double>(N);
	auto s_x = aligned_array<double>(Ne);
	auto d_x = aligned_array<double>(No);

	for (auto i = 0; i < N; ++i)
		x[i] = 0.0;

	x[31] = 1;
	cout << x << endl;

	for (auto i = 0, j=0; i < Ne; ++i, j += 2) {
		s_x[i] = x[j];
		d_x[i] = x[j + 1];
	}
	LiftingTransform<Wavelet>::forward(s_x, d_x);

	cout << s_x << endl;
	cout << d_x << endl;

	for (auto i = 0, j = 0; i < Ne; ++i, j += 2) {
		s_x[i] = x[j];
		d_x[i] = x[j + 1];
	}
	LiftingTransform<Wavelet>::forward_adjoint(s_x, d_x);

	cout << s_x << endl;
	cout << d_x << endl;
	
}


int main(int argc, char* argv[]) {

	using T = double;

	print_info<Daubechies2<T>>();
}