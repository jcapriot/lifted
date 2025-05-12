

#include <array>
#include <stdexcept>
#include <iostream>
#include <typeinfo>
#include <tuple>
#include <numeric>

#include "wavelets.hpp"
#include "test_helpers.h"

#include <gtest/gtest.h>



using namespace wavelets;
using namespace test_helpers;
using std::tuple;


template<typename WVLT_, size_t N_>
struct ConfigSpec {
	using WVLT = WVLT_;
	using T = typename WVLT::type;
	constexpr static size_t N = N_;
};


template<typename T, size_t N>
using ConfigWavelets = tuple<
	//ConfigSpec<Daubechies1<T>, N>,
	//ConfigSpec<Daubechies2<T>, N>,
	//ConfigSpec<Daubechies3<T>, N>,
	//ConfigSpec<Daubechies4<T>, N>,
	//ConfigSpec<Daubechies5<T>, N>,
	//ConfigSpec<Daubechies6<T>, N>,
	ConfigSpec<Daubechies7<T>, N>,
	//ConfigSpec<BiorSpline3_1<T>, N>,
	//ConfigSpec<BiorSpline4_2<T>, N>,
	//ConfigSpec<BiorSpline3_1<T>, N>,
	//ConfigSpec<BiorSpline6_2<T>, N>,
	//ConfigSpec<ReverseBiorSpline3_1<T>, N>,
	//ConfigSpec<ReverseBiorSpline4_2<T>, N>,
	//ConfigSpec<ReverseBiorSpline3_1<T>, N>,
	//ConfigSpec<ReverseBiorSpline6_2<T>, N>,
	ConfigSpec<CDF9_7<T>, N>
	//ConfigSpec<ReverseCDF9_7<T>, N>
>;

template<size_t N>
using ConfigWaveletTypes = tuple_concat<
	//ConfigWavelets<float, N>,
	ConfigWavelets<double, N>
>;


// Test using an even and odd length
using AllTestWaveletTestTypes = tuple_concat<
	ConfigWaveletTypes<512>,
	ConfigWaveletTypes<423>
>;

// Helper: convert std::tuple<Ts...> to ::testing::Types<Ts...>
template <typename Tuple>
struct TupleToTypes;

template <typename... Ts>
struct TupleToTypes<std::tuple<Ts...>> {
	using type = ::testing::Types<Ts...>;
};

template <typename CONFIG>
class TestWavelet : public ::testing::Test {
public:
	using WVLT = typename CONFIG::WVLT;
	using T = typename WVLT::type;
	constexpr static size_t N = CONFIG::N;
};

TYPED_TEST_SUITE_P(TestWavelet);

// 2. Define tests
TYPED_TEST_P(TestWavelet, ValidateForwardBackward) {

	using WVLT = typename TestFixture::WVLT;
	using T = typename WVLT::type;

	T rtol = 1E-5;
	T atol = 1E-7;

	constexpr size_t N = TestFixture::N;
	constexpr size_t N_o = N / 2;
	constexpr size_t N_e = N - N_o;

	std::array<T, N> arr_in;
	std::array<T, N_e> arr_e;
	std::array<T, N_o> arr_o;
	std::array<T, N> arr_out;

	fill_sin(arr_in);

	deinterleave(arr_in, arr_e, arr_o);
	interleave(arr_e, arr_o, arr_out);

	LiftingTransform<WVLT>::forward(arr_e, arr_o);

	LiftingTransform<WVLT>::inverse(arr_e, arr_o);

	interleave(arr_e, arr_o, arr_out);

	for (size_t i = 0; i < arr_out.size(); ++i)
		EXPECT_NEAR(arr_out[i], arr_in[i], atol + rtol * std::abs(arr_in[i]));
}

TYPED_TEST_P(TestWavelet, ValidateAdjoint) {

	using WVLT = typename TestFixture::WVLT;
	using REV_WVLT = typename WVLT::reverse;
	using T = typename WVLT::type;

	T rtol = 1E-2;
	T atol = 0;

	constexpr size_t N = TestFixture::N;

	constexpr size_t N_o = N / 2;
	constexpr size_t N_e = N - N_o;

	std::array<T, N> u;
	std::array<T, N_e> u_s;
	std::array<T, N_o> u_d;

	std::array<T, N> v;
	std::array<T, N_e> v_s;
	std::array<T, N_o> v_d;

	fill_sin(u, -200, 200);
	fill_sin(v_s, -50, 50);
	fill_sin(v_d, -100, -50);
	deinterleave(u, u_s, u_d);

	LiftingTransform<WVLT>::forward(u_s, u_d);
	T v1 = std::inner_product(v_s.begin(), v_s.end(), u_s.begin(), 0.0)
		+ std::inner_product(v_d.begin(), v_d.end(), u_d.begin(), 0.0);


	LiftingTransform<REV_WVLT>::inverse(v_s, v_d);
	interleave(v_s, v_d, v);

	T v2 = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);

	EXPECT_NEAR(v1, v2, atol + rtol * std::abs(v2));
}

TYPED_TEST_P(TestWavelet, ValidateWvltDot) {

	using WVLT = typename TestFixture::WVLT;
	using REV_WVLT = typename WVLT::reverse;
	using T = typename WVLT::type;

	T rtol = 1E-3;
	T atol = 0;

	constexpr size_t N = TestFixture::N;

	constexpr size_t N_o = N / 2;
	constexpr size_t N_e = N - N_o;

	std::array<T, N> u;
	std::array<T, N_e> u_s;
	std::array<T, N_o> u_d;

	std::array<T, N> v;
	std::array<T, N_e> v_s;
	std::array<T, N_o> v_d;

	fill_rand(u, 52345, -200, 200);
	fill_rand(v, 744456, -50, 50);
	
	T v1 = std::inner_product(v.begin(), v.end(), u.begin(), 0.0);

	deinterleave(u, u_s, u_d);
	deinterleave(v, v_s, v_d);

	LiftingTransform<WVLT>::forward(u_s, u_d);
	LiftingTransform<REV_WVLT>::forward(v_s, v_d);

	T v2 = std::inner_product(v_s.begin(), v_s.end(), u_s.begin(), 0.0)
		+ std::inner_product(v_d.begin(), v_d.end(), u_d.begin(), 0.0);

	EXPECT_NEAR(v1, v2, atol + rtol * std::abs(v2));
}

REGISTER_TYPED_TEST_SUITE_P(TestWavelet,
	ValidateForwardBackward, ValidateAdjoint, ValidateWvltDot);

INSTANTIATE_TYPED_TEST_SUITE_P(PocketWavelets, TestWavelet, TupleToTypes<AllTestWaveletTestTypes>::type);

//int main() {
//
//	TestInvertible<32, AllWavelets>()();
//	TestInvertible<35, AllWavelets>()();
//	TestInvertible<64, AllWavelets>()();
//	TestInvertible<65, AllWavelets>()();
//	TestInvertible<500, AllWavelets>()();
//	TestInvertible<501, AllWavelets>()();
//	TestInvertible<1000, AllWavelets>()();
//	TestInvertible<1001, AllWavelets>()();
//}