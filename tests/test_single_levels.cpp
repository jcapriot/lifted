

#include <array>
#include <stdexcept>
#include <iostream>
#include <typeinfo>
#include <tuple>
#include <numeric>
#include <utility>

#include "wavelets.hpp"
#include "test_helpers.h"

#include <gtest/gtest.h>



using namespace wavelets;
using namespace test_helpers;


// Define type lists
using WVLTs = TestWavelets;
using BCs = BoundaryConditions;
using Ns = std::integer_sequence<size_t, 512, 331>;

using WaveletTestMatrix = typename generate_all<WVLTs, Ns, BCs>::type;

// Helper: convert std::tuple<Ts...> to ::testing::Types<Ts...>
template <typename Tuple>
struct TupleToTypes;

template <typename... Ts>
struct TupleToTypes<std::tuple<Ts...>> {
	using type = ::testing::Types<Ts...>;
};

template <typename CONFIG>
class TestTrasformAndAdjoint : public ::testing::Test {
public:
	using WVLT = typename CONFIG::WVLT;
	using T = typename CONFIG::T;
	constexpr static size_t N = CONFIG::N;
	using BC = typename CONFIG::BC;
};

TYPED_TEST_SUITE_P(TestTrasformAndAdjoint);

// 2. Define tests
TYPED_TEST_P(TestTrasformAndAdjoint, ValidateForwardBackward) {

	using WVLT = typename TestFixture::WVLT;
	using T = typename WVLT::type;
	using BC = typename TestFixture::BC;

	T rtol = 1E-7;
	T atol = 0;

	constexpr size_t N = TestFixture::N;
	constexpr size_t N_o = N / 2;
	constexpr size_t N_e = N - N_o;

	std::array<T, N> arr_in;
	std::array<T, N_e> arr_e;
	std::array<T, N_o> arr_o;
	std::array<T, N> arr_out;

	fill_sin(arr_in, -11, 13);

	deinterleave(arr_in, arr_e, arr_o);
	interleave(arr_e, arr_o, arr_out);

	LiftingTransform<WVLT, BC>::forward(arr_e, arr_o);

	LiftingTransform<WVLT, BC>::inverse(arr_e, arr_o);

	interleave(arr_e, arr_o, arr_out);

	for (size_t i = 0; i < arr_out.size(); ++i)
		EXPECT_NEAR(arr_out[i], arr_in[i], atol + rtol * std::abs(arr_in[i]));
}

TYPED_TEST_P(TestTrasformAndAdjoint, ValidateAdjointForwardBackward) {

	using WVLT = typename TestFixture::WVLT;
	using T = typename WVLT::type;
	using BC = typename TestFixture::BC;

	T rtol = 1E-7;
	T atol = 0;

	constexpr size_t N = TestFixture::N;
	constexpr size_t N_o = N / 2;
	constexpr size_t N_e = N - N_o;

	std::array<T, N> arr_in;
	std::array<T, N_e> arr_e;
	std::array<T, N_o> arr_o;
	std::array<T, N> arr_out;

	fill_sin(arr_in, -11, 13);

	deinterleave(arr_in, arr_e, arr_o);
	interleave(arr_e, arr_o, arr_out);

	LiftingTransform<WVLT, BC>::forward_adjoint(arr_e, arr_o);

	LiftingTransform<WVLT, BC>::inverse_adjoint(arr_e, arr_o);

	interleave(arr_e, arr_o, arr_out);

	for (size_t i = 0; i < arr_out.size(); ++i)
		EXPECT_NEAR(arr_out[i], arr_in[i], atol + rtol * std::abs(arr_in[i]));
}

TYPED_TEST_P(TestTrasformAndAdjoint, ValidateForwardAdjoint) {

	using WVLT = typename TestFixture::WVLT;
	using T = typename WVLT::type;
	using BC = typename TestFixture::BC;

	T rtol = 1E-7;
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

	LiftingTransform<WVLT, BC>::forward(u_s, u_d);
	T v1 = std::inner_product(v_s.begin(), v_s.end(), u_s.begin(), 0.0)
		+ std::inner_product(v_d.begin(), v_d.end(), u_d.begin(), 0.0);


	LiftingTransform<WVLT, BC>::forward_adjoint(v_s, v_d);
	interleave(v_s, v_d, v);

	T v2 = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);

	EXPECT_NEAR(v1, v2, atol + rtol * std::abs(v2));
}

TYPED_TEST_P(TestTrasformAndAdjoint, ValidateInverseAdjoint) {

	using WVLT = typename TestFixture::WVLT;
	using T = typename WVLT::type;
	using BC = typename TestFixture::BC;

	T rtol = 1E-7;
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

	fill_sin(u, -200, 101);
	fill_sin(v_s, -50, 51);
	fill_sin(v_d, -100, -51);
	deinterleave(u, u_s, u_d);

	LiftingTransform<WVLT, BC>::inverse(u_s, u_d);
	T v1 = std::inner_product(v_s.begin(), v_s.end(), u_s.begin(), 0.0)
		+ std::inner_product(v_d.begin(), v_d.end(), u_d.begin(), 0.0);


	LiftingTransform<WVLT, BC>::inverse_adjoint(v_s, v_d);
	interleave(v_s, v_d, v);

	T v2 = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);

	EXPECT_NEAR(v1, v2, atol + rtol * std::abs(v2));
}

TYPED_TEST_P(TestTrasformAndAdjoint, ValidateForInvAdjDot) {

	using WVLT = typename TestFixture::WVLT;
	using T = typename WVLT::type;
	using BC = typename TestFixture::BC;

	T rtol = 1E-7;
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
	fill_sin(v, -50, 10);
	deinterleave(u, u_s, u_d);
	deinterleave(v, v_s, v_d);

	T v1 = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);

	LiftingTransform<WVLT, BC>::forward(u_s, u_d);
	LiftingTransform<WVLT, BC>::inverse_adjoint(v_s, v_d);

	T v2 = std::inner_product(v_s.begin(), v_s.end(), u_s.begin(), 0.0)
		+ std::inner_product(v_d.begin(), v_d.end(), u_d.begin(), 0.0);

	EXPECT_NEAR(v1, v2, atol + rtol * std::abs(v2));
}

TYPED_TEST_P(TestTrasformAndAdjoint, ValidateInvForAdjDot) {

	using WVLT = typename TestFixture::WVLT;
	using T = typename WVLT::type;
	using BC = typename TestFixture::BC;

	T rtol = 1E-7;
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
	fill_sin(v, -50, 10);
	deinterleave(u, u_s, u_d);
	deinterleave(v, v_s, v_d);

	T v1 = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);

	LiftingTransform<WVLT, BC>::inverse(u_s, u_d);
	LiftingTransform<WVLT, BC>::forward_adjoint(v_s, v_d);

	T v2 = std::inner_product(v_s.begin(), v_s.end(), u_s.begin(), 0.0)
		+ std::inner_product(v_d.begin(), v_d.end(), u_d.begin(), 0.0);

	EXPECT_NEAR(v1, v2, atol + rtol * std::abs(v2));
}

REGISTER_TYPED_TEST_SUITE_P(TestTrasformAndAdjoint,
	ValidateForwardBackward,
	ValidateAdjointForwardBackward,
	ValidateForwardAdjoint,
	ValidateInverseAdjoint,
	ValidateForInvAdjDot,
	ValidateInvForAdjDot
);

INSTANTIATE_TYPED_TEST_SUITE_P(PocketWavelets, TestTrasformAndAdjoint, TupleToTypes<WaveletTestMatrix>::type);
