#ifndef _LIFTED_TEST_HELPERS_H_
#define _LIFTED_TEST_HELPERS_H_
#include <limits>
#include <cmath>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>
#include <iomanip>
#include <iterator>
#include <type_traits>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "lifted.hpp"

namespace lifted {
namespace detail{

	template<typename VT, typename T = typename VT::value_type>
	static void fill_sin(VT& arr, const T x0 = -10, const T xF = 10, const T scale = 1) {
		size_t n = arr.size();
		T dx = (xF - x0) / (n - 1);
		for (size_t i = 0; i < n; ++i)
			arr[i] = std::sin(x0 + dx * i) * scale;
	}

	template<typename VT, typename T = typename VT::value_type>
	static void fill_rand(VT& arr, const unsigned int seed = 0, const T min = 0, const T max = 1) {
		static_assert(std::is_arithmetic_v<T>, "Container value type must be arithmetic");

		std::mt19937 rng(seed);

		// Select appropriate distribution based on value type
		if constexpr (std::is_integral_v<T>) {
			std::uniform_int_distribution<T> dist(min, max);
			for (size_t i = 0; i < arr.size(); ++i)
				arr[i] = dist(rng);
		}
		else if constexpr (std::is_floating_point_v<T>) {
			std::uniform_real_distribution<T> dist(min, max);
			for (size_t i = 0; i < arr.size(); ++i)
				arr[i] = dist(rng);
		}
	}

	template<typename VT, typename TE, typename TO>
	static void deinterleave(const VT& arr, TE& even, TO& odd) {
		const size_t n = arr.size();
		const size_t ne = even.size();
		const size_t no = odd.size();
		if(no != n / 2){
			throw std::invalid_argument("incompatible lengths of arr and evens for deinterleave.");
		}else if(ne != n - no){
			throw std::invalid_argument("incompatible lengths of arr and odds for deinterleave.");
		}
		for (size_t i = 0, j = 0; i < no; ++i, j += 2) {
			even[i] = arr[j];
			odd[i] = arr[j + 1];
		}
		if (no != ne) {
			even[ne - 1] = arr[arr.size() - 1];
		}
	}

	template<typename TE, typename TO, typename VT>
	static void interleave(const TE& even, const TO& odd, VT& arr) {
		const size_t n = arr.size();
		const size_t ne = even.size();
		const size_t no = odd.size();
		if(no != n / 2){
			throw std::invalid_argument("incompatible lengths of arr and evens for interleave.");
		}else if(ne != n - no){
			throw std::invalid_argument("incompatible lengths of arr and odds for interleave.");
		}
		for (size_t i = 0, j = 0; i < no; ++i, j += 2) {
			arr[j] = even[i];
			arr[j + 1] = odd[i];
		}
		if (no != ne) {
			arr[arr.size() - 1] = even[ne - 1];
		}
	}

	template<typename TE, typename TO, typename VT>
	static void stack(const TE& in1, const TO& in2, VT& arr) {
		size_t n1 = in1.size();
		size_t n2 = in2.size();
		size_t n = arr.size();
		if(n1 + n2 != n){
			throw std::invalid_argument("incompatible lengths of in1, in2 and arr for stack.");
		}
		size_t ii = 0;
		for (size_t i = 0; i < n1; ++i, ++ii)
			arr[ii] = in1[i];
		for (size_t i = 0; i < n2; ++i, ++ii)
			arr[ii] = in2[i];
	}

	// Define some Mock wavelets for testing purposes.
	template<typename T>
	class WaveletSUpdateZeroOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			update_s<T>(0, 1.2)
		);
	};

	template<typename T>
	class WaveletSUpdatePosOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			update_s<T>(3, 1.2, 1.5, 1.6)
		);
	};

	template<typename T>
	class WaveletSUpdateNegOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			update_s<T>(-5, 1.2, 1.5, 1.6)
		);
	};

	template<typename T>
	class WaveletSUpdate {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			update_s<T>(-2, 1.2, 1.3, 1.5, -1.2, -3.2)
		);
	};

	template<typename T>
	class WaveletDUpdateZeroOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			update_d<T>(0, 0.5)
		);
	};

	template<typename T>
	class WaveletDUpdatePosOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			update_d<T>(3, 1.2, 1.5, 1.6)
		);
	};

	template<typename T>
	class WaveletDUpdateNegOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			update_d<T>(-5, 1.2, 1.5, 1.6)
		);
	};

	template<typename T>
	class WaveletDUpdate {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			update_d<T>(-2, 0.5, 1.203, 4.2, 33.213)
		);
	};

	template<typename T>
	class WaveletSUnitUpdateZeroOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			unit_update_s<T, UpdateOperation::add>(0)
		);
	};

	template<typename T>
	class WaveletSUnitUpdatePosOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			unit_update_s<T, UpdateOperation::sub>(3)
		);
	};

	template<typename T>
	class WaveletSUnitUpdateNegOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			unit_update_s<T, UpdateOperation::add>(-3)
		);
	};

	template<typename T>
	class WaveletDUnitUpdateZeroOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			unit_update_d<T, UpdateOperation::sub>(0)
		);
	};

	template<typename T>
	class WaveletDUnitUpdatePosOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			unit_update_d<T, UpdateOperation::add>(3)
		);
	};

	template<typename T>
	class WaveletDUnitUpdateNegOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			unit_update_d<T, UpdateOperation::sub>(-3)
		);
	};

	template<typename T>
	class WaveletSRepeatUpdateZeroOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			repeat_update_s<T, 1>(0, 0.5)
		);
	};

	template<typename T>
	class WaveletSRepeatUpdatePosOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			repeat_update_s<T, 3>(2, 0.5)
		);
	};

	template<typename T>
	class WaveletSRepeatUpdateNegOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			repeat_update_s<T, 3>(-4, 0.5)
		);
	};

	template<typename T>
	class WaveletSRepeatUpdate {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			repeat_update_s<T, 3>(-1, 0.5)
		);
	};

	template<typename T>
	class WaveletDRepeatUpdateZeroOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			repeat_update_d<T, 1>(0, 0.5)
		);
	};

	template<typename T>
	class WaveletDRepeatUpdatePosOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			repeat_update_d<T, 3>(2, 0.5)
		);
	};

	template<typename T>
	class WaveletDRepeatUpdateNegOff {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			repeat_update_d<T, 3>(-4, 0.5)
		);
	};

	template<typename T>
	class WaveletDRepeatUpdate {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			repeat_update_d<T, 3>(-1, 0.5)
		);
	};

	template<typename T>
	class WaveletScale {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			ScaleStep<T>(0.124)
		);
	};

	template<typename T>
	class WaveletMultiScale {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			ScaleStep<T>(5.23, 0.124)
		);
	};
	
	template<typename T>
	class WaveletMultiStep {
	public:
		using type = T;
		constexpr static auto steps = std::make_tuple(
			update_s<type>(-5, 1.2, 1.3, 1.5, -1.2, -3.2),
			update_d<type>(-5, 0.5, 1.203, 4.2, 33.213),
			ScaleStep<type>(0.124)
		);
	};

	#ifndef TESTING_MOCKWAVELETS
	#define TESTING_MOCKWAVELETS \
		X(WaveletSUpdateZeroOff, 0)\
		X(WaveletSUpdateNegOff, 1)\
		X(WaveletSUpdatePosOff, 2)\
		X(WaveletSUpdate, 3)\
		X(WaveletDUpdateZeroOff, 4)\
		X(WaveletDUpdateNegOff, 5)\
		X(WaveletDUpdatePosOff, 6)\
		X(WaveletDUpdate, 7)\
		X(WaveletSUnitUpdateZeroOff, 8)\
		X(WaveletSUnitUpdatePosOff, 9)\
		X(WaveletSUnitUpdateNegOff, 10)\
		X(WaveletDUnitUpdateZeroOff, 11)\
		X(WaveletDUnitUpdatePosOff, 12)\
		X(WaveletDUnitUpdateNegOff, 13)\
		X(WaveletSRepeatUpdateZeroOff, 14)\
		X(WaveletSRepeatUpdateNegOff, 15)\
		X(WaveletSRepeatUpdatePosOff, 16)\
		X(WaveletSRepeatUpdate, 17)\
		X(WaveletDRepeatUpdateZeroOff, 18)\
		X(WaveletDRepeatUpdateNegOff, 19)\
		X(WaveletDRepeatUpdatePosOff, 20)\
		X(WaveletDRepeatUpdate, 21)\
		X(WaveletScale, 22)\
		X(WaveletMultiScale, 23)\
		X(WaveletMultiStep, 24)
	#endif
	#ifndef LIFTED_TESTING_N_MOCKWAVELETS
	#define LIFTED_TESTING_N_MOCKWAVELETS 25
	#endif

	enum class MockWavelet : std::uint32_t {
		#define X(name, value) name = value,
		TESTING_MOCKWAVELETS
		#undef X
	};

	template<MockWavelet WVLT, typename T> struct mock_wavelet_from_enum;

	#define X(name, value) \
	template<typename T> struct mock_wavelet_from_enum<MockWavelet::name, T> { \
		using type = name<T>; \
	};
	TESTING_MOCKWAVELETS
	#undef X


	template<MockWavelet WVLT, typename T>
	using mock_wavelet_from_enum_t = typename mock_wavelet_from_enum<WVLT, T>::type;

	enum class TestVecDir{
		Across = 0,
		Along = 1,
	};

	template<TestVecDir AX> struct test_vecdir_from_enum;
	template<> struct test_vecdir_from_enum<TestVecDir::Across>{using type=Across;};
	template<> struct test_vecdir_from_enum<TestVecDir::Along>{using type=Along;};
	
	template<TestVecDir AX>
	using test_vecdir_from_enum_t = typename test_vecdir_from_enum<AX>::type;


	constexpr static array mock_wavelet_enum_array{
		#define X(name, value) MockWavelet::name,
		TESTING_MOCKWAVELETS
		#undef X
    };

	constexpr static array bc_enum_array{
		#define X(name, value) BoundaryCondition::name,
		LIFTED_BOUNDARY_CONDITIONS
		#undef X
    };

	using test_parameters = std::tuple<MockWavelet, BoundaryCondition, TestVecDir, size_t>;

	void PrintTo(const test_parameters& params, std::ostream* os) {

		const detail::MockWavelet wvlt = std::get<0>(params);
		const BoundaryCondition bc = std::get<1>(params);
		const detail::TestVecDir ax = std::get<2>(params);
		const size_t len = std::get<3>(params);

		std::string wvlt_name;
		switch(wvlt){
			#define X(name, value)\
			case MockWavelet::name: \
				wvlt_name = #name; \
				break;
			TESTING_MOCKWAVELETS
			#undef X
		}

		std::string bc_name;
		switch(bc){
			#define X(name, value)\
			case BoundaryCondition::name: \
				bc_name = #name; \
				break;
			LIFTED_BOUNDARY_CONDITIONS
			#undef X
		}

		std::string axis_name = (ax == TestVecDir::Across)? "Across" : "Along";
		std::string name = wvlt_name + "_" + bc_name + "_" + axis_name + "_" + std::to_string(len);
		*os << name;
	}

}
}
#endif