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

#include "wavelets.hpp"

namespace test_helpers {
	using std::ptrdiff_t;

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
		size_t ne = even.size();
		size_t no = odd.size();
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
		size_t ne = even.size();
		size_t no = odd.size();
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
		size_t ii = 0;
		for (size_t i = 0; i < n1; ++i, ++ii)
			arr[ii] = in1[i];
		for (size_t i = 0; i < n2; ++i, ++ii)
			arr[ii] = in2[i];
	}

	template<typename T>
	struct ArrayView {
		T* data;
		size_t length;
		ptrdiff_t stride;

		T& operator[](size_t idx) { return data[idx * stride]; }
		const T& operator[](size_t idx) const { return data[idx * stride]; }

		size_t size() const { return length; }
	};

	template<typename T>
	std::vector<ArrayView<T>> all_arrays_along_axis(
		T* data,
		const std::vector<size_t>& shape,
		const std::vector<ptrdiff_t>& strides,
		size_t axis)
	{

		std::vector<ArrayView<T>> lines;

		// Compute total number of outer points: product of all dims except axis
		size_t num_lines = 1;
		for (size_t i = 0; i < shape.size(); ++i) {
			if (i != axis) num_lines *= shape[i];
		}

		// Stride pattern for the outer axes
		std::vector<size_t> outer_shape, outer_strides;
		for (size_t i = 0; i < shape.size(); ++i) {
			if (i != axis) {
				outer_shape.push_back(shape[i]);
				outer_strides.push_back(strides[i]);
			}
		}

		// Iterate over all combinations of outer indices
		std::vector<size_t> idx(outer_shape.size(), 0);
		for (size_t i = 0; i < num_lines; ++i) {
			// Compute flat offset for outer index
			size_t offset = 0;
			for (size_t j = 0, k = 0; j < shape.size(); ++j) {
				if (j == axis) continue;
				offset += idx[k] * strides[j];
				++k;
			}

			lines.push_back(ArrayView<T>{
				data + offset,
					shape[axis],
					strides[axis]
			});

			// Increment multi-index
			for (int j = static_cast<int>(idx.size()) - 1; j >= 0; --j) {
				if (++idx[j] < outer_shape[j]) break;
				idx[j] = 0;
			}
		}

		return lines;
	}


	using wavelets::detail::update_s;
	using wavelets::detail::update_d;
	using wavelets::detail::scale;

	// Define some Mock wavelets for testing purposes.
	class WaveletSUpdate {
	public:
		using steps = std::tuple <
			update_s<-5, 1.2, 1.3, 1.5, -1.2, -3.2>
		>;
		constexpr static size_t n_steps = std::tuple_size<steps>::value;

		using type = double;
	};

	class WaveletDUpdate {
	public:
		using steps = std::tuple <
			update_d<-5, 0.5, 1.203, 4.2, 33.213>
		>;
		constexpr static size_t n_steps = std::tuple_size<steps>::value;

		using type = double;
	};

	class WaveletScale {
	public:
		using steps = std::tuple <
			scale<0.124>
		>;
		constexpr static size_t n_steps = std::tuple_size<steps>::value;

		using type = double;
	};

	class WaveletSDUpdate {
	public:
		using steps = std::tuple <
			update_s<-5, 1.2, 1.3, 1.5, -1.2, -3.2>,
			update_d<-5, 0.5, 1.203, 4.2, 33.213>
		>;
		constexpr static size_t n_steps = std::tuple_size<steps>::value;

		using type = double;
	};

	class WaveletSDUpdateScale {
	public:
		using steps = std::tuple <
			update_s<-5, 1.2, 1.3, 1.5, -1.2, -3.2>,
			update_d<-5, 1.2, 1.3, 1.5, -1.2, -3.2>,
			scale<0.124>
		>;
		constexpr static size_t n_steps = std::tuple_size<steps>::value;

		using type = double;
	};

	namespace detail {

		template<typename WVLT_, size_t N_, typename BC_>
		struct ConfigSpec {
			using WVLT = WVLT_;
			using T = typename WVLT::type;
			constexpr static size_t N = N_;
			using BC = BC_;
		};

		template<typename... Lists>
		struct tuple_concat;

		template<>
		struct tuple_concat<> {
			using type = std::tuple<>;
		};

		// Single list: return as is
		template<typename List>
		struct tuple_concat<List> {
			using type = List;
		};

		// Two or more lists: concatenate first two, then recurse
		template<typename... Ts1, typename... Ts2, typename... Rest>
		struct tuple_concat<std::tuple<Ts1...>, std::tuple<Ts2...>, Rest...> {
			using type = typename tuple_concat<std::tuple<Ts1..., Ts2...>, Rest...>::type;
		};

		// Helper to generate ConfigSpec for a given WVLT and BC over all N
		template<typename WVLT, typename BC, size_t... Ns>
		struct generate_for_wvlt_bc {
			using type = std::tuple<ConfigSpec<WVLT, Ns, BC>...>;
		};

		// Generate for all BCs
		template<typename WVLT, typename NSeq, typename... BCs>
		struct generate_for_wvlt;

		template<typename WVLT, size_t... Ns, typename... BCs>
		struct generate_for_wvlt<WVLT, std::integer_sequence<size_t, Ns...>, BCs...> {
		private:
			template<typename BC>
			using one = typename generate_for_wvlt_bc<WVLT, BC, Ns...>::type;

		public:
			using type = typename tuple_concat<one<BCs>...>::type;
		};

		// Generate for all WVLTs
		template<typename WVLTs, typename NSeq, typename BCs>
		struct generate_all;

		template<typename... WVLTs, size_t... Ns, typename... BCs>
		struct generate_all<std::tuple<WVLTs...>, std::integer_sequence<size_t, Ns...>, std::tuple<BCs...>> {
		private:
			template<typename WVLT>
			using one = typename generate_for_wvlt<WVLT, std::integer_sequence<size_t, Ns...>, BCs...>::type;

		public:
			using type = typename tuple_concat<one<WVLTs>...>::type;
		};

	}

	using detail::tuple_concat;
	using detail::generate_all;

	using TestWavelets = std::tuple<
		WaveletSUpdate,
		WaveletDUpdate,
		WaveletScale,
		WaveletSDUpdate,
		WaveletSDUpdateScale
	>;

	template<typename T>
	using AllWavelets = std::tuple <
		wavelets::Haar<T>,
		wavelets::Daubechies1<T>,
		wavelets::Daubechies2<T>,
		wavelets::Daubechies3<T>,
		wavelets::Daubechies4<T>,
		wavelets::Daubechies5<T>,
		wavelets::Daubechies6<T>,
		wavelets::Daubechies7<T>,
		wavelets::BiorSpline3_1<T>,
		wavelets::BiorSpline4_2<T>,
		wavelets::BiorSpline2_4<T>,
		wavelets::BiorSpline6_2<T>,
		wavelets::ReverseBiorSpline3_1<T>,
		wavelets::ReverseBiorSpline4_2<T>,
		wavelets::ReverseBiorSpline2_4<T>,
		wavelets::ReverseBiorSpline6_2<T>,
		wavelets::CDF5_3<T>,
		wavelets::ReverseCDF5_3<T>,
		wavelets::CDF9_7<T>,
		wavelets::ReverseCDF9_7<T>
	> ;

	using AllBoundaryConditions = std::tuple<
		wavelets::ZeroBoundary,
		wavelets::ConstantBoundary,
		wavelets::PeriodicBoundary,
		wavelets::SymmetricBoundary,
		wavelets::ReflectBoundary
	>;
}
