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

namespace test_helpers {

	template<typename VT, typename T = typename VT::value_type>
	static void fill_sin(VT& arr, const T x0=-10, const T xF=10) {
		size_t n = arr.size();
		T dx = (xF - x0) / (n - 1);
		for (size_t i = 0; i < n; ++i)
			arr[i] = std::sin(x0 + dx * i);
	}

	template<typename VT, typename T = typename VT::value_type>
	static void fill_rand(VT& arr, const unsigned int seed=0, const T min=0, const T max=1) {
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
			arr[j+1] = odd[i];
		}
		if (no != ne) {
			arr[arr.size() - 1] = even[ne - 1];
		}
	}

	template<typename VT, typename T = typename VT::value_type>
		static void assert_allclose(
			const VT& actual, const VT& desired,
			const T rtol = 1E-7, const T atol = 0.0,
			const bool equal_nan = true, const bool equal_inf = true
		) {
		size_t n = actual.size();

		if (actual.size() != desired.size()) {
			throw std::invalid_argument("Size mismatch: actual.size() = " + std::to_string(actual.size()) +
				", desired.size() = " + std::to_string(desired.size()));
		}


		T max_rel = 0.0;
		T max_abs = 0.0;

		size_t n_mismatch = 0;

		for (size_t i=0; i < n; ++i) {
			T ai = actual[i];
			T bi = desired[i];

			bool ai_nan = std::isnan(ai);
			bool bi_nan = std::isnan(bi);
			bool ai_inf = std::isinf(ai);
			bool bi_inf = std::isinf(bi);

			if (ai_nan || bi_nan) {
				if (!(ai_nan && bi_nan && equal_nan)) {
					std::ostringstream oss;
					oss << "NaN mismatch at index " << i
						<< ": actual = " << ai << ", desired = " << bi;
					throw std::runtime_error(oss.str());
				}
				continue;
			}

			if (ai_inf || bi_inf) {
				if (equal_inf) {
					if (!(ai == bi)) {
						std::ostringstream oss;
						oss << "Inf mismatch at index " << i
							<< ": actual = " << ai << ", desired = " << bi;
						throw std::runtime_error(oss.str());
					}
				}
				else {
					std::ostringstream oss;
					oss << "Inf value encountered but equal_inf is false at index " << i
						<< ": actual = " << ai << ", desired = " << bi;
					throw std::runtime_error(oss.str());
				}
				continue;
			}
			T abs_diff = std::abs(ai - bi);
			T thresh = atol + rtol * std::abs(bi);
			if (abs_diff > thresh) {
				max_abs = std::max(max_abs, abs_diff);
				max_rel = std::max(max_rel, abs_diff / std::abs(bi));
				++n_mismatch;
			}
		}
		if (n_mismatch > 0) {
			std::ostringstream oss;
			oss << std::setprecision(15)
				<< "Mismatched values " << n_mismatch << " / " << desired.size() << std::endl
				<< "Max absolute difference among violations: " << max_abs << std::endl
				<< "Max relative difference among violations: " << max_rel << std::endl;
			throw std::runtime_error(oss.str());
		}
	}

	namespace detail {

		template<typename Tuple>
		struct ForEachType;

		template<typename... Args>
		struct ForEachType<std::tuple<Args...>> {
		};

		template<typename... Tuples>
		struct tuple_cat_type {
			using type = decltype(std::tuple_cat(std::declval<Tuples>()...));
		};
	}

	template<typename... Tuples>
	using tuple_concat = typename detail::tuple_cat_type<Tuples...>::type;

	using detail::ForEachType;
}
