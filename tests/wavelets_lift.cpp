

#include <vector>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <numeric>
#include <cstring>
#include <immintrin.h>
#include <cstdlib>

using std::vector;
using std::size_t;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

using std::cout;
using std::endl;

#include <algorithm>
#include <iterator>
#include <iostream>

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
	os << "{ ";
	std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(os, ", "));
	os << "}";
	return os;
}

#include "wavelets.hpp"
#include "ndarray.hpp"

using ndarray::aligned_array;
using ndarray::aligned_ndarray;

using namespace wavelets;



template<typename WVLT>
static void verify_transform(size_t n) {
	using T = typename WVLT::type;
	
	auto arr = aligned_array<T>(n);
	for (size_t i = 0; i < n; ++i)
		arr[i] = 0;
	arr[n / 2] = 1.0;
	//fill_arr(arr);

	size_t m = n / 2;
	auto evens = aligned_array<T>(m);
	auto odds = aligned_array<T>(m);
	fill_even_odd(arr, evens, odds);
	
	cout << "==========================" << endl;
	cout << "Verifying: " << typeid(WVLT).name() << endl
		<< "n = " << n << endl << endl
		<< "inputs" << endl
		<< "------" << endl << endl
		<< "arr: " << endl << arr << endl << endl;

	auto s = aligned_array<T>(m);
	auto d = aligned_array<T>(m);

	LiftingTransform<WVLT>::forward(
		evens.data(), odds.data(),
		evens.size()
	);
	cout << "forward" << endl
		<< "-------" << endl << endl
		<< "s: " << endl << evens << endl
		<< "d: " << endl << odds << endl << endl;

	LiftingTransform<WVLT>::inverse(
		evens.data(), odds.data(),
		evens.size()
	);

	auto arr2 = aligned_array<T>(n);
	for (size_t i = 0, j = 0; j < m; ++j, i += 2) {
		arr2[i] = evens[j];
		arr2[i + 1] = odds[j];
	}

	cout << "inverse" << endl
		<< "-------" << endl << endl
		<< "arr: " << endl << arr2 << endl << endl;
}

/*
template<typename WVLT>
static void verify_transform_vec(size_t n) {
	using T = typename WVLT::type;

	size_t v_len = wavelets_lift::detail::VLEN<T>::val;

	auto arr = aligned_array<T>(n * v_len);
	fill_arr(arr, v_len);

	size_t m = n / 2;
	auto evens = aligned_array<T>(m * v_len);
	auto odds = aligned_array<T>(m * v_len);
	fill_even_odd(arr, evens, odds, v_len);

	cout << "Verifying: " << typeid(WVLT).name() << endl
		<< "n = " << n << endl << endl
		<< "inputs" << endl
		<< "------" << endl << endl
		<< "arr: " << endl << arr << endl
		<< "evens: " << endl << evens << endl
		<< "odds: " << endl << odds << endl << endl;

	auto evens_out = aligned_array<T>(m * v_len);
	auto odds_out = aligned_array<T>(m * v_len);

	//WVLT::forward_lift(
	//	evens.data(), odds.data(),
	//	evens_out.data(), odds_out.data(),
	//	evens.size()
	//);
	//cout << "out-of-place forward" << endl
	//	<< "--------------------" << endl << endl
	//	<< "evens: " << endl << evens_out << endl
	//	<< "odds: " << endl << odds_out << endl << endl;

	//WVLT::inverse_lift(
	//	evens_out.data(), odds_out.data(),
	//	evens.data(), odds.data(),
	//	evens.size()
	//);
	//cout << "out-of-place inverse" << endl
	//	<< "--------------------" << endl << endl
	//	<< "evens: " << endl << evens << endl
	//	<< "odds: " << endl << odds << endl << endl;

	WVLT::forward_lift_vec(evens.data(), odds.data(), m);
	cout << "in-place forward" << endl
		<< "----------------" << endl << endl
		<< "evens: " << endl << evens << endl
		<< "odds: " << endl << odds << endl << endl;

	//WVLT::inverse_lift(evens.data(), odds.data(), evens.size());
	//cout << "in-place inverse" << endl
	//	<< "----------------" << endl << endl
	//	<< "evens: " << endl << evens << endl
	//	<< "odds: " << endl << odds << endl << endl;
}
*/

template<typename WVLT>
static void time_transform(size_t n, size_t n_repeats=1024) {
	using T = typename WVLT::type;
	size_t m = n / 2;

	auto arr_vec = std::vector<aligned_array<T>>(n_repeats);
	auto evens_vec = std::vector<aligned_array<T>>(n_repeats);
	auto odds_vec = std::vector<aligned_array<T>>(n_repeats);

	for (size_t i = 0; i < n_repeats; ++i) {
		auto arr = aligned_array<T>(n);

		fill_arr(arr);
		for (size_t j = 0; j < n; ++j) {
			arr[j] *= (i + 1);
		}

		auto evens = aligned_array<T>(m);
		auto odds = aligned_array<T>(m);
		fill_even_odd(arr, evens, odds);

		auto evens_out = aligned_array<T>(m);
		auto odds_out = aligned_array<T>(m);

		arr_vec[i] = arr;
		evens_vec[i] = evens;
		odds_vec[i] = odds;
	}


	cout << "==========================" << endl;
	cout << "Timing: " << typeid(WVLT).name() << endl;
	cout << "n = " << n << ", repeats = " << n_repeats << endl;

	auto tick = high_resolution_clock::now();
	auto tock = high_resolution_clock::now();
	duration<double, std::micro> ms_double = tock - tick;

	tick = high_resolution_clock::now();
	for (size_t i = 0; i < n_repeats; ++i)
		LiftingTransform<WVLT>::forward(evens_vec[i].data(), odds_vec[i].data(), evens_vec[i].size());
	tock = high_resolution_clock::now();

	ms_double = tock - tick;
	cout << "forward: " << ms_double.count() / n_repeats << "mu s" << endl;

	tick = high_resolution_clock::now();
	for (size_t i = 0; i < n_repeats; ++i)
		LiftingTransform<WVLT>::inverse(evens_vec[i].data(), odds_vec[i].data(), evens_vec[i].size());
	tock = high_resolution_clock::now();

	ms_double = tock - tick;
	cout << "inverse: " << ms_double.count() / n_repeats << "mu s" << endl;

	cout << endl;
}

#define is_aligned(POINTER, BYTE_COUNT) \
    (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)

/*
template<typename WVLT>
static void time_transform_vec(size_t n, size_t n_repeats = 1024) {


	using T = typename WVLT::type;

	size_t v_len = wavelets_lift::detail::VLEN<T>::val;
	size_t n_vec_repeats = n_repeats / v_len;

		auto arr = aligned_array<T>(n * v_len);
	fill_arr(arr, v_len);

	size_t m = n / 2;
	auto evens = aligned_array<T>(m * v_len);
	auto odds = aligned_array<T>(m * v_len);
	fill_even_odd(arr, evens, odds, v_len);

	auto evens_out = aligned_array<T>(m * v_len);
	auto odds_out = aligned_array<T>(m * v_len);

	cout << "Timing: " << typeid(WVLT).name() << endl;
	cout << "n = " << n << ", repeats = " << n_repeats << endl << endl;


	cout << "evens aligned ?" << is_aligned(evens.data(), v_len * sizeof(T)) << endl;
	cout << "odds aligned ?" << is_aligned(odds.data(), v_len * sizeof(T)) << endl;

	//cout << "Out of place" << endl
	//	<< "------------" << endl;
	//auto tick = high_resolution_clock::now();
	//for (size_t i = 0; i < n_repeats; ++i)
	//	WVLT::forward_lift(
	//		evens.data(), odds.data(),
	//		evens_out.data(), odds_out.data(),
	//		evens.size()
	//	);
	//auto tock = high_resolution_clock::now();

	//duration<double, std::micro> ms_double = tock - tick;
	//cout << "forward: " << ms_double.count() / n_repeats << "mu s" << endl;

	//tick = high_resolution_clock::now();
	//for (size_t i = 0; i < n_repeats; ++i)
	//	WVLT::inverse_lift(
	//		evens_out.data(), odds_out.data(),
	//		evens.data(), odds.data(),
	//		evens.size()
	//	);
	//tock = high_resolution_clock::now();

	//ms_double = tock - tick;
	//cout << "inverse: " << ms_double.count() / n_repeats << "mu s" << endl;

	cout << endl << "In place" << endl
		<< "---------" << endl;
	auto tick = high_resolution_clock::now();
	for (size_t i = 0; i < n_vec_repeats; ++i)
		WVLT::forward_lift_vec(evens.data(), odds.data(), m);
	auto tock = high_resolution_clock::now();

	duration<double, std::micro> ms_double = tock - tick;
	cout << "forward: " << ms_double.count() / n_repeats << "mu s" << endl;

	//tick = high_resolution_clock::now();
	//for (size_t i = 0; i < n_repeats; ++i)
	//	WVLT::inverse_lift(evens.data(), odds.data(), evens.size());
	//tock = high_resolution_clock::now();

	//ms_double = tock - tick;
	//cout << "inverse: " << ms_double.count() / n_repeats << "mu s" << endl;
}
*/

int main(int argc, char* argv[]) {
	if (argc > 4) {
		throw std::invalid_argument("Invalid number of inputs.");
	}
	size_t n;
	if (argc < 2) {
		n = 32;
	}
	else {
		std::string arg_n = argv[1];
		n = std::stoi(arg_n);
		n += n % 2;
	}
	if (argc < 3 && n < 50) {
		verify_transform<Haar<double>>(n);
		verify_transform<Daubechies2<double>>(n);
		verify_transform<Daubechies3<double>>(n);
		verify_transform<Daubechies4<double>>(n);
		verify_transform<Daubechies5<double>>(n);
		verify_transform<Daubechies6<double>>(n);
		verify_transform<BiorSpline3_1<double>>(n);
		verify_transform<ReverseBiorSpline3_1<double>>(n);

		verify_transform<BiorSpline4_2<double>>(n);
		verify_transform<ReverseBiorSpline4_2<double>>(n);
		verify_transform<BiorSpline2_4<double>>(n);
		verify_transform<ReverseBiorSpline2_4<double>>(n);
		verify_transform<BiorSpline6_2<double>>(n);
		verify_transform<ReverseBiorSpline6_2<double>>(n);
	}
	else{
		size_t n_repeats;
		if (argc < 3) {
			n_repeats = 1024;
		}
		else {
			std::string arg_nr = argv[2];
			n_repeats = std::stoi(arg_nr);
		}
		time_transform<Haar<double>>(n, n_repeats);
		time_transform<Daubechies2<double>>(n, n_repeats);
		time_transform<Daubechies3<double>>(n, n_repeats);
		time_transform<Daubechies4<double>>(n, n_repeats);
		time_transform<Daubechies5<double>>(n, n_repeats);
		time_transform<Daubechies6<double>>(n, n_repeats);
		time_transform<BiorSpline3_1<double>>(n, n_repeats);
		time_transform<ReverseBiorSpline3_1<double>>(n, n_repeats);
		time_transform<CDF9_7<double>>(n, n_repeats);
		time_transform<ReverseCDF9_7<double>>(n, n_repeats);
	}
}