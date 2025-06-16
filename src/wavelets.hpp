#ifndef WAVELETS_HDRONLY_H
#define WAVELETS_HDRONLY_H

#ifndef __cplusplus
#error This file is C++ and requires a C++ compiler.
#endif

#if defined(__GNUC__)
#define WAVELETS_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define WAVELETS_RESTRICT __restrict
#else
#define WAVELETS_RESTRICT
#endif

#include <algorithm>
#include <concepts>
#include <iostream>
#include <vector>

#include "ndarray.hpp"

namespace wavelets {
	using ndarray::size_v;
	using ndarray::stride_v;
	using size_t = size_v::value_type;
	using stride_t = stride_v::value_type;

	namespace detail {

		using std::cout;
		using std::endl;
		using namespace ndarray::detail;

		// Concept: has .data() and .size()
		template<typename T>
		concept HasDataAndSize = requires(T t) {
			{ t.data() } -> std::same_as<decltype(std::data(t))>;
			{ t.size() } -> std::convertible_to<std::size_t>;
		};

		// Helper to get element type from .data()
		template<typename T>
		using element_type_t = std::remove_pointer_t<decltype(std::declval<T>().data())>;

		// Combined constraint: both containers must have data/size and same element type
		template<typename C1, typename C2>
		concept CompatibleContainers = HasDataAndSize<C1> && HasDataAndSize<C2> &&
			std::same_as<element_type_t<C1>, element_type_t<C2>>;

		// only enable vector support for gcc>=5.0 and clang>=5.0
#ifndef WAVELETS_NO_VECTORS
#define WAVELETS_NO_VECTORS
#if defined(__INTEL_COMPILER)
#	undef WAVELETS_NO_VECTORS
#elif defined(_MSC_VER)
#	undef WAVELETS_NO_VECTORS
// do nothing. This is necessary because this compiler also sets __GNUC__.
#elif defined(__clang__)
// AppleClang has their own version numbering
#ifdef __apple_build_version__
#  if (__clang_major__ > 9) || (__clang_major__ == 9 && __clang_minor__ >= 1)
#     undef WAVELETS_NO_VECTORS
#  endif
#elif __clang_major__ >= 5
#  undef WAVELETS_NO_VECTORS
#endif
#elif defined(__GNUC__)
#if __GNUC__>=5
#undef WAVELETS_NO_VECTORS
#endif
#endif
#endif

		template<typename T> struct VLEN { static constexpr size_t val = 1; };

#ifndef WAVELETS_NO_VECTORS
#if (defined(__AVX512F__))
		template<> struct VLEN<float> { static constexpr size_t val = 16; };
		template<> struct VLEN<double> { static constexpr size_t val = 8; };
#elif (defined(__AVX__))
		template<> struct VLEN<float> { static constexpr size_t val = 8; };
		template<> struct VLEN<double> { static constexpr size_t val = 4; };
#elif (defined(__SSE2__))
		template<> struct VLEN<float> { static constexpr size_t val = 4; };
		template<> struct VLEN<double> { static constexpr size_t val = 2; };
#elif (defined(__VSX__))
		template<> struct VLEN<float> { static constexpr size_t val = 4; };
		template<> struct VLEN<double> { static constexpr size_t val = 2; };
#elif (defined(__ARM_NEON__) || defined(__ARM_NEON))
		template<> struct VLEN<float> { static constexpr size_t val = 4; };
		template<> struct VLEN<double> { static constexpr size_t val = 2; };
#else
#define WAVELETS_NO_VECTORS
#endif
#endif

		// Boundary conditions
		struct ZeroBoundary {
			static inline size_t index_of(const ptrdiff_t i, const size_t n) {
				if ((i < 0) || (i >= ptrdiff_t(n))) {
					return n;
				}
				else {
					return i;
				}
			}
		};

		struct ConstantBoundary {
			static inline size_t index_of(const ptrdiff_t i, const size_t n) {
				if (i < 0) {
					return 0;
				}
				else if (i >= ptrdiff_t(n)) {
					return n - 1;
				}
				else {
					return i;
				}
			}
		};

		struct PeriodicBoundary {
			static inline size_t index_of(const ptrdiff_t i, const size_t n) {
				return i % n;
			}
		};

		struct SymmetricBoundary {
			static inline size_t index_of(const ptrdiff_t i, const size_t n) {
				ptrdiff_t io = i;
				while ((io >= ptrdiff_t(n)) || (io < 0)) {
					if (io < 0) {
						io = -(io + 1);
					}
					else {
						io = 2 * (n - 1) - (io - 1);
					}
				}
				return io;
			}
		};

		struct ReflectBoundary {
			static inline size_t index_of(const ptrdiff_t i, const size_t n) {
				ptrdiff_t io = i;
				while ((io >= ptrdiff_t(n)) || (io < 0)) {
					if (io < 0) {
						io = -io;
					}
					else {
						io = 2 * (n - 1) - io;
					}
				}
				return io;
			}
		};

		// compile time unrolling of step loops
		template<ptrdiff_t offset, auto val, auto... vals>
		struct Lift {

			constexpr static size_t n_vals = sizeof...(vals) + 1;

			template<typename V>
			static inline V apply(const V* x, const size_t i, const stride_t stride_i, const size_t j) {
				return val * x[(i + offset) * stride_i + j] + Lift<offset + 1, vals...>::apply(x, i, stride_i, j);
			}

			template<typename BC, typename V>
			static inline V bc_apply(const V* x, const ptrdiff_t i, const size_t n, const stride_t stride_i, const size_t j) {
				size_t i_bc = BC::index_of(i + offset, n);
				if (i_bc == n) {
					return Lift<offset + 1, vals...>::template bc_apply<BC>(x, i, n, stride_i, j);
				}
				else {
					return val * x[i_bc * stride_i + j] + Lift<offset + 1, vals...>::template bc_apply<BC>(x, i, n, stride_i, j);
				}
			}

			template<typename V>
			static inline V bc_adj_apply(const V* x, const ptrdiff_t i, const size_t n, const stride_t stride_i, const size_t j) {
				if (i == ptrdiff_t(n)) {
					return V(0.0);
				}
				else if (i < 0) {
					return Lift<offset + 1, vals...>::bc_adj_apply(x, i + 1, n, stride_i, j);
				}
				else {
					return val * x[i * stride_i + j] + Lift<offset + 1, vals...>::bc_adj_apply(x, i + 1, n, stride_i, j);
				}
			}

		};

		template<ptrdiff_t offset, auto val>
		struct Lift<offset, val> {

			const static size_t n_vals = 1;

			template<typename V>
			static inline V apply(const V* x, const size_t i, const stride_t stride_i, const size_t j) {
				return val * x[(i + offset) * stride_i + j];
			}

			template<typename BC, typename V>
			static inline V bc_apply(const V* x, const ptrdiff_t i, const size_t n, const stride_t stride_i, const size_t j) {
				size_t i_bc = BC::index_of(i + offset, n);
				if (i_bc == n) {
					return V(0.0);
				}
				else {
					return val * x[i_bc * stride_i + j];
				}
			}

			template<typename V>
			static inline V bc_adj_apply(const V* x, const ptrdiff_t i, const size_t n, const stride_t stride_i, const size_t j) {
				if ((i == ptrdiff_t(n)) || (i < 0)){
					return V(0.0);
				}
				else {
					return val * x[i * stride_i + j];
				}
			}
		};

		// for adjoints with reversing the values
		template<auto...>
		struct ReverseValues;

		template<>
		struct ReverseValues<> {
			template<template<auto...> class Target, auto... Result>
			struct Apply {
				using type = Target<Result...>;
			};
		};

		template<auto First, auto... Rest>
		struct ReverseValues<First, Rest...> {
			template<template<auto...> class Target, auto... Result>
			struct Apply {
				using type = typename ReverseValues<Rest...>::template Apply<Target, First, Result...>::type;
			};
		};

		// Template to bind offset before applying reversed vals
		template<ptrdiff_t offset>
		struct LiftWrapper {
			template<auto... ReversedVals>
			using type = Lift<offset, ReversedVals...>;
		};

		template<ptrdiff_t offset, auto... vals>
		using ReversedLift = typename ReverseValues<vals...>::template Apply<
			LiftWrapper<offset>::template type
		>::type;

		// lifting steps
		template<ptrdiff_t offset, auto... vals>
		struct update_d {
			constexpr static size_t n_vals = sizeof...(vals);

			constexpr static ptrdiff_t max_offset = ptrdiff_t(n_vals) - 1 + offset;
			constexpr static size_t n_front = (offset < 0) ? -offset : 0;
			constexpr static size_t n_back = (max_offset < 0) ? 0 : max_offset;

			constexpr static ptrdiff_t offset_r = -max_offset;

			constexpr static ptrdiff_t max_offset_r = ptrdiff_t(n_vals) - 1 + offset_r;
			constexpr static size_t n_front_r = (offset_r < 0) ? -offset_r : 0;
			constexpr static size_t n_back_r = (max_offset_r < 0) ? 0 : max_offset_r;

			using lifter = Lift<offset, vals...>;
			using lifter_r = ReversedLift<offset_r, vals...>;

			template<typename BC, typename V>
			static inline void forward(const V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {

				for (ptrdiff_t i = 0; i < ptrdiff_t(n_front); ++i)
					for (size_t j = 0; j < m; ++j)
						d[i * stride_d + j] += lifter::template bc_apply<BC>(s, i, ns, stride_s, j);
				for (size_t i = n_front; i < nd - n_back; ++i)
					for (size_t j = 0; j < m; ++j)
						d[i * stride_d + j] += lifter::apply(s, i, stride_s, j);
				for (ptrdiff_t i = nd - n_back; i < ptrdiff_t(nd); ++i)
					for (size_t j = 0; j < m; ++j)
						d[i * stride_d + j] += lifter::template bc_apply<BC>(s, i, ns, stride_s, j);
			}

			template<typename BC, typename V>
			static inline void inverse(const V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {

				for (ptrdiff_t i = 0; i < ptrdiff_t(n_front); ++i)
					for (size_t j = 0; j < m; ++j)
						d[i * stride_d + j] -= lifter::template bc_apply<BC>(s, i, ns, stride_s, j);
				for (size_t i = n_front; i < nd - n_back; ++i)
					for (size_t j = 0; j < m; ++j)
						d[i * stride_d + j] -= lifter::apply(s, i, stride_s, j);
				for (ptrdiff_t i = nd - n_back; i < ptrdiff_t(nd); ++i)
					for (size_t j = 0; j < m; ++j)
						d[i * stride_d + j] -= lifter::template bc_apply<BC>(s, i, ns, stride_s, j);
			}

			template<typename BC, typename V>
			static inline void forward_adjoint(V* WAVELETS_RESTRICT s, const V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {

				for (ptrdiff_t i = offset, j = 1 - n_vals; i < 0; ++i, ++j) {
					size_t io = BC::index_of(i, ns);
					if ((io != ns) && (ptrdiff_t(io) != i)) {
						for (size_t k = 0; k < m; ++k)
							s[io * stride_s + k] += lifter_r::bc_adj_apply(d, j, nd, stride_d, k);
					}
				}

				for (ptrdiff_t i = 0; i < ptrdiff_t(n_front_r); ++i)
					for (size_t k = 0; k < m; ++k)
						s[i * stride_s + k] += lifter_r::template bc_apply<ZeroBoundary>(d, i, nd, stride_d, k);
				for (size_t i = n_front_r; i < nd - n_back_r; ++i)
					for (size_t k = 0; k < m; ++k)
						s[i * stride_s + k] += lifter_r::apply(d, i, stride_d, k);
				for (ptrdiff_t i = nd - n_back_r; i < ptrdiff_t(ns); ++i)
					for (size_t k = 0; k < m; ++k)
						s[i * stride_s + k] += lifter_r::template bc_apply<ZeroBoundary>(d, i, nd, stride_d, k);

				for (ptrdiff_t i = nd, j = nd + offset_r; i < ptrdiff_t(nd) - offset_r; ++i, ++j) {
					size_t io = BC::index_of(i, ns);
					if ((io != ns) && (ptrdiff_t(io) != i)) {
						for (size_t k = 0; k < m; ++k)
							s[io * stride_s + k] += lifter_r::bc_adj_apply(d, j, nd, stride_d, k);
					}
				}
			}

			template<typename BC, typename V>
			static inline void inverse_adjoint(V* WAVELETS_RESTRICT s, const V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {

				for (ptrdiff_t i = offset, j = 1 - n_vals; i < 0; ++i, ++j) {
					size_t io = BC::index_of(i, ns);
					if ((io != ns) && (ptrdiff_t(io) != i)) {
						for (size_t k = 0; k < m; ++k)
							s[io * stride_s + k] -= lifter_r::bc_adj_apply(d, j, nd, stride_d, k);
					}
				}

				for (ptrdiff_t i = 0; i < ptrdiff_t(n_front_r); ++i)
					for (size_t k = 0; k < m; ++k)
						s[i * stride_s + k] -= lifter_r::template bc_apply<ZeroBoundary>(d, i, nd, stride_d, k);
				for (size_t i = n_front_r; i < nd - n_back_r; ++i)
					for (size_t k = 0; k < m; ++k)
						s[i * stride_s + k] -= lifter_r::apply(d, i, stride_d, k);
				for (ptrdiff_t i = nd - n_back_r; i < ptrdiff_t(ns); ++i)
					for (size_t k = 0; k < m; ++k)
						s[i * stride_s + k] -= lifter_r::template bc_apply<ZeroBoundary>(d, i, nd, stride_d, k);

				for (ptrdiff_t i = nd, j = nd + offset_r; i < ptrdiff_t(nd) - offset_r; ++i, ++j) {
					size_t io = BC::index_of(i, ns);
					if ((io != ns) && (ptrdiff_t(io) != i)) {
						for (size_t k = 0; k < m; ++k)
							s[io * stride_s + k] -= lifter_r::bc_adj_apply(d, j, nd, stride_d, k);
					}
				}
			}
		};

		template<ptrdiff_t offset, auto... vals>
		struct update_s {
			constexpr static size_t n_vals = sizeof...(vals);

			constexpr static ptrdiff_t max_offset = ptrdiff_t(n_vals) - 1 + offset;
			constexpr static size_t n_front = (offset < 0) ? -offset : 0;
			constexpr static size_t n_back = (max_offset < 0) ? 0 : max_offset;

			constexpr static ptrdiff_t offset_r = -max_offset;

			constexpr static ptrdiff_t max_offset_r = ptrdiff_t(n_vals) - 1 + offset_r;
			constexpr static size_t n_front_r = (offset_r < 0) ? -offset_r : 0;
			constexpr static size_t n_back_r = (max_offset_r < 0) ? 0 : max_offset_r;

			using lifter = Lift<offset, vals...>;
			using lifter_r = ReversedLift<offset_r, vals...>;

			template<typename BC, typename V>
			static inline void forward(V* WAVELETS_RESTRICT s, const V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {

				for (ptrdiff_t i = 0; i < ptrdiff_t(n_front); ++i)
					for (size_t j = 0; j < m; ++j)
						s[i * stride_s + j] += lifter::template bc_apply<BC>(d, i, nd, stride_d, j);
				for (size_t i = n_front; i < nd - n_back; ++i)
					for (size_t j = 0; j < m; ++j)
						s[i * stride_s + j] += lifter::apply(d, i, stride_d, j);
				for (ptrdiff_t i = nd - n_back; i < ptrdiff_t(ns); ++i)
					for (size_t j = 0; j < m; ++j)
						s[i * stride_s + j] += lifter::template bc_apply<BC>(d, i, nd, stride_d, j);
			}

			template<typename BC, typename V>
			static inline void inverse(V* WAVELETS_RESTRICT s, const V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {

				for (ptrdiff_t i = 0; i < ptrdiff_t(n_front); ++i)
					for (size_t j = 0; j < m; ++j)
						s[i * stride_s + j] -= lifter::template bc_apply<BC>(d, i, nd, stride_d, j);
				for (size_t i = n_front; i < nd - n_back; ++i)
					for (size_t j = 0; j < m; ++j)
						s[i * stride_s + j] -= lifter::apply(d, i, stride_d, j);
				for (ptrdiff_t i = nd - n_back; i < ptrdiff_t(ns); ++i)
					for (size_t j = 0; j < m; ++j)
						s[i * stride_s + j] -= lifter::template bc_apply<BC>(d, i, nd, stride_d, j);
			}

			template<typename BC, typename V>
			static inline void forward_adjoint(const V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {

				for (ptrdiff_t i = offset, j = 1 - n_vals; i < 0; ++i, ++j) {
					size_t io = BC::index_of(i, nd);
					if ((io != nd) && (ptrdiff_t(io) != i)) {
						for (size_t k = 0; k < m; ++k)
							d[io * stride_d + k] += lifter_r::bc_adj_apply(s, j, ns, stride_s, k);
					}
				}

				for (ptrdiff_t i = 0; i < ptrdiff_t(n_front_r); ++i)
					for (size_t k = 0; k < m; ++k)
						d[i * stride_d + k] += lifter_r::template bc_apply<ZeroBoundary>(s, i, ns, stride_s, k);
				for (size_t i = n_front_r; i < nd - n_back_r; ++i)
					for (size_t k = 0; k < m; ++k)
						d[i * stride_d + k] += lifter_r::apply(s, i, stride_s, k);
				for (ptrdiff_t i = nd - n_back_r; i < ptrdiff_t(nd); ++i)
					for (size_t k = 0; k < m; ++k)
						d[i * stride_d + k] += lifter_r::template bc_apply<ZeroBoundary>(s, i, ns, stride_s, k);


				for (ptrdiff_t i = nd, j = nd + offset_r; i < ptrdiff_t(ns) - offset_r; ++i, ++j) {
					size_t io = BC::index_of(i, nd);
					if ((io != nd) && (ptrdiff_t(io) != i)) {
						for (size_t k = 0; k < m; ++k)
							d[io * stride_d + k] += lifter_r::bc_adj_apply(s, j, ns, stride_s, k);
					}
				}
			}

			template<typename BC, typename V>
			static inline void inverse_adjoint(const V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {

				for (ptrdiff_t i = offset, j = 1 - n_vals; i < 0; ++i, ++j) {
					size_t io = BC::index_of(i, nd);
					if ((io != nd) && (ptrdiff_t(io) != i)) {
						for (size_t k = 0; k < m; ++k)
							d[io * stride_d + k] -= lifter_r::bc_adj_apply(s, j, ns, stride_s, k);
					}
				}

				for (ptrdiff_t i = 0; i < ptrdiff_t(n_front_r); ++i)
					for (size_t k = 0; k < m; ++k)
						d[i * stride_d + k] -= lifter_r::template bc_apply<ZeroBoundary>(s, i, ns, stride_s, k);
				for (size_t i = n_front_r; i < nd - n_back_r; ++i)
					for (size_t k = 0; k < m; ++k)
						d[i * stride_d + k] -= lifter_r::apply(s, i, stride_s, k);
				for (ptrdiff_t i = nd - n_back_r; i < ptrdiff_t(nd); ++i)
					for (size_t k = 0; k < m; ++k)
						d[i * stride_d + k] -= lifter_r::template bc_apply<ZeroBoundary>(s, i, ns, stride_s, k);


				for (ptrdiff_t i = nd, j = nd + offset_r; i < ptrdiff_t(ns) - offset_r; ++i, ++j) {
					size_t io = BC::index_of(i, nd);
					if ((io != nd) && (ptrdiff_t(io) != i)) {
						for (size_t k = 0; k < m; ++k)
							d[io * stride_d + k] -= lifter_r::bc_adj_apply(s, j, ns, stride_s, k);
					}
				}
			}
		};

		template<auto val1, auto val2 = val1>
		struct scale {

			template<typename BC, typename V>
			static inline void forward(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m)
			{
				for (size_t i = 0; i < nd; ++i)
					for (size_t j = 0; j < m; ++j)
						d[i * stride_d + j] /= val1;
				for (size_t i = 0; i < ns; ++i)
					for (size_t j = 0; j < m; ++j)
						s[i * stride_s + j] *= val2;
			}


			template<typename BC, typename V>
			static inline void inverse(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m)
			{
				for (size_t i = 0; i < nd; ++i)
					for (size_t j = 0; j < m; ++j)
						d[i * stride_d + j] *= val1;
				for (size_t i = 0; i < ns; ++i)
					for (size_t j = 0; j < m; ++j)
						s[i * stride_s + j] /= val2;
			}


			template<typename BC, typename V>
			static inline void forward_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m)
			{
				forward<BC>(s, d, ns, nd, stride_s, stride_d, m);
			}


			template<typename BC, typename V>
			static inline void inverse_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m)
			{
				inverse<BC>(s, d, ns, nd, stride_s, stride_d, m);
			}
		};

		template<typename WVLT, typename BC, size_t step, size_t n_step>
		struct step_builder {

			template<typename V>
			static inline void forward(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {

				std::tuple_element_t<step, typename WVLT::steps>::template forward<BC>(s, d, ns, nd, stride_s, stride_d, m);
				step_builder<WVLT, BC, step + 1, n_step>::forward(s, d, ns, nd, stride_s, stride_d, m);
			}

			template<typename V>
			static inline void inverse(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {

				step_builder<WVLT, BC, step + 1, n_step>::inverse(s, d, ns, nd, stride_s, stride_d, m);
				std::tuple_element_t<step, typename WVLT::steps>::template inverse<BC>(s, d, ns, nd, stride_s, stride_d, m);
			}

			template<typename V>
			static inline void forward_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {

				step_builder<WVLT, BC, step + 1, n_step>::forward_adjoint(s, d, ns, nd, stride_s, stride_d, m);
				std::tuple_element_t<step, typename WVLT::steps>::template forward_adjoint<BC>(s, d, ns, nd, stride_s, stride_d, m);
			}

			template<typename V>
			static inline void inverse_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {

				std::tuple_element_t<step, typename WVLT::steps>::template inverse_adjoint<BC>(s, d, ns, nd, stride_s, stride_d, m);
				step_builder<WVLT, BC, step + 1, n_step>::inverse_adjoint(s, d, ns, nd, stride_s, stride_d, m);
			}
		};

		template<typename WVLT, typename BC, size_t step>
		struct step_builder<WVLT, BC, step, step> {

			template<typename V>
			static inline void forward(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {}

			template<typename V>
			static inline void inverse(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {}

			template<typename V>
			static inline void forward_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {}

			template<typename V>
			static inline void inverse_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {}
		};

		template<typename WVLT, typename BC=ZeroBoundary>
		class LiftingTransform {

		public:
			using type = typename WVLT::type;
			using boundary_condition = BC;

			template<typename V>
			static void forward(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m
			) {
				step_builder<WVLT, BC, 0, WVLT::n_steps>::forward(s, d, ns, nd, stride_s, stride_d, m);
			}

			template<typename V>
			static void forward(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {
				forward(s, d, ns, nd, 1, 1, 1);
			}

			template<typename C1, typename C2>
			static void forward(C1& WAVELETS_RESTRICT s, C2& WAVELETS_RESTRICT d) {
				forward(s.data(), d.data(), s.size(), d.size());
			}

			template<typename V>
			static void forward(V* WAVELETS_RESTRICT x, const size_t n) {
				size_t nd = n / 2;
				size_t ns = n - nd;
				V* WAVELETS_RESTRICT s = x;
				V* WAVELETS_RESTRICT d = x + 1;
				forward(s, d, ns, nd, 2, 2, 1);
			}

			template<typename C1>
			static void forward(C1& WAVELETS_RESTRICT x) {
				forward(x.data(), x.size());
			}

			template<typename V>
			static void inverse(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {
				step_builder<WVLT, BC, 0, WVLT::n_steps>::inverse(s, d, ns, nd, stride_s, stride_d, m);
			}

			template<typename V>
			static void inverse(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {
				inverse(s, d, ns, nd, 1, 1, 1);
			}

			template<typename C1, typename C2>
			static void inverse(C1& WAVELETS_RESTRICT s, C2& WAVELETS_RESTRICT d) {
				inverse(s.data(), d.data(), s.size(), d.size());
			}

			template<typename V>
			static void inverse(V* WAVELETS_RESTRICT x, const size_t n) {
				size_t nd = n / 2;
				size_t ns = n - nd;
				V* WAVELETS_RESTRICT s = x;
				V* WAVELETS_RESTRICT d = x + 1;
				inverse(s, d, ns, nd, 2, 2, 1);
			}

			template<typename C1>
			static void inverse(C1& WAVELETS_RESTRICT x) {
				inverse(x.data(), x.size());
			}

			template<typename V>
			static void forward_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {
				step_builder<WVLT, BC, 0, WVLT::n_steps>::forward_adjoint(s, d, ns, nd, stride_s, stride_d, m);
			}

			template<typename V>
			static void forward_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {
				forward_adjoint(s, d, ns, nd, 1, 1, 1);
			}

			template<typename C1, typename C2>
			static void forward_adjoint(C1& WAVELETS_RESTRICT s, C2& WAVELETS_RESTRICT d) {
				forward_adjoint(s.data(), d.data(), s.size(), d.size());
			}

			template<typename V>
			static void forward_adjoint(V* WAVELETS_RESTRICT x, const size_t n) {
				size_t nd = n / 2;
				size_t ns = n - nd;
				V* WAVELETS_RESTRICT s = x;
				V* WAVELETS_RESTRICT d = x + 1;
				forward_adjoint(s, d, ns, nd, 2, 2, 1);
			}

			template<typename C1>
			static void forward_adjoint(C1& WAVELETS_RESTRICT x) {
				forward_adjoint(x.data(), x.size());
			}

			template<typename V>
			static void inverse_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d,
				const size_t ns, const size_t nd,
				const stride_t stride_s, const stride_t stride_d,
				const size_t m) {
				step_builder<WVLT, BC, 0, WVLT::n_steps>::inverse_adjoint(s, d, ns, nd, stride_s, stride_d, m);
			}

			template<typename V>
			static void inverse_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {
				inverse_adjoint(s, d, ns, nd, 1, 1, 1);
			}

			template<typename C1, typename C2>
			static void inverse_adjoint(C1& WAVELETS_RESTRICT s, C2& WAVELETS_RESTRICT d) {
				inverse_adjoint(s.data(), d.data(), s.size(), d.size());
			}

			template<typename V>
			static void inverse_adjoint(V* WAVELETS_RESTRICT x, const size_t n) {
				size_t nd = n / 2;
				size_t ns = n - nd;
				V* WAVELETS_RESTRICT s = x;
				V* WAVELETS_RESTRICT d = x + 1;
				inverse_adjoint(s, d, ns, nd, 2, 2, 1);
			}

			template<typename C1>
			static void inverse_adjoint(C1& WAVELETS_RESTRICT x) {
				inverse_adjoint(x.data(), x.size());
			}
		};

		template<typename T> aligned_array<T> alloc_tmp(const size_v& shape,
			const size_t axsize, const size_t n_levels)
		{
			auto othersize = prod(shape) / axsize;
			auto sz = axsize;
			// for each lvl of the transform, need to add an extra element if the resulting ns shape is odd
			// so there is working space for the diver to deinterleave s onto d at each level.
			int ns = axsize;
			for (size_t lvl = 0; lvl < n_levels; ++lvl) {
				if (ns % 2 == 1) sz += 1;
				ns = ns - ns / 2;
			}
			auto tmpsize = sz * ((othersize >= VLEN<T>::val) ? VLEN<T>::val : 1);
			return aligned_array<T>(tmpsize);
		}

		template<typename T> aligned_array<T> alloc_tmp(const size_v& shape,
			const size_v& axes, const size_t n_levels)
		{
			size_t fullsize = prod(shape);
			size_t tmpsize = 0;
			for (size_t i = 0; i < axes.size(); ++i)
			{
				auto axsize = shape[axes[i]];
				auto othersize = fullsize / axsize;
				int ns = axsize;
				for (size_t lvl = 0; lvl < n_levels; ++lvl) {
					if (ns % 2 == 1) axsize += 1;
					ns = ns - ns / 2;
				}
				auto sz = axsize * ((othersize >= VLEN<T>::val) ? VLEN<T>::val : 1);
				if (sz > tmpsize) tmpsize = sz;
			}
			return aligned_array<T>(tmpsize);
		}

		// copy in operations (copy, copy & interleave, copy & deinterleave)

		template <typename T, size_t vlen> inline void vec_copy_input(const multi_iter<vlen>& it,
			const cndarr<T>& src, T* WAVELETS_RESTRICT dst, const size_t from, const size_t to)
		{	
			for (size_t i = from, ii = 0; i < to; ++i, ++ii)
				for (size_t j = 0; j < vlen; ++j)
					dst[ii * vlen + j] = src[it.iofs(j, i)];
		}

		template <typename T, size_t vlen> inline void vec_copy_input(const multi_iter<vlen>& it,
			const cndarr<T>& src, T* WAVELETS_RESTRICT dst)
		{
			vec_copy_input(it, src, dst, 0, it.length_in());
		}

		template <typename T, size_t vlen> inline void copy_input(const multi_iter<vlen>& it,
			const cndarr<T>& src, T* WAVELETS_RESTRICT dst, const size_t from, const size_t to)
		{
			if (dst == &src[it.iofs(from)]) return;  // in-place
			for (size_t i = from, ii = 0; i < to; ++i, ++ii)
				dst[ii] = src[it.iofs(i)];
		}

		template <typename T, size_t vlen> inline void copy_input(const multi_iter<vlen>& it,
			const cndarr<T>& src, T* WAVELETS_RESTRICT dst)
		{
			copy_input(it, src, dst, 0, it.length_in());
		}

		template <typename T, size_t vlen> inline void copy_inout(const multi_iter<vlen>& it,
			const cndarr<T>& src, ndarr<T>& dst) {
			if (&dst[it.oofs(0)] == &src[it.iofs(0)]) return; //in-place
			for (size_t i = 0; i < it.length_in(); ++i)
				dst[it.oofs(i)] = src[it.iofs(i)];
		}

		template <typename T, size_t vlen> inline void vec_interleave_input(const multi_iter<vlen>& it,
			const cndarr<T>& src, T* WAVELETS_RESTRICT dst, const size_t len)
		{
			size_t nd = len / 2;
			size_t ns = len - nd;

			size_t i = 0;
			size_t ii = 0;
			for (; i < nd; ++i, ii += 2)
				for (size_t j = 0; j < vlen; ++j) {
					dst[ii * vlen + j] = src[it.iofs(j, i)];
					dst[(ii + 1) * vlen + j] = src[it.iofs(j, i + ns)];
				}
			for (; i < ns; ++i, ii += 2)
				for(size_t j = 0; j < vlen; ++j)
					dst[ii * vlen + j] = src[it.iofs(j, i)];
		}

		template <typename T, size_t vlen> inline void vec_interleave_input(const multi_iter<vlen>& it,
			const cndarr<T>& src, T* WAVELETS_RESTRICT dst)
		{
			vec_interleave_input(it, src, dst, it.length_in());
		}

		template <typename T, size_t vlen> inline void interleave_input(const multi_iter<vlen>& it,
			const cndarr<T>& src, T* WAVELETS_RESTRICT dst, const size_t len)
		{
			size_t nd = len / 2;
			size_t ns = len - nd;

			size_t i = 0;
			size_t ii = 0;
			for (; i < nd; ++i, ii += 2) {
				dst[ii] = src[it.iofs(i)];
				dst[ii + 1] = src[it.iofs(i + ns)];
			}
			for (; i < ns; ++i, ii += 2)
				dst[ii] = src[it.iofs(i)];
		}

		template <typename T, size_t vlen> inline void interleave_input(const multi_iter<vlen>& it,
			const cndarr<T>& src, T* WAVELETS_RESTRICT dst)
		{
			interleave_input(it, src, dst, it.length_in());
		}

		template <typename T, size_t vlen> inline void vec_deinterleave_input(const multi_iter<vlen>& it,
			const cndarr<T>& src, T* WAVELETS_RESTRICT dst, const size_t len)
		{
			size_t nd = len / 2;
			size_t ns = len - nd;

			size_t i = 0;
			size_t ii = 0;
			for (; i < nd; ++i, ii += 2)
				for (size_t j = 0; j < vlen; ++j) {
					dst[i * vlen + j] = src[it.iofs(j, ii)];
					dst[(i + ns) * vlen + j] = src[it.iofs(j, ii + 1)];
				}
			for(; i < ns; ++i, ii += 2)
				for (size_t j = 0; j < vlen; ++j)
					dst[i * vlen + j] = src[it.iofs(j, ii)];
		}

		template <typename T, size_t vlen> inline void vec_deinterleave_input(const multi_iter<vlen>& it,
			const cndarr<T>& src, T* WAVELETS_RESTRICT dst)
		{
			vec_deinterleave_input(it, src, dst, it.length_in());
		}

		template <typename T, size_t vlen> inline void deinterleave_input(const multi_iter<vlen>& it,
			const cndarr<T>& src, T* WAVELETS_RESTRICT dst, const size_t len)
		{
			size_t nd = len / 2;
			size_t ns = len - nd;

			size_t i = 0;
			size_t ii = 0;
			for (size_t i = 0, ii=0; i < nd; ++i, ii+=2) {
				dst[i] = src[it.iofs(ii)];
				dst[i + ns] = src[it.iofs(ii + 1)];
			}
			for (; i < ns; ++i, ii += 2)
				dst[i] = src[it.iofs(ii)];
		}

		template <typename T, size_t vlen> inline void deinterleave_input(const multi_iter<vlen>& it,
			const cndarr<T>& src, T* WAVELETS_RESTRICT dst)
		{
			deinterleave_input(it, src, dst, it.length_in());
		}

		// copy out operations (copy, copy & interleave, copy & deinterleave)

		template<typename T, size_t vlen> inline void vec_copy_output(const multi_iter<vlen>& it,
			const T* WAVELETS_RESTRICT src, ndarr<T>& dst, const size_t from, const size_t to)
		{
			for (size_t i = from, ii = 0; i < to; ++i, ++ii)
				for (size_t j = 0; j < vlen; ++j)
					dst[it.oofs(j, i)] = src[ii * vlen + j];
		}

		template<typename T, size_t vlen> inline void vec_copy_output(const multi_iter<vlen>& it,
			const T* WAVELETS_RESTRICT src, ndarr<T>& dst)
		{
			vec_copy_output(it, src, dst, 0, it.length_out());
		}

		template<typename T, size_t vlen> inline void copy_output(const multi_iter<vlen>& it,
			const T* WAVELETS_RESTRICT src, ndarr<T>& dst, const size_t from, const size_t to)
		{
			if (src == &dst[it.oofs(from)]) return;  // in-place
			for (size_t i = from, ii = 0; i < to; ++i, ++ii)
				dst[it.oofs(i)] = src[ii];
		}

		template<typename T, size_t vlen> inline void copy_output(const multi_iter<vlen>& it,
			const T* WAVELETS_RESTRICT src, ndarr<T>& dst)
		{
			copy_output(it, src, dst, it.length_out());
		}

		template<typename T, size_t vlen> inline void vec_interleave_output(const multi_iter<vlen>& it,
			const T* WAVELETS_RESTRICT src, ndarr<T>& dst, const size_t len)
		{
			size_t nd = len / 2;
			size_t ns = len - nd;

			size_t i = 0;
			size_t ii = 0;
			for (; i < nd; ++i, ii += 2)
				for (size_t j = 0; j < vlen; ++j) {
					dst[it.oofs(j, ii)] = src[i * vlen + j];
					dst[it.oofs(j, ii + 1)] = src[(i + ns) * vlen + j];
				}
			for(; i < ns; ++i, ii += 2)
				for (size_t j = 0; j < vlen; ++j)
					dst[it.oofs(j, ii)] = src[i * vlen + j];
		}

		template<typename T, size_t vlen> inline void vec_interleave_output(const multi_iter<vlen>& it,
			const T* WAVELETS_RESTRICT src, ndarr<T>& dst)
		{
			vec_interleave_output(it, src, dst, it.length_out());
		}

		template<typename T, size_t vlen> inline void interleave_output(const multi_iter<vlen>& it,
			const T* WAVELETS_RESTRICT src, ndarr<T>& dst, const size_t len)
		{
			size_t nd = len / 2;
			size_t ns = len - nd;

			size_t i = 0;
			size_t ii = 0;
			for (; i < nd; ++i, ii += 2) {
				dst[it.oofs(ii)] = src[i];
				dst[it.oofs(ii + 1)] = src[i + ns];
			}
			for (; i < ns; ++i, ii += 2)
				dst[it.oofs(ii)] = src[i];
		}

		template<typename T, size_t vlen> inline void interleave_output(const multi_iter<vlen>& it,
			const T* WAVELETS_RESTRICT src, ndarr<T>& dst)
		{
			interleave_output(it, src, dst, it.length_out());
		}

		template<typename T, size_t vlen> inline void vec_deinterleave_output(const multi_iter<vlen>& it,
			const T* WAVELETS_RESTRICT src, ndarr<T>& dst, const size_t len)
		{
			size_t nd = len / 2;
			size_t ns = len - nd;

			size_t i = 0;
			size_t ii = 0;
			for (; i < nd; ++i, ii += 2)
				for (size_t j = 0; j < vlen; ++j) {
					dst[it.oofs(j, i)] = src[ii * vlen + j];
					dst[it.oofs(j, i + ns)] = src[(ii + 1) * vlen + j];
				}
			for (; i < ns; ++i, ii += 2)
				for (size_t j = 0; j < vlen; ++j)
					dst[it.oofs(j, i)] = src[ii * vlen + j];

		}

		template<typename T, size_t vlen> inline void vec_deinterleave_output(const multi_iter<vlen>& it,
			const T* WAVELETS_RESTRICT src, ndarr<T>& dst)
		{
			vec_deinterleave_output(it, src, dst, it.length_out());

		}

		template<typename T, size_t vlen> inline void deinterleave_output(const multi_iter<vlen>& it,
			const T* WAVELETS_RESTRICT src, ndarr<T>& dst, const size_t len)
		{
			size_t nd = len / 2;
			size_t ns = len - nd;

			size_t i = 0;
			size_t ii = 0;
			for (; i < nd; ++i, ii += 2){
				dst[it.oofs(i)] = src[ii];
				dst[it.oofs(i + ns)] = src[ii + 1];
			}
			for (; i < ns; ++i, ii += 2)
				dst[it.oofs(i)] = src[ii];
		}

		template<typename T, size_t vlen> inline void deinterleave_output(const multi_iter<vlen>& it,
			const T* WAVELETS_RESTRICT src, ndarr<T>& dst)
		{
			deinterleave_output(it, src, dst, it.length_out());
		}

		// buffer interleaves and de-interleaves
		template <typename T, size_t vlen> inline void vec_deinterleave(
			const T* WAVELETS_RESTRICT src, T* WAVELETS_RESTRICT dst, const size_t len)
		{
			size_t nd = len / 2;
			size_t ns = len - nd;

			size_t i = 0;
			size_t ii = 0;
			for (; i < nd; ++i, ii += 2)
				for (size_t j = 0; j < vlen; ++j) {
					dst[i * vlen + j] = src[ii * vlen + j];
					dst[(i + ns) * vlen + j] = src[(ii + 1) * vlen + j];
				}
			for (; i < ns; ++i, ii += 2)
				for (size_t j = 0; j < vlen; ++j)
					dst[i * vlen + j] = src[ii * vlen + j];
		}

		template <typename T> inline void deinterleave(
			const T* WAVELETS_RESTRICT src, T* WAVELETS_RESTRICT dst, const size_t len)
		{
			size_t nd = len / 2;
			size_t ns = len - nd;

			size_t i = 0;
			size_t ii = 0;
			for (size_t i = 0, ii = 0; i < nd; ++i, ii += 2) {
				dst[i] = src[ii];
				dst[i + ns] = src[ii + 1];
			}
			for (; i < ns; ++i, ii += 2)
				dst[i] = src[ii];
		}

		template<typename T, size_t vlen> inline void vec_interleave(
			const T* WAVELETS_RESTRICT src, T* WAVELETS_RESTRICT dst, const size_t len)
		{
			size_t nd = len / 2;
			size_t ns = len - nd;

			size_t i = 0;
			size_t ii = 0;
			for (; i < nd; ++i, ii += 2)
				for (size_t j = 0; j < vlen; ++j) {
					dst[ii * vlen + j] = src[i * vlen + j];
					dst[(ii + 1) * vlen + j] = src[(i + ns) * vlen + j];
				}
			for (; i < ns; ++i, ii += 2)
				for (size_t j = 0; j < vlen; ++j)
					dst[ii * vlen + j] = src[i * vlen + j];

		}

		template <typename T> inline void interleave(
			const T* WAVELETS_RESTRICT src, T* WAVELETS_RESTRICT dst, const size_t len)
		{
			size_t nd = len / 2;
			size_t ns = len - nd;

			size_t i = 0;
			size_t ii = 0;
			for (; i < nd; ++i, ii += 2) {
				dst[ii] = src[i];
				dst[ii + 1] = src[i + ns];
			}
			for (; i < ns; ++i, ii += 2)
				dst[ii] = src[i];
		}

		template<typename WVLT, typename BC>
		struct forward_driver {
			using wavelet = WVLT;
			using trans = LiftingTransform<WVLT, BC>;
			const static int dir = 1;

			template<typename T, size_t vlen>
			static inline void apply(const multi_iter<vlen>& it, const cndarr<T>& ain, ndarr<T>& aout, T* WAVELETS_RESTRICT buf, const size_t n, const size_t level) {
				if (level == 0) return;
				
				size_t nd = n / 2;
				size_t ns = n - nd;

				// perform first level transform

				// deinterleave input into buffer
				deinterleave_input(it, ain, buf, n);

				// get pointers to s and d
				T* s = buf;
				T* d = buf + ns;
				// perform transform inplace on s and d (which have unit stride)
				trans::forward(s, d, ns, nd, 1, 1, 1);

				// copy d into output
				copy_output(it, d, aout, ns, n);
				
				// If I have more levels... keep going
				for (size_t lvl = 1; lvl < level; ++lvl) {
					// use "d" as the new buffer to deinterleave into (which should have enough space),
					// alloc_temp function adds an extra buffer element for odd transform lengths at each level)
					deinterleave(s, d, ns);
					nd = ns / 2;
					ns = ns - nd;
					
					// get pointers to the starts of s and d;
					s = d;
					d = d + ns;
					trans::forward(s, d, ns, nd, 1, 1, 1);
					
					// copy d into output;
					copy_output(it, d, aout, ns, ns + nd);
				}
				// copy last s into output;
				copy_output(it, s, aout, 0, ns);
			}

			template<typename T, size_t vlen>
			static inline void vec_apply(const multi_iter<vlen>& it, const cndarr<T>& ain, ndarr<T>& aout, T* WAVELETS_RESTRICT buf, const size_t n, const size_t level) {
				if (level == 0) return;

				size_t nd = n / 2;
				size_t ns = n - nd;

				vec_deinterleave_input(it, ain, buf, n);

				T* s = buf;
				T* d = buf + vlen * ns;
				trans::forward(s, d, ns, nd, vlen, vlen, vlen);

				vec_copy_output(it, d, aout, ns, n);

				for (size_t lvl = 1; lvl < level; ++lvl) {
					vec_deinterleave<T, vlen>(s, d, ns);

					nd = ns / 2;
					ns = ns - nd;

					s = d;
					d = s + vlen * ns;
					trans::forward(s, d, ns, nd, vlen, vlen, vlen);

					vec_copy_output(it, d, aout, ns, ns + nd);
				}
				vec_copy_output(it, s, aout, 0, ns);
			}
		};

		template<typename WVLT, typename BC>
		struct inverse_driver {
			using wavelet = WVLT;
			using trans = LiftingTransform<WVLT, BC>;
			const static int dir = -1;

			template<typename T, size_t vlen>
			static inline void apply(const multi_iter<vlen>& it, const cndarr<T>& ain, ndarr<T>& aout, T* WAVELETS_RESTRICT buf, const size_t n, const size_t n_levels = 1) {
				if (n_levels == 0) return;

				auto ns_s = std::vector<size_t>(n_levels);
				auto nd_s = std::vector<size_t>(n_levels);
				auto buffer_offsets = std::vector<size_t>(n_levels);
				size_t nd = n / 2;
				size_t ns = n - nd;
				size_t offsets = 0;
				for (size_t lvl = 0; lvl < n_levels; ++lvl) {
					buffer_offsets[lvl] = offsets;
					ns_s[lvl] = ns;
					nd_s[lvl] = nd;
					offsets += ns;
					nd = ns / 2;
					ns = ns - nd;
				}

				T* s = buf + buffer_offsets[n_levels - 1];
				T* d;
				ns = ns_s[n_levels - 1];
				// copy last level's s into the s_buffer;
				copy_input(it, ain, s, 0, ns);
				for (size_t lvl = n_levels; lvl-- > 1;) {  // loops to the second to last level
					ns = ns_s[lvl];
					nd = nd_s[lvl];

					d = s + ns;
					// copy this level's d into the d_buffer;
					copy_input(it, ain, d, ns, ns + nd);

					// transform this level in place
					trans::inverse(s, d, ns, nd, 1, 1, 1);
					
					// interleave s and d into the next s_buffer;
					T* next_s = buf + buffer_offsets[lvl - 1];
					interleave(s, next_s, ns + nd);
					s = next_s;
				}
				ns = ns_s[0];
				nd = nd_s[0];
				d = s + ns;
				// copy the last levels input into the d buffer;
				copy_input(it, ain, d, ns, ns + nd);

				trans::inverse(s, d, ns, nd, 1, 1, 1);

				// interleave the buffer into the output;
				interleave_output(it, buf, aout);
			}

			template<typename T, size_t vlen>
			static inline void vec_apply(const multi_iter<vlen>& it, const cndarr<T>& ain, ndarr<T>& aout, T* WAVELETS_RESTRICT buf, const size_t n, const size_t n_levels = 1) {
				if (n_levels == 0) return;

				auto ns_s = std::vector<size_t>(n_levels);
				auto nd_s = std::vector<size_t>(n_levels);
				auto buffer_offsets = std::vector<size_t>(n_levels);
				size_t nd = n / 2;
				size_t ns = n - nd;
				size_t offsets = 0;
				for (size_t lvl = 0; lvl < n_levels; ++lvl) {
					buffer_offsets[lvl] = offsets;
					ns_s[lvl] = ns;
					nd_s[lvl] = nd;
					offsets += ns;
					nd = ns / 2;
					ns = ns - nd;
				}

				T* s = buf + vlen * buffer_offsets[n_levels - 1];
				T* d;
				ns = ns_s[n_levels - 1];
				// copy last level's s into the s_buffer;
				vec_copy_input(it, ain, s, 0, ns);
				for (size_t lvl = n_levels; lvl-- > 1;) {  // loops to the second to last level
					ns = ns_s[lvl];
					nd = nd_s[lvl];

					d = s + vlen * ns;
					// copy this level's d into the d_buffer;
					vec_copy_input(it, ain, d, ns, ns + nd);

					// transform this level in place
					trans::inverse(s, d, ns, nd, vlen, vlen, vlen);

					T* next_s = buf + vlen * buffer_offsets[lvl - 1];
					// interleave s and d into the next s_buffer;
					vec_interleave<T, vlen>(s, next_s, ns + nd);
					s = next_s;
				}
				ns = ns_s[0];
				nd = nd_s[0];
				d = s + vlen * ns;
				// copy the last levels input into the d buffer;
				vec_copy_input(it, ain, d, ns, ns + nd);

				trans::inverse(s, d, ns, nd, vlen, vlen, vlen);

				// interleave the buffer into the output;
				vec_interleave_output(it, buf, aout);
			}
		};

		template<typename WVLT, typename BC>
		struct forward_adjoint_driver {
			using wavelet = WVLT;
			using trans = LiftingTransform<WVLT, BC>;
			const static int dir = -1;

			template<typename T, size_t vlen>
			static inline void apply(const multi_iter<vlen>& it, const cndarr<T>& ain, ndarr<T>& aout, T* WAVELETS_RESTRICT buf, const size_t n, const size_t n_levels = 1) {
				if (n_levels == 0) return;

				auto ns_s = std::vector<size_t>(n_levels);
				auto nd_s = std::vector<size_t>(n_levels);
				auto buffer_offsets = std::vector<size_t>(n_levels);
				size_t nd = n / 2;
				size_t ns = n - nd;
				size_t offsets = 0;
				for (size_t lvl = 0; lvl < n_levels; ++lvl) {
					buffer_offsets[lvl] = offsets;
					ns_s[lvl] = ns;
					nd_s[lvl] = nd;
					offsets += ns;
					nd = ns / 2;
					ns = ns - nd;
				}

				T* s = buf + buffer_offsets[n_levels - 1];
				T* d;
				ns = ns_s[n_levels - 1];
				// copy last level's s into the s_buffer;
				copy_input(it, ain, s, 0, ns);
				for (size_t lvl = n_levels; lvl-- > 1;) {  // loops to the second to last level
					ns = ns_s[lvl];
					nd = nd_s[lvl];

					d = s + ns;
					// copy this level's d into the d_buffer;
					copy_input(it, ain, d, ns, ns + nd);

					// transform this level in place
					trans::forward_adjoint(s, d, ns, nd, 1, 1, 1);

					// interleave s and d into the next s_buffer;
					T* next_s = buf + buffer_offsets[lvl - 1];
					interleave(s, next_s, ns + nd);
					s = next_s;
				}
				ns = ns_s[0];
				nd = nd_s[0];
				d = s + ns;
				// copy the last levels input into the d buffer;
				copy_input(it, ain, d, ns, ns + nd);

				trans::forward_adjoint(s, d, ns, nd, 1, 1, 1);

				// interleave the buffer into the output;
				interleave_output(it, buf, aout);
			}

			template<typename T, size_t vlen>
			static inline void vec_apply(const multi_iter<vlen>& it, const cndarr<T>& ain, ndarr<T>& aout, T* WAVELETS_RESTRICT buf, const size_t n, const size_t n_levels = 1) {
				if (n_levels == 0) return;

				auto ns_s = std::vector<size_t>(n_levels);
				auto nd_s = std::vector<size_t>(n_levels);
				auto buffer_offsets = std::vector<size_t>(n_levels);
				size_t nd = n / 2;
				size_t ns = n - nd;
				size_t offsets = 0;
				for (size_t lvl = 0; lvl < n_levels; ++lvl) {
					buffer_offsets[lvl] = offsets;
					ns_s[lvl] = ns;
					nd_s[lvl] = nd;
					offsets += ns;
					nd = ns / 2;
					ns = ns - nd;
				}

				T* s = buf + vlen * buffer_offsets[n_levels - 1];
				T* d;
				ns = ns_s[n_levels - 1];
				// copy last level's s into the s_buffer;
				vec_copy_input(it, ain, s, 0, ns);
				for (size_t lvl = n_levels; lvl-- > 1;) {  // loops to the second to last level
					ns = ns_s[lvl];
					nd = nd_s[lvl];

					d = s + vlen * ns;
					// copy this level's d into the d_buffer;
					vec_copy_input(it, ain, d, ns, ns + nd);

					// transform this level in place
					trans::forward_adjoint(s, d, ns, nd, vlen, vlen, vlen);

					T* next_s = buf + vlen * buffer_offsets[lvl - 1];
					// interleave s and d into the next s_buffer;
					vec_interleave<T, vlen>(s, next_s, ns + nd);
					s = next_s;
				}
				ns = ns_s[0];
				nd = nd_s[0];
				d = s + vlen * ns;
				// copy the last levels input into the d buffer;
				vec_copy_input(it, ain, d, ns, ns + nd);

				trans::forward_adjoint(s, d, ns, nd, vlen, vlen, vlen);

				// interleave the buffer into the output;
				vec_interleave_output(it, buf, aout);
			}
		};

		template<typename WVLT, typename BC>
		struct inverse_adjoint_driver {
			using wavelet = WVLT;
			using trans = LiftingTransform<WVLT, BC>;
			const static int dir = 1;

			template<typename T, size_t vlen>
			static inline void apply(const multi_iter<vlen>& it, const cndarr<T>& ain, ndarr<T>& aout, T* WAVELETS_RESTRICT buf, const size_t n, const size_t level) {
				if (level == 0) return;

				size_t nd = n / 2;
				size_t ns = n - nd;

				// perform first level transform

				// deinterleave input into buffer
				deinterleave_input(it, ain, buf, n);

				// get pointers to s and d
				T* s = buf;
				T* d = buf + ns;
				// perform transform inplace on s and d (which have unit stride)
				trans::inverse_adjoint(s, d, ns, nd, 1, 1, 1);

				// copy d into output
				copy_output(it, d, aout, ns, n);

				// If I have more levels... keep going
				for (size_t lvl = 1; lvl < level; ++lvl) {
					// use "d" as the new buffer to deinterleave into (which should have enough space),
					// alloc_temp function adds an extra buffer element for odd transform lengths at each level)
					deinterleave(s, d, ns);
					nd = ns / 2;
					ns = ns - nd;

					// get pointers to the starts of s and d;
					s = d;
					d = d + ns;
					trans::inverse_adjoint(s, d, ns, nd, 1, 1, 1);

					// copy d into output;
					copy_output(it, d, aout, ns, ns + nd);
				}
				// copy last s into output;
				copy_output(it, s, aout, 0, ns);
			}

			template<typename T, size_t vlen>
			static inline void vec_apply(const multi_iter<vlen>& it, const cndarr<T>& ain, ndarr<T>& aout, T* WAVELETS_RESTRICT buf, const size_t n, const size_t level) {
				if (level == 0) return;

				size_t nd = n / 2;
				size_t ns = n - nd;

				vec_deinterleave_input(it, ain, buf, n);

				T* s = buf;
				T* d = buf + vlen * ns;
				trans::inverse_adjoint(s, d, ns, nd, vlen, vlen, vlen);

				vec_copy_output(it, d, aout, ns, n);

				for (size_t lvl = 1; lvl < level; ++lvl) {
					vec_deinterleave<T, vlen>(s, d, ns);

					nd = ns / 2;
					ns = ns - nd;

					s = d;
					d = s + vlen * ns;
					trans::inverse_adjoint(s, d, ns, nd, vlen, vlen, vlen);

					vec_copy_output(it, d, aout, ns, ns + nd);
				}
				vec_copy_output(it, s, aout, 0, ns);
			}
		};

		template<typename WVLT>
		static size_t max_level(size_t n) {
			if (WVLT::width == 0) return 0;
			if (n < WVLT::width - 1) return 0;
			size_t lvl = 0;
			while (n >= 2 * (WVLT::width - 1)) {
				lvl += 1;
				n = (n + 1) / 2;  // If n is even, this is equivalent to n/2, if it is odd, then this is n - n/2
			}
			return lvl;
		}

		template<typename WVLT>
		static size_t max_level(const size_v& shape, const size_v& axes) {
			size_t min_n = shape[axes[0]];
			for (size_t i = 1; i < shape.size(); ++i)
				if (shape[axes[i]] < min_n) min_n = shape[axes[i]];
			return max_level<WVLT>(min_n);
		}

		template<typename driver, typename T>
		static void general_nd(
			const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
			const T* data_in, T* data_out, const size_t n_threads
		) {
			static constexpr auto vlen = VLEN<T>::val;

			auto ain = cndarr<T>(data_in, shape, stride_in);
			auto aout = ndarr<T>(data_out, shape, stride_out);

			for (size_t iax = 0; iax < axes.size(); ++iax) {
				size_t ax = axes[iax];
				size_t len = ain.shape(ax);
				size_t level = levels[iax];
				if (level == 0) level = max_level<typename driver::wavelet>(len);

				threading::thread_map(
					threading::thread_count(n_threads, ain.shape(), axes[iax], vlen),
					[&] {
						auto storage = alloc_tmp<T>(ain.shape(), len, level);
						const auto& tin(iax == 0 ? ain : aout);
						multi_iter<vlen> it(tin, aout, ax);

#ifndef WAVELETS_NO_VECTORS
						if (vlen > 1) {
							while (it.remaining() >= vlen) {
								it.advance(vlen);

								driver::vec_apply(it, tin, aout, storage.data(), len, level);
							}
						}
#endif
						while (it.remaining() > 0) {
							it.advance(1);
							driver::apply(it, tin, aout, storage.data(), len, level);
						}
					}
				);  // end of parallel region
			}
		}

		template<typename driver, typename T>
		static void general_nd(
			const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
			const T* data_in, T* data_out, const size_t n_threads
		) {
			static constexpr auto vlen = VLEN<T>::val;

			size_t lvl = level;
			if (lvl == 0) lvl = max_level<typename driver::wavelet>(shape, axes);

			auto shapes = std::vector<size_v>(lvl);
			shapes[0] = size_v(shape);
			for (size_t ilvl = 1; ilvl < lvl; ++ilvl){
				shapes[ilvl] = size_v(shapes[ilvl - 1]);
				auto& shape_ = shapes[ilvl];
				for (auto ax : axes) shape_[ax] = shape_[ax] - shape_[ax] / 2;
			}

			auto stride_in_ = stride_v(stride_in);

			if constexpr (driver::dir < 0) {
				if (data_in != data_out) {
					auto ain = cndarr<T>(data_in, shape, stride_in);
					auto aout = ndarr<T>(data_out, shape, stride_out);

					size_t min_ax = 0;
					stride_t min_stride = std::abs(stride_in[min_ax]);
					for (size_t iax = 1; iax < shape.size(); ++iax) {
						stride_t t_str = std::abs(stride_in[iax]);
						if (t_str < min_stride) {
							min_ax = iax;
							min_stride = t_str;
						}
					}

					threading::thread_map(
						threading::thread_count(n_threads, ain.shape(), min_ax, vlen),
						[&] {
							multi_iter<vlen> it(ain, aout, min_ax);

							while (it.remaining() > 0) {
								it.advance(1);
								copy_inout(it, ain, aout);
							}
						}
					);  // end of parallel region

					data_in = data_out;
					stride_in_ = stride_out;
				}
			}

			for (size_t ilvl = 0; ilvl < lvl; ++ilvl) {

				size_v shape_;
				if constexpr (driver::dir < 0) {
					shape_ = shapes[lvl - 1 - ilvl];
				}
				else {
					shape_ = shapes[ilvl];
				}

				auto ain = cndarr<T>(data_in, shape_, stride_in_);
				auto aout = ndarr<T>(data_out, shape_, stride_out);

				for (size_t iax = 0; iax < axes.size(); ++iax) {
					size_t ax = axes[iax];
					size_t len = ain.shape(ax);

					threading::thread_map(
						threading::thread_count(n_threads, ain.shape(), axes[iax], vlen),
						[&] {
							auto storage = alloc_tmp<T>(ain.shape(), len, level);
							const auto& tin(iax == 0 ? ain : aout);
							multi_iter<vlen> it(tin, aout, ax);

#ifndef WAVELETS_NO_VECTORS
							if (vlen > 1) {
								while (it.remaining() >= vlen) {
									it.advance(vlen);

									driver::vec_apply(it, tin, aout, storage.data(), len, 1);
								}
							}
#endif
							while (it.remaining() > 0) {
								it.advance(1);
								driver::apply(it, tin, aout, storage.data(), len, 1);
							}
						}
					);  // end of parallel region
				}

				data_in = data_out;
				stride_in_ = stride_out;
			}
		}

		template<typename T> class Daubechies1 {
			constexpr static T sc = T(1.41421356237309504880168872420969807856967187537694807317668L);
		public:
			using type = T;

			using steps = std::tuple<
				update_d<0, -1>,
				update_s<0, T(0.5)>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
			const static size_t width = 2;
		};

		template<typename T> class Daubechies2 {
		private:
			constexpr static T s1_p0 = T(-1.73205080756887729352744634150587236694280525381038062805581L);
			constexpr static T s2_p0 = T(0.433012701892219323381861585376468091735701313452595157013952L);
			constexpr static T s2_p1 = T(-0.0669872981077806766181384146235319082642986865474048429860483L);
			constexpr static T sc = T(1.93185165257813657349948639945779473526780967801680910080469L);
		public:
			using steps = std::tuple<
				update_d<0, s1_p0>,
				update_s<0, s2_p0, s2_p1>,
				update_d<-1, 1>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;

			using type = T;
			const static size_t width = 4;
		};

		template<typename T> class Daubechies3 {
		private:
			constexpr static T s1_p0  = T(2.42549724391195830853534735792805013148173539788185404597169L);

			constexpr static T s2_m0 = T(-0.352387657674855469319634307041734500360168717199626110489802L);
			constexpr static T s2_m1 = T(0.0793394561851575665709854468556681022464175974577020488100928L);

			constexpr static T s3_p1 = T(-2.89534745414509893429485976731702946529529368748135552662476L);
			constexpr static T s3_p2 = T(0.561414909153505374525726237570322812332695442156902611352901L);

			constexpr static T s4_m2 = T(-0.019750529242293006004979050052766598262001873036524279141228L);

			constexpr static T sc = T(0.431879991517282793698835833676951360096586647014001193148804L);

		public:

			using steps = std::tuple<
				update_s<0, s1_p0>,
				update_d<-1, s2_m1, s2_m0>,
				update_s<1, s3_p1, s3_p2>,
				update_d<-2, s4_m2>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
			const static size_t width = 6;

			using type = T;
		};

		template<typename T> class Daubechies4 {
		private:
			constexpr static T s1_p0 = T(-3.10293148583032589470063811758833883204289885795495381963662L);
			constexpr static T s2_p0 = T(0.353449187617773130690791641916614771198571210734232133909753L);
			constexpr static T s2_p1 = T(0.116026410344205376526201432171415851440295089908831794682144L);
			constexpr static T s3_p0 = T(-1.13273740414015604279309800164179185383471162488546344575547L);
			constexpr static T s3_p1 = T(0.181090277146292103917336470393403415884795293238801624079639L);
			constexpr static T s4_p0 = T(-0.0661005526900487668472230596604034876925334941709204294099145L);
			constexpr static T s4_p1 = T(-0.221414210158528887858396206711438754339106682715345633623713L);
			constexpr static T s5_m3 = T(0.23869459776013294525643502372428025694730153962927746023463L);
			constexpr static T s5_m2 = T(-1.34832352097184759074938562094799080953526636376300686976116L);
			constexpr static T s5_m1 = T(4.5164219554111574424668864244565045399107667587479110837686L);

			constexpr static T sc = T(2.27793811152208700221465706595215439338715747993503607887708L);

		public:

			using steps = std::tuple<
				update_d<0, s1_p0>,
				update_s<0, s2_p0, s2_p1>,
				update_d<0, s3_p0, s3_p1>,
				update_s<0, s4_p0, s4_p1>,
				update_d<-3, s5_m3, s5_m2, s5_m1>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;

			using type = T;
			const static size_t width = 8;
		};

		template<typename T> class Daubechies5 {
		private:
			constexpr static T s1_p0 = T(3.77151921168926894757552366178790881698024090635846510016028L);
			constexpr static T s2_m1 = T(0.0698842923341383375213513979361008347245073868657407740184605L);
			constexpr static T s2_p0 = T(-0.247729291360329682294999795251047782875275189599695961483240L);
			constexpr static T s3_p1 = T(-7.59757973540575405854153663362110608451687695913930970651012L);
			constexpr static T s3_p2 = T(3.03368979135389213275426841861823490779263829963269704719700L);
			constexpr static T s4_m3 = T(0.0157993238122452355950658463497927543465183810596906799882017L);
			constexpr static T s4_m2 = T(-0.0503963526115147652552264735912933014908335872176681805787989L);
			constexpr static T s5_p3 = T(-1.10314632826313117312599754225554155377077047715385824307618L);
			constexpr static T s5_p4 = T(0.172572555625945876870245572286755166281402778242781642128057L);
			constexpr static T s6_m4 = T(-0.002514343828207810116980039456958207159821965712982241017850L);

			constexpr static T sc = T(0.347389040193226710460416754871479443615028875117970218575475L);

		public:

			using steps = std::tuple<
				update_s< 0, s1_p0>,
				update_d<-1, s2_m1, s2_p0>,
				update_s< 1, s3_p1, s3_p2>,
				update_d<-3, s4_m3, s4_m2>,
				update_s< 3, s5_p3, s5_p4>,
				update_d<-4, s6_m4>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;

			using type = T;
			const static size_t width = 10;
		};

		template<typename T> class Daubechies6 {
		private:
			constexpr static T s1_p0 = T(-4.43446829869067456522902386901421595117929955784397002766726L);

			constexpr static T s2_p0 = T(0.214593450003008209156010000739498614149390366235009628657503L);
			constexpr static T s2_p1 = T(-0.0633131925460940277584543253045170400260191584783746185619816L);

			constexpr static T s3_m1 = T(9.97001561279871984191261047001888284698487613742269103059835L);
			constexpr static T s3_m2 = T(-4.49311316887101738480447295831022985715670304865394507715493L);

			constexpr static T s4_p2 = T(0.179649626662970342602164106271580972989375373440522637339242L);
			constexpr static T s4_p3 = T(0.400695200906159800144370836431846103438795499715994915842164L);

			constexpr static T s5_m1 = T(0.00550295326884471036285017850066003823120949674410943525332981L);
			constexpr static T s5_m2 = T(-0.0430453536805558366602898703076915479625247841579933421946333L);

			constexpr static T s6_p2 = T(-0.122882256743032948063403419686360741867410363679212929915483L);
			constexpr static T s6_p3 = T(-0.428776772090593906072415971370850703089199760064884803407522L);

			constexpr static T s7_m3 = T(2.3322158873585518129700437453502313783105215392342007896945L);
			constexpr static T s7_m4 = T(-0.6683849734985699469411475049338822868631946666405609745744L);
			constexpr static T s7_m5 = T(0.0922130623951882057243173036464618224518216084115197289578L);

			constexpr static T sc = T(3.08990011938602961884790035600227126232046198472673752023023L);

		public:

			using steps = std::tuple<
				update_d< 0, s1_p0>,
				update_s< 0, s2_p0, s2_p1>,
				update_d<-2, s3_m2, s3_m1>,
				update_s< 2, s4_p2, s4_p3>,
				update_d<-2, s5_m2, s5_m1>,
				update_s< 2, s6_p2, s6_p3>,
				update_d<-5, s7_m5, s7_m4, s7_m3>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;

			using type = T;
			const static size_t width = 12;
		};

		template<typename T> class Daubechies7 {
		private:
			constexpr static T s1_p0 = T(5.09349848430361493158448369309319682065288729718818488170086L);

			constexpr static T s2_m0 = T(-0.189042092071992120719115069286184373462757526526706607687359L);
			constexpr static T s2_m1 = T(0.0573987259809472316318163158166229520747344869480313379225195L);

			constexpr static T s3_p1 = T(-12.2854449967343086359903552773950797906521905606178682213636L);
			constexpr static T s3_p2 = T(5.95920876247802982415703053009457554606483451817620285866937L);

			constexpr static T s4_m2 = T(-0.0604278631317421373037302780741631438456816254617632977400450L);
			constexpr static T s4_m3 = T(0.0291354832120604111715672746297275711217338322811099054777142L);

			constexpr static T s5_p3 = T(-3.97071066749950430300434793946545058865144871092579097855408L);
			constexpr static T s5_p4 = T(1.56044025996325478842482192099426071408509934413107380064471L);

			constexpr static T s6_m4 = T(-0.0126913773899277263576544929765396597865052704453285425832652L);
			constexpr static T s6_m5 = T(0.00330657330114293172083753191386779325367868962178489518206046L);

			constexpr static T s7_p5 = T(-0.414198450444716993956500507305024397477047469411911796987340L);
			constexpr static T s7_p6 = T(0.0508158836433382836486473921674908683852045493076552967984957L);

			constexpr static T s8_m6 = T(-0.000406214488767545513490343915060150692174820527241063790923L);

			constexpr static T sc = T(0.299010707585297416977548850152193565671577389737970684943879L);

		public:

			using steps = std::tuple<
				update_s< 0, s1_p0>,
				update_d<-1, s2_m1, s2_m0>,
				update_s< 1, s3_p1, s3_p2>,
				update_d<-3, s4_m3, s4_m2>,
				update_s< 3, s5_p3, s5_p4>,
				update_d<-5, s6_m5, s6_m4>,
				update_s< 5, s7_p5, s7_p6>,
				update_d<-6, s8_m6>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;

			using type = T;
			const static size_t width = 14;
		};

		template<typename T> class Daubechies8 {
		private:
			constexpr static T s1_p0 = T(-5.74964161202428952531641742297874447770069863700611439780625L);

			constexpr static T s2_p0 = T(0.189042092071992120719115069286184373462757526526706607687359L);
			constexpr static T s2_p1 = T(-0.0522692017709505657615046200264570878753855329566801514443159L);

			constexpr static T s3_m1 = T(14.5428209935956353268617650552874052350090521140251529737701L);
			constexpr static T s3_m2 = T(-7.40210683018195497281259144457822959078824042693518644132815L);

			constexpr static T s4_p2 = T(0.0609092565131526712500223989293871769934896033103158195108610L);
			constexpr static T s4_p3 = T(-0.0324020739799994261972421468359233560093660827487021893317097L);

			constexpr static T s5_m3 = T(5.81871648960616924395028985550090363087448278263304640355032L);
			constexpr static T s5_m4 = T(-2.75569878011462525067005258197637909961926722730653432065204L);

			constexpr static T s6_p4 = T(0.0179741053616847069172743389156784247398807427207560577921141L);
			constexpr static T s6_p5 = T(-0.00655821883883417186334513866087578753320413229517368432598099L);

			constexpr static T s7_m5 = T(1.05058183627305042173272934526705555053668924546503033598454L);
			constexpr static T s7_m6 = T(-0.247286508228382323711914584543135429166229548148776820566842L);

			constexpr static T s8_p6 = T(0.00155378941435181290328472355847245468306658119261351754857778L);
			constexpr static T s8_p7 = T(-0.000171095706397052251935603348141260065760590013485603128919513L);

			constexpr static T s9_m7 = T(0.0272403229721260106412588252370521928932024767680952595018L);

			constexpr static T sc = T(3.55216212499884228308604950160526405248328786633495753507726L);

		public:

			using steps = std::tuple<
				update_d< 0, s1_p0>,
				update_s< 0, s2_p0, s2_p1>,
				update_d<-2, s3_m2, s3_m1>,
				update_s< 2, s4_p2, s4_p3>,
				update_d<-4, s5_m4, s5_m3>,
				update_s< 4, s6_p4, s6_p5>,
				update_d<-6, s7_m6, s7_m5>,
				update_s< 6, s8_p6, s8_p7>,
				update_d<-7, s9_m7>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;

			using type = T;
			const static size_t width = 16;
		};

		template<typename T> class BiorSpline3_1 {
			constexpr static T s1_m1 = T(-0.333333333333333333333333333333333333333333333333333333333333L);
			constexpr static T s2_p0 = T(-1.125L);
			constexpr static T s2_p1 = T(-0.375L);
			constexpr static T s3_p0 = T(0.444444444444444444444444444444444444444444444444444444444444L);
			constexpr static T sc = T(2.12132034355964257320253308631454711785450781306542210976502L);
		public:
			using type = T;

			using steps = std::tuple<
				update_s<-1, s1_m1>,
				update_d<0, s2_p0, s2_p1>,
				update_s<0, s3_p0>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
			const static size_t width = 4;
		};

		template<typename T> class ReverseBiorSpline3_1 {
			constexpr static T s1_p1 = T(0.333333333333333333333333333333333333333333333333333333333333L);
			constexpr static T s2_p0 = T(1.125L);
			constexpr static T s2_m1 = T(0.375L);
			constexpr static T s3_p0 = T(-0.444444444444444444444444444444444444444444444444444444444444L);
			constexpr static T sc = T(0.471404520791031682933896241403232692856557291792316024392227L);
		public:
			using type = T;

			using steps = std::tuple<
				update_d<1, s1_p1>,
				update_s<-1, s2_m1, s2_p0>,
				update_d<0, s3_p0>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
			const static size_t width = 4;
		};

		template<typename T> class BiorSpline4_2 {
			constexpr static T s1 = T(-0.25L);
			constexpr static T s3 = T(0.1875L);
			constexpr static T sc = T(2.82842712474619009760337744841939615713934375075389614635336L);
		public:
			using type = T;

			using steps = std::tuple<
				update_s<0, s1, s1>,
				update_d<-1, -1, -1>,
				update_s<0, s3, s3>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
			const static size_t width = 7;
		};

		template<typename T> class ReverseBiorSpline4_2 {
			constexpr static T s1 = T(0.25L);
			constexpr static T s3 = T(-0.1875L);
			constexpr static T sc = T(0.353553390593273762200422181052424519642417968844237018294170L);
		public:
			using type = T;

			using steps = std::tuple<
				update_d<0, s1, s1>,
				update_s<-1, 1, 1>,
				update_d<0, s3, s3>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
			const static size_t width = 7;
		};

		template<typename T> class BiorSpline2_4 {
			constexpr static T s1 = T(-0.5L);
			constexpr static T s2_0 = T(0.296875L);
			constexpr static T s2_1 = T(-0.046875L);
			constexpr static T sc = T(1.41421356237309504880168872420969807856967187537694807317668L);
		public:
			using type = T;

			using steps = std::tuple<
				update_d<0, s1, s1>,
				update_s<-2, s2_1, s2_0, s2_0, s2_1>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
			const static size_t width = 9;
		};

		template<typename T> class ReverseBiorSpline2_4 {
			constexpr static T s1 = T(0.5L);
			constexpr static T s2_0 = T(-0.296875L);
			constexpr static T s2_1 = T(0.046875L);
			constexpr static T sc = T(0.707106781186547524400844362104849039284835937688474036588340L);
		public:
			using type = T;

			using steps = std::tuple<
				update_s<0, s1, s1>,
				update_d<-1, s2_1, s2_0, s2_0, s2_1>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
			const static size_t width = 9;
		};

		template<typename T> class BiorSpline6_2 {
			constexpr static T s1 = T(-0.166666666666666666666666666666666666666666666666666666666667L);
			constexpr static T s2 = T(-0.5625L);
			constexpr static T s3 = T(-1.33333333333333333333333333333333333333333333333333333333333L);
			constexpr static T s4 = T(0.15625L);
			constexpr static T sc = T(5.65685424949238019520675489683879231427868750150779229270672L);
		public:
			using type = T;

			using steps = std::tuple<
				update_d<0, s1, s1>,
				update_s<-1, s2, s2>,
				update_d<0, s3, s3>,
				update_s<-1, s4, s4>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
			const static size_t width = 9;
		};

		template<typename T> class ReverseBiorSpline6_2 {
			constexpr static T s1 = T(0.166666666666666666666666666666666666666666666666666666666667L);
			constexpr static T s2 = T(0.5625L);
			constexpr static T s3 = T(1.33333333333333333333333333333333333333333333333333333333333L);
			constexpr static T s4 = T(-0.15625L);
			constexpr static T sc = T(0.176776695296636881100211090526212259821208984422118509147085L);
		public:
			using type = T;

			using steps = std::tuple<
				update_s<-1, s1, s1>,
				update_d<0, s2, s2>,
				update_s<-1, s3, s3>,
				update_d<0, s4, s4>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
			const static size_t width = 9;
		};

		template<typename T> class CDF5_3 {
			constexpr static T s1 = T(0.5L);
			constexpr static T s2 = T(-0.25L);
			constexpr static T sc = T(0.707106781186547524400844362104849039284835937688474036588340L);
		public:
			using type = T;

			using steps = std::tuple<
				update_s<-1, s1, s1>,
				update_d<0, s2, s2>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
			const static size_t width = 5;
		};

		template<typename T> class ReverseCDF5_3 {
			constexpr static T s1 = T(0.5L);
			constexpr static T s2 = T(-0.25L);
			constexpr static T sc = T(0.707106781186547524400844362104849039284835937688474036588340L);
		public:
			using type = T;

			using steps = std::tuple<
				update_d<0, s1, s1>,
				update_s<-1, s2, s2>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
			const static size_t width = 5;
		};

		template<typename T> class CDF9_7{
			constexpr static T s1 = T(-1.58613434205992355842831545133740131985598525529112656778527L);
			constexpr static T s2 = T(-0.0529801185729614146241295675034771089920773921055534971880099L);
			constexpr static T s3 = T(0.882911075530933295919790099002837930341944716149740640164429L);
			constexpr static T s4 = T(0.443506852043971152115604215168913719478036852964167569567164L);
			constexpr static T sc = T(1.14960439886024115979507564219148965843436742907448991688182L);
		public:
			using type = T;

			using steps = std::tuple<
				update_d<0, s1, s1>,
				update_s<-1, s2, s2>,
				update_d<0, s3, s3>,
				update_s<-1, s4, s4>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
			const static size_t width = 9;
		};

		template<typename T> class ReverseCDF9_7 {
			constexpr static T s1 = T(1.58613434205992355842831545133740131985598525529112656778527L);
			constexpr static T s2 = T(0.0529801185729614146241295675034771089920773921055534971880099L);
			constexpr static T s3 = T(-0.882911075530933295919790099002837930341944716149740640164429L);
			constexpr static T s4 = T(-0.443506852043971152115604215168913719478036852964167569567164L);
			constexpr static T sc = T(0.86986445162478127129589381456403049869640519768690701998711L);
		public:
			using type = T;

			using steps = std::tuple<
				update_s<-1, s1, s1>,
				update_d<0, s2, s2>,
				update_s<-1, s3, s3>,
				update_d<0, s4, s4>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
			const static size_t width = 9;
		};

		enum class Wavelet {
			Daubechies1,
			Daubechies2,
			Daubechies3,
			Daubechies4,
			Daubechies5,
			Daubechies6,
			Daubechies7,
			Daubechies8,
			BiorSpline3_1,
			BiorSpline4_2,
			BiorSpline2_4,
			BiorSpline6_2,
			CDF5_3,
			CDF9_7
		};

		enum class BoundaryCondition {
			ZERO,
			PERIODIC,
			CONSTANT,
			SYMMETRIC,
			REFLECT
		};

		template<typename WVLT, typename BC>
		struct ForwardTransform {
			using driver = forward_driver<WVLT, BC>;

			template<typename T>
			static void apply(
				const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
				const T* data_in, T* data_out, const size_t n_threads = 1
			) {
				general_nd<driver>(shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
			}

			template<typename T>
			static void apply(
				const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
				const T* data_in, T* data_out, const size_t n_threads = 1
			) {
				general_nd<driver>(shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
			}
		};

		template<typename WVLT, typename BC>
		struct ForwardAdjointTransform {
			using driver = forward_adjoint_driver<WVLT, BC>;

			template<typename T>
			static void apply(
				const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
				const T* data_in, T* data_out, const size_t n_threads = 1
			) {
				general_nd<driver>(shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
			}

			template<typename T>
			static void apply(
				const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
				const T* data_in, T* data_out, const size_t n_threads = 1
			) {
				general_nd<driver>(shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
			}
		};

		template<typename WVLT, typename BC>
		struct InverseTransform {
			using driver = inverse_driver<WVLT, BC>;

			template<typename T>
			static void apply(
				const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
				const T* data_in, T* data_out, const size_t n_threads = 1
			) {
				general_nd<driver>(shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
			}

			template<typename T>
			static void apply(
				const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
				const T* data_in, T* data_out, const size_t n_threads = 1
			) {
				general_nd<driver>(shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
			}
		};

		template<typename WVLT, typename BC>
		struct InverseAdjointTransform {
			using driver = inverse_adjoint_driver<WVLT, BC>;

			template<typename T>
			static void apply(
				const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
				const T* data_in, T* data_out, const size_t n_threads = 1
			) {
				general_nd<driver>(shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
			}

			template<typename T>
			static void apply(
				const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
				const T* data_in, T* data_out, const size_t n_threads = 1
			) {
				general_nd<driver>(shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
			}
		};

		template<typename WVLT, template<typename, typename> class OP>
		struct BCDispatch {

			template<typename T>
			static void apply(
				const BoundaryCondition bc,
				const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
				const T* data_in, T* data_out, size_t n_threads = 1
			) {
				switch (bc) {
				case BoundaryCondition::ZERO:
					OP<WVLT, ZeroBoundary>::apply(shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case BoundaryCondition::PERIODIC:
					OP<WVLT, PeriodicBoundary>::apply(shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case BoundaryCondition::CONSTANT:
					OP<WVLT, ConstantBoundary>::apply(shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case BoundaryCondition::SYMMETRIC:
					OP<WVLT, SymmetricBoundary>::apply(shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case BoundaryCondition::REFLECT:
					OP<WVLT, ReflectBoundary>::apply(shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				}
			}

			template<typename T>
			static void apply(
				const BoundaryCondition bc,
				const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
				const T* data_in, T* data_out, size_t n_threads = 1
			) {
				switch (bc) {
				case BoundaryCondition::ZERO:
					OP<WVLT, ZeroBoundary>::apply(shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case BoundaryCondition::PERIODIC:
					OP<WVLT, PeriodicBoundary>::apply(shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case BoundaryCondition::CONSTANT:
					OP<WVLT, ConstantBoundary>::apply(shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case BoundaryCondition::SYMMETRIC:
					OP<WVLT, SymmetricBoundary>::apply(shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case BoundaryCondition::REFLECT:
					OP<WVLT, ReflectBoundary>::apply(shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				}
			}
		};

		template<template<typename, typename> class OP>
		struct WVLTBCDispatch {

			template<typename T>
			static void apply(
				const Wavelet wvlt, const BoundaryCondition bc,
				const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
				const T* data_in, T* data_out, size_t n_threads = 1
			) {
				switch (wvlt) {
					// Daubechies Wavelets
				case Wavelet::Daubechies1:
					BCDispatch<Daubechies1<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case Wavelet::Daubechies2:
					BCDispatch<Daubechies2<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case Wavelet::Daubechies3:
					BCDispatch<Daubechies3<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case Wavelet::Daubechies4:
					BCDispatch<Daubechies4<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case Wavelet::Daubechies5:
					BCDispatch<Daubechies5<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case Wavelet::Daubechies6:
					BCDispatch<Daubechies6<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case Wavelet::Daubechies7:
					BCDispatch<Daubechies7<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case Wavelet::Daubechies8:
					BCDispatch<Daubechies8<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
					// Bior Wavelets
				case Wavelet::BiorSpline3_1:
					BCDispatch<BiorSpline3_1<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case Wavelet::BiorSpline4_2:
					BCDispatch<BiorSpline4_2<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case Wavelet::BiorSpline6_2:
					BCDispatch<BiorSpline6_2<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case Wavelet::BiorSpline2_4:
					BCDispatch<BiorSpline2_4<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case Wavelet::CDF5_3:
					BCDispatch<CDF5_3<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				case Wavelet::CDF9_7:
					BCDispatch<CDF9_7<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads);
					break;
				}
			}

			template<typename T>
			static void apply(
				const Wavelet wvlt, const BoundaryCondition bc,
				const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
				const T* data_in, T* data_out, size_t n_threads = 1
			) {
				switch (wvlt) {
					// Daubechies Wavelets
				case Wavelet::Daubechies1:
					BCDispatch<Daubechies1<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case Wavelet::Daubechies2:
					BCDispatch<Daubechies2<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case Wavelet::Daubechies3:
					BCDispatch<Daubechies3<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case Wavelet::Daubechies4:
					BCDispatch<Daubechies4<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case Wavelet::Daubechies5:
					BCDispatch<Daubechies5<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case Wavelet::Daubechies6:
					BCDispatch<Daubechies6<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case Wavelet::Daubechies7:
					BCDispatch<Daubechies7<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case Wavelet::Daubechies8:
					BCDispatch<Daubechies8<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
					// Bior Wavelets
				case Wavelet::BiorSpline3_1:
					BCDispatch<BiorSpline3_1<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case Wavelet::BiorSpline4_2:
					BCDispatch<BiorSpline4_2<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case Wavelet::BiorSpline6_2:
					BCDispatch<BiorSpline6_2<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case Wavelet::BiorSpline2_4:
					BCDispatch<BiorSpline2_4<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case Wavelet::CDF5_3:
					BCDispatch<CDF5_3<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				case Wavelet::CDF9_7:
					BCDispatch<CDF9_7<T>, OP>::apply(bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads);
					break;
				}
			}
		};

		template<typename T>
		static void lwt(
			const Wavelet wvlt, const BoundaryCondition bc,
			const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
			const T* data_in, T* data_out, size_t n_threads = 1
		) {
			WVLTBCDispatch<ForwardTransform>::apply(
				wvlt, bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads
			);
		}

		template<typename T>
		static void lwt(
			const Wavelet wvlt, const BoundaryCondition bc,
			const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
			const T* data_in, T* data_out, size_t n_threads = 1
		) {
			WVLTBCDispatch<ForwardTransform>::apply(
				wvlt, bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads
			);
		}

		template<typename T>
		static void lwt_adjoint(
			const Wavelet wvlt, const BoundaryCondition bc,
			const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
			const T* data_in, T* data_out, size_t n_threads = 1
		) {
			WVLTBCDispatch<ForwardAdjointTransform>::apply(
				wvlt, bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads
			);
		}

		template<typename T>
		static void lwt_adjoint(
			const Wavelet wvlt, const BoundaryCondition bc,
			const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
			const T* data_in, T* data_out, size_t n_threads = 1
		) {
			WVLTBCDispatch<ForwardAdjointTransform>::apply(
				wvlt, bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads
			);
		}

		template<typename T>
		static void ilwt(
			const Wavelet wvlt, const BoundaryCondition bc,
			const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
			const T* data_in, T* data_out, size_t n_threads = 1
		) {
			WVLTBCDispatch<InverseTransform>::apply(
				wvlt, bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads
			);
		}

		template<typename T>
		static void ilwt(
			const Wavelet wvlt, const BoundaryCondition bc,
			const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
			const T* data_in, T* data_out, size_t n_threads = 1
		) {
			WVLTBCDispatch<InverseTransform>::apply(
				wvlt, bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads
			);
		}

		template<typename T>
		static void ilwt_adjoint(
			const Wavelet wvlt, const BoundaryCondition bc,
			const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
			const T* data_in, T* data_out, size_t n_threads = 1
		) {
			WVLTBCDispatch<InverseAdjointTransform>::apply(
				wvlt, bc, shape, stride_in, stride_out, axes, levels, data_in, data_out, n_threads
			);
		}

		template<typename T>
		static void ilwt_adjoint(
			const Wavelet wvlt, const BoundaryCondition bc,
			const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
			const T* data_in, T* data_out, size_t n_threads = 1
		) {
			WVLTBCDispatch<InverseAdjointTransform>::apply(
				wvlt, bc, shape, stride_in, stride_out, axes, level, data_in, data_out, n_threads
			);
		}


		static inline size_t max_level(const Wavelet wvlt, size_t n) {
			switch (wvlt) {
			case Wavelet::Daubechies1: return max_level<Daubechies1<float>>(n);
			case Wavelet::Daubechies2: return max_level<Daubechies2<float>>(n);
			case Wavelet::Daubechies3: return max_level<Daubechies3<float>>(n);
			case Wavelet::Daubechies4: return max_level<Daubechies4<float>>(n);
			case Wavelet::Daubechies5: return max_level<Daubechies5<float>>(n);
			case Wavelet::Daubechies6: return max_level<Daubechies6<float>>(n);
			case Wavelet::Daubechies7: return max_level<Daubechies7<float>>(n);
			case Wavelet::Daubechies8: return max_level<Daubechies8<float>>(n);
			case Wavelet::BiorSpline3_1: return max_level<BiorSpline3_1<float>>(n);
			case Wavelet::BiorSpline4_2: return max_level<BiorSpline4_2<float>>(n);
			case Wavelet::BiorSpline6_2: return max_level<BiorSpline6_2<float>>(n);
			case Wavelet::BiorSpline2_4: return max_level<BiorSpline2_4<float>>(n);
			case Wavelet::CDF5_3: return max_level<CDF5_3<float>>(n);
			case Wavelet::CDF9_7: return max_level<CDF9_7<float>>(n);
			}
		}
	}

#ifndef WAVELETS_NO_VECTORS
	const static bool vector_support = true;
#if (defined(__AVX512F__))
    const static size_t vector_byte_length = 64;
#elif (defined(__AVX__))
	const static size_t vector_byte_length = 32;
#elif (defined(__SSE2__))
    const static size_t vector_byte_length = 16;
#elif (defined(__VSX__))
	const static size_t vector_byte_length = 16;
#elif (defined(__ARM_NEON__) || defined(__ARM_NEON))
    const static size_t vector_byte_length = 16;
#endif
#else
    const static bool vector_support = false;
	const static size_t vector_byte_length = 0;
#endif

	// Hard Coded Wavelets
	using detail::Daubechies1;
	using detail::Daubechies2;
	using detail::Daubechies3;
	using detail::Daubechies4;
	using detail::Daubechies5;
	using detail::Daubechies6;
	using detail::Daubechies7;

	using detail::BiorSpline3_1;
	using detail::BiorSpline4_2;
	using detail::BiorSpline2_4;
	using detail::BiorSpline6_2;
	using detail::ReverseBiorSpline3_1;
	using detail::ReverseBiorSpline4_2;
	using detail::ReverseBiorSpline2_4;
	using detail::ReverseBiorSpline6_2;
	using detail::CDF5_3;
	using detail::ReverseCDF5_3;
	using detail::CDF9_7;
	using detail::ReverseCDF9_7;

	// Define Haar
	template<typename T>
	using Haar = Daubechies1<T>;

	using detail::Wavelet;

	// Boundary Conditions
	using detail::ZeroBoundary;
	using detail::ConstantBoundary;
	using detail::PeriodicBoundary;
	using detail::SymmetricBoundary;
	using detail::ReflectBoundary;

	using detail::BoundaryCondition;

	// The transform driver
	using detail::LiftingTransform;

	// Main transform functions
	using detail::max_level;

	using detail::ForwardTransform;
	using detail::InverseTransform;
	using detail::ForwardAdjointTransform;
	using detail::InverseAdjointTransform;

	using detail::lwt;
	using detail::lwt_adjoint;
	using detail::ilwt;
	using detail::ilwt_adjoint;
}

#endif