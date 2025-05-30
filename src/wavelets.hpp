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

namespace wavelets {
	namespace detail {

		using std::size_t;
		using std::cout;
		using std::endl;

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
				if ((i < 0) || (i >= n)) {
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
				else if (i >= n) {
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
				while ((io >= n) || (io < 0)) {
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
				while ((io >= n) || (io < 0)) {
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
			static inline V apply(const V* x, const size_t i) {
				return val * x[i + offset] + Lift<offset + 1, vals...>::apply(x, i);
			}

			template<typename BC, typename V>
			static inline V bc_apply(const V* x, const ptrdiff_t i, const size_t n) {
				size_t i_bc = BC::index_of(i + offset, n);
				if (i_bc == n) {
					return Lift<offset + 1, vals...>::template bc_apply<BC>(x, i, n);
				}
				else {
					return val * x[i_bc] + Lift<offset + 1, vals...>::template bc_apply<BC>(x, i, n);
				}
			}

			template<typename V>
			static inline V bc_adj_apply(const V* x, const ptrdiff_t i, const size_t n) {
				if (i == n) {
					return V(0.0);
				}
				else if (i < 0) {
					return Lift<offset + 1, vals...>::bc_adj_apply(x, i + 1, n);
				}
				else {
					return val * x[i] + Lift<offset + 1, vals...>::bc_adj_apply(x, i + 1, n);
				}
			}
		};

		template<ptrdiff_t offset, auto val>
		struct Lift<offset, val> {

			const static size_t n_vals = 1;

			template<typename V>
			static inline V apply(const V* x, const size_t i) {
				return val * x[i + offset];
			}

			template<typename BC, typename V>
			static inline V bc_apply(const V* x, const ptrdiff_t i, const size_t n) {
				size_t i_bc = BC::index_of(i + offset, n);
				if (i_bc == n) {
					return V(0.0);
				}
				else {
					return val * x[i_bc];
				}
			}

			template<typename V>
			static inline V bc_adj_apply(const V* x, const ptrdiff_t i, const size_t n) {
				if ((i == n) || (i < 0)){
					return V(0.0);
				}
				else {
					return val * x[i];
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

			constexpr static ptrdiff_t max_offset = n_vals - 1 + offset;
			constexpr static size_t n_front = (offset < 0) ? -offset : 0;
			constexpr static size_t n_back = (max_offset < 0) ? 0 : max_offset;

			constexpr static ptrdiff_t offset_r = -max_offset;

			constexpr static ptrdiff_t max_offset_r = n_vals - 1 + offset_r;
			constexpr static size_t n_front_r = (offset_r < 0) ? -offset_r : 0;
			constexpr static size_t n_back_r = (max_offset_r < 0) ? 0 : max_offset_r;

			using lifter = Lift<offset, vals...>;
			using lifter_r = ReversedLift<offset_r, vals...>;

			template<typename BC, typename V>
			static inline void forward(const V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {

				for (ptrdiff_t i = 0; i < n_front; ++i)
					d[i] += lifter::template bc_apply<BC>(s, i, ns);
				for (size_t i = n_front; i < nd - n_back; ++i)
					d[i] += lifter::apply(s, i);
				for (ptrdiff_t i = nd - n_back; i < nd; ++i)
					d[i] += lifter::template bc_apply<BC>(s, i, ns);
			}

			template<typename BC, typename V>
			static inline void inverse(const V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {

				for (ptrdiff_t i = 0; i < n_front; ++i)
					d[i] -= lifter::template bc_apply<BC>(s, i, ns);
				for (size_t i = n_front; i < nd - n_back; ++i)
					d[i] -= lifter::apply(s, i);
				for (ptrdiff_t i = nd - n_back; i < nd; ++i)
					d[i] -= lifter::template bc_apply<BC>(s, i, ns);
			}

			template<typename BC, typename V>
			static inline void forward_adjoint(V* WAVELETS_RESTRICT s, const V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {
				
				for (ptrdiff_t i = offset, j = 1 - n_vals; i < 0; ++i, ++j) {
					size_t io = BC::index_of(i, ns);
					if ((io != ns) && (io != i)) {
						s[io] += lifter_r::bc_adj_apply(d, j, nd);
					}
				}

				for (ptrdiff_t i = 0; i < n_front_r; ++i)
					s[i] += lifter_r::template bc_apply<ZeroBoundary>(d, i, nd);
				for (size_t i = n_front_r; i < nd - n_back_r; ++i)
					s[i] += lifter_r::apply(d, i);
				for (ptrdiff_t i = nd - n_back_r; i < ns; ++i)
					s[i] += lifter_r::template bc_apply<ZeroBoundary>(d, i, nd);

				for (ptrdiff_t i = nd, j = nd + offset_r; i < nd - offset_r; ++i, ++j) {
					size_t io = BC::index_of(i, ns);
					if ((io != ns) && (io != i)) {
						s[io] += lifter_r::bc_adj_apply(d, j, nd);
					}
				}
			}

			template<typename BC, typename V>
			static inline void inverse_adjoint(V* WAVELETS_RESTRICT s, const V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {

				for (ptrdiff_t i = offset, j = 1 - n_vals; i < 0; ++i, ++j) {
					size_t io = BC::index_of(i, ns);
					if ((io != ns) && (io != i)) {
						s[io] -= lifter_r::bc_adj_apply(d, j, nd);
					}
				}

				for (ptrdiff_t i = 0; i < n_front_r; ++i)
					s[i] -= lifter_r::template bc_apply<ZeroBoundary>(d, i, nd);
				for (size_t i = n_front_r; i < nd - n_back_r; ++i)
					s[i] -= lifter_r::apply(d, i);
				for (ptrdiff_t i = nd - n_back_r; i < ns; ++i)
					s[i] -= lifter_r::template bc_apply<ZeroBoundary>(d, i, nd);

				for (ptrdiff_t i = nd, j = nd + offset_r; i < nd - offset_r; ++i, ++j) {
					size_t io = BC::index_of(i, ns);
					if ((io != ns) && (io != i)) {
						s[io] -= lifter_r::bc_adj_apply(d, j, nd);
					}
				}
			}
		};

		template<ptrdiff_t offset, auto... vals>
		struct update_s {
			constexpr static size_t n_vals = sizeof...(vals);

			constexpr static ptrdiff_t max_offset = n_vals - 1 + offset;
			constexpr static size_t n_front = (offset < 0) ? -offset : 0;
			constexpr static size_t n_back = (max_offset < 0) ? 0 : max_offset;

			constexpr static ptrdiff_t offset_r = -max_offset;

			constexpr static ptrdiff_t max_offset_r = n_vals - 1 + offset_r;
			constexpr static size_t n_front_r = (offset_r < 0) ? -offset_r : 0;
			constexpr static size_t n_back_r = (max_offset_r < 0) ? 0 : max_offset_r;

			using lifter = Lift<offset, vals...>;
			using lifter_r = ReversedLift<offset_r, vals...>;

			template<typename BC, typename V>
			static inline void forward(V* WAVELETS_RESTRICT s, const V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {

				for (ptrdiff_t i = 0; i < n_front; ++i)
					s[i] += lifter::template bc_apply<BC>(d, i, nd);
				for (size_t i = n_front; i < nd - n_back; ++i)
					s[i] += lifter::apply(d, i);
				for (ptrdiff_t i = nd - n_back; i < ns; ++i)
					s[i] += lifter::template bc_apply<BC>(d, i, nd);
			}

			template<typename BC, typename V>
			static inline void inverse(V* WAVELETS_RESTRICT s, const V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {

				for (ptrdiff_t i = 0; i < n_front; ++i)
					s[i] -= lifter::template bc_apply<BC>(d, i, nd);
				for (size_t i = n_front; i < nd - n_back; ++i)
					s[i] -= lifter::apply(d, i);
				for (ptrdiff_t i = nd - n_back; i < ns; ++i)
					s[i] -= lifter::template bc_apply<BC>(d, i, nd);
			}

			template<typename BC, typename V>
			static inline void forward_adjoint(const V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {

				for (ptrdiff_t i = offset, j = 1 - n_vals; i < 0; ++i, ++j) {
					size_t io = BC::index_of(i, nd);
					if ((io != nd) && (io != i)) {
						d[io] += lifter_r::bc_adj_apply(s, j, ns);
					}
				}

				for (ptrdiff_t i = 0; i < n_front_r; ++i)
					d[i] += lifter_r::template bc_apply<ZeroBoundary>(s, i, ns);
				for (size_t i = n_front_r; i < nd - n_back_r; ++i)
					d[i] += lifter_r::apply(s, i);
				for (ptrdiff_t i = nd - n_back_r; i < nd; ++i)
					d[i] += lifter_r::template bc_apply<ZeroBoundary>(s, i, ns);


				for (ptrdiff_t i = nd, j = nd + offset_r; i < ns - offset_r; ++i, ++j) {
					size_t io = BC::index_of(i, nd);
					if ((io != nd) && (io != i)) {
						d[io] += lifter_r::bc_adj_apply(s, j, nd);
					}
				}
			}

			template<typename BC, typename V>
			static inline void inverse_adjoint(const V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {

				for (ptrdiff_t i = offset, j = 1 - n_vals; i < 0; ++i, ++j) {
					size_t io = BC::index_of(i, nd);
					if ((io != nd) && (io != i)) {
						d[io] -= lifter_r::bc_adj_apply(s, j, ns);
					}
				}

				for (ptrdiff_t i = 0; i < n_front_r; ++i)
					d[i] -= lifter_r::template bc_apply<ZeroBoundary>(s, i, ns);
				for (size_t i = n_front_r; i < nd - n_back_r; ++i)
					d[i] -= lifter_r::apply(s, i);
				for (ptrdiff_t i = nd - n_back_r; i < nd; ++i)
					d[i] -= lifter_r::template bc_apply<ZeroBoundary>(s, i, ns);


				for (ptrdiff_t i = nd, j = nd + offset_r; i < ns - offset_r; ++i, ++j) {
					size_t io = BC::index_of(i, nd);
					if ((io != nd) && (io != i)) {
						d[io] -= lifter_r::bc_adj_apply(s, j, nd);
					}
				}
			}
		};

		template<auto val>
		struct scale {
			template<typename BC, typename V>
			static inline void forward(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd)
			{
				for (size_t i = 0; i < nd; ++i)
					d[i] /= val;
				for (size_t i = 0; i < ns; ++i)
					s[i] *= val;
			}

			template<typename BC, typename V>
			static inline void inverse(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd)
			{
				for (size_t i = 0; i < nd; ++i)
					d[i] *= val;
				for (size_t i = 0; i < ns; ++i)
					s[i] /= val;
			}

			template<typename BC, typename V>
			static inline void forward_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd)
			{
				forward<BC>(s, d, ns, nd);
			}

			template<typename BC, typename V>
			static inline void inverse_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd)
			{
				inverse<BC>(s, d, ns, nd);
			}
		};

		template<typename WVLT, typename BC, size_t step, size_t n_step>
		struct step_builder {

			template<typename V>
			static inline void forward(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {

				std::tuple_element_t<step, typename WVLT::steps>::template forward<BC>(s, d, ns, nd);
				step_builder<WVLT, BC, step + 1, n_step>::forward(s, d, ns, nd);
			}

			template<typename V>
			static inline void inverse(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {

				step_builder<WVLT, BC, step + 1, n_step>::inverse(s, d, ns, nd);
				std::tuple_element_t<step, typename WVLT::steps>::template inverse<BC>(s, d, ns, nd);
			}

			template<typename V>
			static inline void forward_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {

				step_builder<WVLT, BC, step + 1, n_step>::forward_adjoint(s, d, ns, nd);
				std::tuple_element_t<step, typename WVLT::steps>::template forward_adjoint<BC>(s, d, ns, nd);
			}

			template<typename V>
			static inline void inverse_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {

				std::tuple_element_t<step, typename WVLT::steps>::template inverse_adjoint<BC>(s, d, ns, nd);
				step_builder<WVLT, BC, step + 1, n_step>::inverse_adjoint(s, d, ns, nd);
			}
		};

		template<typename WVLT, typename BC, size_t step>
		struct step_builder<WVLT, BC, step, step> {

			template<typename V>
			static inline void forward(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {}
			
			template<typename V>
			static inline void inverse(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {}

			template<typename V>
			static inline void forward_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {}

			template<typename V>
			static inline void inverse_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {}
		};

		template<typename T> class Haar {
			constexpr static T sc = 1.41421356237309504880168872420969807856967187537694807317668;
		public:
			using type = T;

			using steps = std::tuple<
				update_d<0, -1>,
				update_s<0, T(0.5)>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
		};

		template<typename T> class Daubechies2 {
		private:
			constexpr static T s1_p0 = -1.73205080756887729352744634150587236694280525381038062805581;
			constexpr static T s2_p0 =  0.433012701892219323381861585376468091735701313452595157013952;
			constexpr static T s2_p1 = -0.0669872981077806766181384146235319082642986865474048429860483;
			constexpr static T sc =  1.93185165257813657349948639945779473526780967801680910080469;
		public:
			using steps = std::tuple<
				update_d<0, s1_p0>,
				update_s<0, s2_p0, s2_p1>,
				update_d<-1, 1>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;

			using type = T;
		};

		template<typename T> class Daubechies3 {
		private:
			constexpr static T s1_p0  = 2.42549724391195830853534735792805013148173539788185404597169;

			constexpr static T s2_m0 = -0.352387657674855469319634307041734500360168717199626110489802;
			constexpr static T s2_m1 = 0.0793394561851575665709854468556681022464175974577020488100928;

			constexpr static T s3_p1 = -2.89534745414509893429485976731702946529529368748135552662476;
			constexpr static T s3_p2 = 0.561414909153505374525726237570322812332695442156902611352901;

			constexpr static T s4_m2 = -0.019750529242293006004979050052766598262001873036524279141228;

			constexpr static T sc = 0.431879991517282793698835833676951360096586647014001193148804;

		public:
			
			using steps = std::tuple<
				update_s<0, s1_p0>,
				update_d<-1, s2_m1, s2_m0>,
				update_s<1, s3_p1, s3_p2>,
				update_d<-2, s4_m2>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;

			using type = T;
		};

		template<typename T> class Daubechies4 {
		private:
			constexpr static T s1_p0 = -3.10293148583032589470063811758833883204289885795495381963662;
			constexpr static T s2_p0 = 0.353449187617773130690791641916614771198571210734232133909753;
			constexpr static T s2_p1 = 0.116026410344205376526201432171415851440295089908831794682144;
			constexpr static T s3_p0 = -1.13273740414015604279309800164179185383471162488546344575547;
			constexpr static T s3_p1 = 0.181090277146292103917336470393403415884795293238801624079639;
			constexpr static T s4_p0 = -0.0661005526900487668472230596604034876925334941709204294099145;
			constexpr static T s4_p1 = -0.221414210158528887858396206711438754339106682715345633623713;
			constexpr static T s5_m3 = 0.23869459776013294525643502372428025694730153962927746023463;
			constexpr static T s5_m2 = -1.34832352097184759074938562094799080953526636376300686976116;
			constexpr static T s5_m1 = 4.5164219554111574424668864244565045399107667587479110837686;

			constexpr static T sc = 2.27793811152208700221465706595215439338715747993503607887708;

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
		};

		template<typename T> class Daubechies5 {
		private:
			constexpr static T s1_p0 = 3.77151921168926894757552366178790881698024090635846510016028;
			constexpr static T s2_m1 = 0.0698842923341383375213513979361008347245073868657407740184605;
			constexpr static T s2_p0 = -0.247729291360329682294999795251047782875275189599695961483240;
			constexpr static T s3_p1 = -7.59757973540575405854153663362110608451687695913930970651012;
			constexpr static T s3_p2 = 3.03368979135389213275426841861823490779263829963269704719700;
			constexpr static T s4_m3 = 0.0157993238122452355950658463497927543465183810596906799882017;
			constexpr static T s4_m2 = -0.0503963526115147652552264735912933014908335872176681805787989;
			constexpr static T s5_p3 = -1.10314632826313117312599754225554155377077047715385824307618;
			constexpr static T s5_p4 = 0.172572555625945876870245572286755166281402778242781642128057;
			constexpr static T s6_m4 = -0.002514343828207810116980039456958207159821965712982241017850;

			constexpr static T sc = 0.347389040193226710460416754871479443615028875117970218575475;

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
		};

		template<typename T> class Daubechies6 {
		private:
			constexpr static T s1_p0 = -4.43446829869067456522902386901421595117929955784397002766726;

			constexpr static T s2_p0 = 0.214593450003008209156010000739498614149390366235009628657503;
			constexpr static T s2_p1 = -0.0633131925460940277584543253045170400260191584783746185619816;

			constexpr static T s3_m1 = 9.97001561279871984191261047001888284698487613742269103059835;
			constexpr static T s3_m2 = -4.49311316887101738480447295831022985715670304865394507715493;

			constexpr static T s4_p2 = 0.179649626662970342602164106271580972989375373440522637339242;
			constexpr static T s4_p3 = 0.400695200906159800144370836431846103438795499715994915842164;

			constexpr static T s5_m1 = 0.00550295326884471036285017850066003823120949674410943525332981;
			constexpr static T s5_m2 = -0.0430453536805558366602898703076915479625247841579933421946333;

			constexpr static T s6_p2 = -0.122882256743032948063403419686360741867410363679212929915483;
			constexpr static T s6_p3 = -0.428776772090593906072415971370850703089199760064884803407522;

			constexpr static T s7_m3 = 2.3322158873585518129700437453502313783105215392342007896945;
			constexpr static T s7_m4 = -0.6683849734985699469411475049338822868631946666405609745744;
			constexpr static T s7_m5 = 0.0922130623951882057243173036464618224518216084115197289578;

			constexpr static T sc = 3.08990011938602961884790035600227126232046198472673752023023;

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
		};

		template<typename T> class Daubechies7 {
		private:
			constexpr static T s1_p0 = 5.09349848430361493158448369309319682065288729718818488170086L;

			constexpr static T s2_m0 = -0.189042092071992120719115069286184373462757526526706607687359;
			constexpr static T s2_m1 = 0.0573987259809472316318163158166229520747344869480313379225195;

			constexpr static T s3_p1 = -12.2854449967343086359903552773950797906521905606178682213636;
			constexpr static T s3_p2 = 5.95920876247802982415703053009457554606483451817620285866937;

			constexpr static T s4_m2 = -0.0604278631317421373037302780741631438456816254617632977400450;
			constexpr static T s4_m3 = 0.0291354832120604111715672746297275711217338322811099054777142;

			constexpr static T s5_p3 = -3.97071066749950430300434793946545058865144871092579097855408;
			constexpr static T s5_p4 = 1.56044025996325478842482192099426071408509934413107380064471;

			constexpr static T s6_m4 = -0.0126913773899277263576544929765396597865052704453285425832652;
			constexpr static T s6_m5 = 0.00330657330114293172083753191386779325367868962178489518206046;

			constexpr static T s7_p5 = -0.414198450444716993956500507305024397477047469411911796987340;
			constexpr static T s7_p6 = 0.0508158836433382836486473921674908683852045493076552967984957;

			constexpr static T s8_m6 = -0.000406214488767545513490343915060150692174820527241063790923;

			constexpr static T sc = 0.299010707585297416977548850152193565671577389737970684943879;

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
		};

		// Need to forward declare these so they can be used as the reverse of BiorSplines
		template<typename T> class BiorSpline3_1 {
			constexpr static T s1_m1 = -0.333333333333333333333333333333333333333333333333333333333333;
			constexpr static T s2_p0 = -1.125;
			constexpr static T s2_p1 = -0.375;
			constexpr static T s3_p0 = 0.444444444444444444444444444444444444444444444444444444444444;
			constexpr static T sc = 2.12132034355964257320253308631454711785450781306542210976502;
		public:
			using type = T;

			using steps = std::tuple<
				update_s<-1, s1_m1>,
				update_d<0, s2_p0, s2_p1>,
				update_s<0, s3_p0>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
		};

		template<typename T> class ReverseBiorSpline3_1 {
			constexpr static T s1_p1 = 0.333333333333333333333333333333333333333333333333333333333333;
			constexpr static T s2_p0 = 1.125;
			constexpr static T s2_m1 = 0.375;
			constexpr static T s3_p0 = -0.444444444444444444444444444444444444444444444444444444444444;
			constexpr static T sc = 0.471404520791031682933896241403232692856557291792316024392227;
		public:
			using type = T;

			using steps = std::tuple<
				update_d<1, s1_p1>,
				update_s<-1, s2_m1, s2_p0>,
				update_d<0, s3_p0>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
		};

		template<typename T> class BiorSpline4_2 {
			constexpr static T s1 = -0.25;
			constexpr static T s3 = 0.1875;
			constexpr static T sc = 2.82842712474619009760337744841939615713934375075389614635336;
		public:
			using type = T;

			using steps = std::tuple<
				update_s<0, s1, s1>,
				update_d<-1, -1, -1>,
				update_s<0, s3, s3>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
		};

		template<typename T> class ReverseBiorSpline4_2 {
			constexpr static T s1 = 0.25;
			constexpr static T s3 = -0.1875;
			constexpr static T sc = 0.353553390593273762200422181052424519642417968844237018294170;
		public:
			using type = T;

			using steps = std::tuple<
				update_d<0, s1, s1>,
				update_s<-1, 1, 1>,
				update_d<0, s3, s3>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
		};

		template<typename T> class BiorSpline2_4 {
			constexpr static T s1 = -0.5;
			constexpr static T s2_0 = 0.296875;
			constexpr static T s2_1 = -0.046875;
			constexpr static T sc = 1.41421356237309504880168872420969807856967187537694807317668;
		public:
			using type = T;

			using steps = std::tuple<
				update_d<0, s1, s1>,
				update_s<-2, s2_1, s2_0, s2_0, s2_1>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
		};

		template<typename T> class ReverseBiorSpline2_4 {
			constexpr static T s1 = 0.5;
			constexpr static T s2_0 = -0.296875;
			constexpr static T s2_1 = 0.046875;
			constexpr static T sc = 0.707106781186547524400844362104849039284835937688474036588340;
		public:
			using type = T;

			using steps = std::tuple<
				update_s<0, s1, s1>,
				update_d<-1, s2_1, s2_0, s2_0, s2_1>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;
		};

		template<typename T> class BiorSpline6_2 {
			constexpr static T s1 = -0.166666666666666666666666666666666666666666666666666666666667;
			constexpr static T s2 = -0.5625;
			constexpr static T s3 = -1.33333333333333333333333333333333333333333333333333333333333;
			constexpr static T s4 = 0.15625;
			constexpr static T sc = 5.65685424949238019520675489683879231427868750150779229270672;
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
		};

		template<typename T> class ReverseBiorSpline6_2 {
			constexpr static T s1 = 0.166666666666666666666666666666666666666666666666666666666667;
			constexpr static T s2 = 0.5625;
			constexpr static T s3 = 1.33333333333333333333333333333333333333333333333333333333333;
			constexpr static T s4 = -0.15625;
			constexpr static T sc = 0.176776695296636881100211090526212259821208984422118509147085;
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
		};

		template<typename T> class CDF9_7{
			constexpr static T s1 = -1.58613434205992355842831545133740131985598525529112656778527;
			constexpr static T s2 = -0.0529801185729614146241295675034771089920773921055534971880099;
			constexpr static T s3 = 0.882911075530933295919790099002837930341944716149740640164429;
			constexpr static T s4 = 0.443506852043971152115604215168913719478036852964167569567164;
			constexpr static T sc = 1.14960439886024115979507564219148965843436742907448991688182;
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
		};

		template<typename T> class ReverseCDF9_7 {
			constexpr static T s1 = 1.58613434205992355842831545133740131985598525529112656778527;
			constexpr static T s2 = 0.0529801185729614146241295675034771089920773921055534971880099;
			constexpr static T s3 = -0.882911075530933295919790099002837930341944716149740640164429;
			constexpr static T s4 = -0.443506852043971152115604215168913719478036852964167569567164;
			constexpr static T sc = 0.86986445162478127129589381456403049869640519768690701998711;
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
		};

		template<typename WVLT, typename BC=ZeroBoundary> class LiftingTransform {

		public:
			using type = typename WVLT::type;
			using boundary_condition = BC;

			template<typename V>
			static void forward(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {
				step_builder<WVLT, BC, 0, WVLT::n_steps>::forward(s, d, ns, nd);
			}

			template<typename V>
			static void inverse(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {
				step_builder<WVLT, BC, 0, WVLT::n_steps>::inverse(s, d, ns, nd);
			}

			template<typename V>
			static void forward_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {
				step_builder<WVLT, BC, 0, WVLT::n_steps>::forward_adjoint(s, d, ns, nd);
			}

			template<typename V>
			static void inverse_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t ns, const size_t nd) {
				step_builder<WVLT, BC, 0, WVLT::n_steps>::inverse_adjoint(s, d, ns, nd);
			}

			template<typename V>
			static void forward(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t n) {
				forward(s, d, n, n);
			}

			template<typename V>
			static void inverse(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t n) {
				inverse(s, d, n, n);
			}

			template<typename C1, typename C2>
			static void forward(C1& WAVELETS_RESTRICT s, C2& WAVELETS_RESTRICT d) {
				forward(s.data(), d.data(), s.size(), d.size());
			}

			template<typename C1, typename C2>
			static void inverse(C1& WAVELETS_RESTRICT s, C2& WAVELETS_RESTRICT d) {
				inverse(s.data(), d.data(), s.size(), d.size());
			}

			template<typename V>
			static void forward_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t n) {
				forward_adjoint(s, d, n, n);
			}

			template<typename V>
			static void inverse_adjoint(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t n) {
				inverse_adjoint(s, d, n, n);
			}

			template<typename C1, typename C2>
			static void forward_adjoint(C1& WAVELETS_RESTRICT s, C2& WAVELETS_RESTRICT d) {
				forward_adjoint(s.data(), d.data(), s.size(), d.size());
			}

			template<typename C1, typename C2>
			static void inverse_adjoint(C1& WAVELETS_RESTRICT s, C2& WAVELETS_RESTRICT d) {
				inverse_adjoint(s.data(), d.data(), s.size(), d.size());
			}

		};
	}

	// Hard Coded Wavelets
	using detail::Haar;
	template<typename T>
	using Daubechies1 = Haar<T>;
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
	using detail::CDF9_7;
	using detail::ReverseCDF9_7;

	template<typename T>
	using AllWavelets = std::tuple<
		Haar<T>,
		Daubechies1<T>,
		Daubechies2<T>,
		Daubechies3<T>,
		Daubechies4<T>,
		Daubechies5<T>,
		Daubechies6<T>,
		Daubechies7<T>,
		BiorSpline3_1<T>,
		BiorSpline4_2<T>,
		BiorSpline2_4<T>,
		BiorSpline6_2<T>,
		ReverseBiorSpline3_1<T>,
		ReverseBiorSpline4_2<T>,
		ReverseBiorSpline2_4<T>,
		ReverseBiorSpline6_2<T>,
		CDF9_7<T>,
		ReverseCDF9_7<T>
	>;


	// Boundary Conditions
	using detail::ZeroBoundary;
	using detail::ConstantBoundary;
	using detail::PeriodicBoundary;
	using detail::SymmetricBoundary;
	using detail::ReflectBoundary;

	using BoundaryConditions = std::tuple<
		ZeroBoundary,
		ConstantBoundary,
		PeriodicBoundary,
		SymmetricBoundary,
		ReflectBoundary
	>;

	// The Main transform function
	using detail::LiftingTransform;
}

#endif