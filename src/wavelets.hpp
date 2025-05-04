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

namespace wavelets {
	namespace detail {

		using std::size_t;

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

		template<ptrdiff_t offset, auto val, auto... vals>
		struct lift {

			template<typename V>
			static inline V apply(const V* x, const size_t i) {
				return val * x[i + offset] + lift<offset + 1, vals...>::apply(x, i);
			}
			template<typename V>
			static inline V wrap_apply(const V* x, const ptrdiff_t i, const size_t n) {
				size_t iw = (i + offset) % n;
				return val * x[iw] + lift<offset + 1, vals...>::wrap_apply(x, i, n);
			}
		};

		template<ptrdiff_t offset, auto val>
		struct lift<offset, val> {

			template<typename V>
			static inline V apply(const V* x, const size_t i) {
				return val * x[i + offset];
			}
			template<typename V>
			static inline V wrap_apply(const V* x, const ptrdiff_t i, const size_t n) {
				size_t iw = (i + offset) % n;
				return val * x[iw];
			}
		};

		//template<ptrdiff_t offset, auto... vals>
		//struct periodic_wrap {
		//	size_t n_vals =

		//	template<typename V>
		//	static inline void front(V* x, const V* y, const size_t n) {
		//	}
		//};

		template<ptrdiff_t offset, auto... vals>
		struct update_d {
			constexpr static size_t n_vals = sizeof...(vals);
			constexpr static size_t n_front = (offset < 0) ? -offset : 0;
			constexpr static size_t n_back = (offset < 0) ? n_vals - 1 : offset + n_vals - 1;
			using lifter = lift<offset, vals...>;

			template<typename V>
			static inline void forward(const V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t n) {
				// front wrap
				for (ptrdiff_t i = 0; i < n_front; ++i)
					d[i] += lifter::wrap_apply(s, i, n);

				// main loop
				for (size_t i = n_front; i < n - n_back; ++i)
					d[i] += lifter::apply(s, i);

				// back wrap
				for (ptrdiff_t i = n - n_back; i < n; ++i)
					d[i] += lifter::wrap_apply(s, i, n);
			}

			template<typename V>
			static inline void inverse(const V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t n) {
				size_t i_end = (offset < 0) ? (n - (n_vals - 1)) : (n - (offset + n_vals - 1));

				// front wrap
				for (ptrdiff_t i = 0; i < n_front; ++i)
					d[i] -= lifter::wrap_apply(s, i, n);

				// main loop
				for (size_t i = n_front; i < n - n_back; ++i)
					d[i] -= lifter::apply(s, i);

				// back wrap
				for (ptrdiff_t i = n - n_back; i < n; ++i)
					d[i] -= lifter::wrap_apply(s, i, n);
			}
		};

		template<ptrdiff_t offset, auto... vals>
		struct update_s {
			constexpr static size_t n_vals = sizeof...(vals);
			constexpr static size_t n_front = (offset < 0) ? -offset : 0;
			constexpr static size_t n_back = (offset < 0) ? n_vals - 1 : offset + n_vals - 1;
			using lifter = lift<offset, vals...>;

			template<typename V>
			static inline void forward(V* WAVELETS_RESTRICT s, const V* WAVELETS_RESTRICT d, const size_t n) {

				// front wrap
				for (ptrdiff_t i = 0; i < n_front; ++i)
					s[i] += lifter::wrap_apply(d, i, n);

				// main loop
				for (size_t i = n_front; i < n - n_back; ++i)
					s[i] += lifter::apply(d, i);

				// back wrap
				for (ptrdiff_t i = n - n_back; i < n; ++i)
					s[i] += lifter::wrap_apply(d, i, n);
			}

			template<typename V>
			static inline void inverse(V* WAVELETS_RESTRICT s, const V* WAVELETS_RESTRICT d, const size_t n) {

				// front wrap
				for (ptrdiff_t i = 0; i < n_front; ++i)
					s[i] -= lifter::wrap_apply(d, i, n);

				// main loop
				for (size_t i = n_front; i < n - n_back; ++i)
					s[i] -= lifter::apply(d, i);

				// back wrap
				for (ptrdiff_t i = n - n_back; i < n; ++i)
					s[i] -= lifter::wrap_apply(d, i, n);
			}
		};

		template<auto val>
		struct scale {
			template<typename V>
			static inline void forward(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t n)
			{
				for (size_t i = 0; i < n; ++i) {
					d[i] /= val;
					s[i] *= val;
				}
			}

			template<typename V>
			static inline void inverse(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, const size_t n)
			{
				for (size_t i = 0; i < n; ++i) {
					d[i] *= val;
					s[i] /= val;
				}
			}
		};

		template<typename WVLT, size_t step, size_t n_step>
		struct step_builder {

			template<typename V>
			static inline void forward(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, size_t n) {

				std::tuple_element_t<step, typename WVLT::steps>::forward(s, d, n);
				step_builder<WVLT, step + 1, n_step>::forward(s, d, n);
			}

			template<typename V>
			static inline void inverse(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, size_t n) {

				step_builder<WVLT, step + 1, n_step>::inverse(s, d, n);
				std::tuple_element_t<step, typename WVLT::steps>::inverse(s, d, n);
			}
		};

		template<typename WVLT, size_t step>
		struct step_builder<WVLT, step, step> {
			template<typename V>
			static inline void forward(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, size_t n) {
			}
			template<typename V>
			static inline void inverse(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, size_t n) {
			}
		};

		template<typename T> class Haar {
			constexpr static T sc = T(1.41421356237309504880168872420969807856967187537694807317668);
		public:
			using type = T;

			using steps = std::tuple<
				update_d<0, -1>,
				update_s<0, T(0.5)>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;

			using reverse = Haar<T>;
		};

		template<typename T> class Daubechies2 {
		private:
			constexpr static T s1_p0 = T(-1.73205080756887729352744634150587236694280525381038062805581);
			constexpr static T s2_p0 = T( 0.433012701892219323381861585376468091735701313452595157013952);
			constexpr static T s2_p1 = T(-0.0669872981077806766181384146235319082642986865474048429860483);
			constexpr static T sc = T( 1.93185165257813657349948639945779473526780967801680910080469);
		public:
			using steps = std::tuple<
				update_d<0, s1_p0>,
				update_s<0, s2_p0, s2_p1>,
				update_d<-1, 1>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;

			using type = T;
			using reverse = Daubechies2<T>;
		};

		template<typename T> class Daubechies3 {
		private:
			constexpr static T s1_p0  = T(2.42549724391195830853534735792805013148173539788185404597169);
			constexpr static T s2_m1 = T(0.0793394561851575665709854468556681022464175974577020488100928);
			constexpr static T s2_p0 = T(-0.352387657674855469319634307041734500360168717199626110489802);
			constexpr static T s3_p1 = T(-2.89534745414509893429485976731702946529529368748135552662476);
			constexpr static T s3_p2 = T(0.561414909153505374525726237570322812332695442156902611352901);
			constexpr static T s4_m2 = T(-0.019750529242293006004979050052766598262001873036524279141228);

			constexpr static T sc = T(0.431879991517282793698835833676951360096586647014001193148804);

		public:
			
			using steps = std::tuple<
				update_s<0, s1_p0>,
				update_d<-1, s2_m1, s2_p0>,
				update_s<1, s3_p1, s3_p2>,
				update_d<-2, s4_m2>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;

			using type = T;
			using reverse = Daubechies3<T>;
		};

		template<typename T> class Daubechies4 {
		private:
			constexpr static T s1_p0 = T(-3.10293148583032589470063811758833883204289885795495381963662);
			constexpr static T s2_p0 = T(0.353449187617773130690791641916614771198571210734232133909753);
			constexpr static T s2_p1 = T(0.116026410344205376526201432171415851440295089908831794682144);
			constexpr static T s3_p0 = T(-1.13273740414015604279309800164179185383471162488546344575547);
			constexpr static T s3_p1 = T(0.181090277146292103917336470393403415884795293238801624079639);
			constexpr static T s4_p0 = T(-0.0661005526900487668472230596604034876925334941709204294099145);
			constexpr static T s4_p1 = T(-0.221414210158528887858396206711438754339106682715345633623713);
			constexpr static T s5_m3 = T(0.23869459776013294525643502372428025694730153962927746023463);
			constexpr static T s5_m2 = T(-1.34832352097184759074938562094799080953526636376300686976116);
			constexpr static T s5_m1 = T(4.5164219554111574424668864244565045399107667587479110837686);

			constexpr static T sc = T(2.27793811152208700221465706595215439338715747993503607887708);

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
			using reverse = Daubechies4<T>;
		};

		template<typename T> class Daubechies5 {
		private:
			constexpr static T s1_p0 = T(3.77151921168926894757552366178790881698024090635846510016028);
			constexpr static T s2_m1 = T(0.0698842923341383375213513979361008347245073868657407740184605);
			constexpr static T s2_p0 = T(-0.247729291360329682294999795251047782875275189599695961483240);
			constexpr static T s3_p1 = T(-7.59757973540575405854153663362110608451687695913930970651012);
			constexpr static T s3_p2 = T(3.03368979135389213275426841861823490779263829963269704719700);
			constexpr static T s4_m3 = T(0.0157993238122452355950658463497927543465183810596906799882017);
			constexpr static T s4_m2 = T(-0.0503963526115147652552264735912933014908335872176681805787989);
			constexpr static T s5_p3 = T(-1.10314632826313117312599754225554155377077047715385824307618);
			constexpr static T s5_p4 = T(0.172572555625945876870245572286755166281402778242781642128057);
			constexpr static T s6_m4 = T(-0.002514343828207810116980039456958207159821965712982241017850);

			constexpr static T sc = T(0.347389040193226710460416754871479443615028875117970218575475);

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
			using reverse = Daubechies5<T>;
		};

		template<typename T> class Daubechies6 {
		private:
			constexpr static T s1_p0 = T(-4.43446829869067456522902386901421595117929955784397002766726);

			constexpr static T s2_p0 = T(0.214593450003008209156010000739498614149390366235009628657503);
			constexpr static T s2_p1 = T(-0.0633131925460940277584543253045170400260191584783746185619816);

			constexpr static T s3_m1 = T(9.97001561279871984191261047001888284698487613742269103059835);
			constexpr static T s3_m2 = T(-4.49311316887101738480447295831022985715670304865394507715493);

			constexpr static T s4_p2 = T(0.179649626662970342602164106271580972989375373440522637339242);
			constexpr static T s4_p3 = T(0.400695200906159800144370836431846103438795499715994915842164);

			constexpr static T s5_m1 = T(0.00550295326884471036285017850066003823120949674410943525332981);
			constexpr static T s5_m2 = T(-0.0430453536805558366602898703076915479625247841579933421946333);

			constexpr static T s6_p2 = T(-0.122882256743032948063403419686360741867410363679212929915483);
			constexpr static T s6_p3 = T(-0.428776772090593906072415971370850703089199760064884803407522);

			constexpr static T s7_m3 = T(2.3322158873585518129700437453502313783105215392342007896945);
			constexpr static T s7_m4 = T(-0.6683849734985699469411475049338822868631946666405609745744);
			constexpr static T s7_m5 = T(0.0922130623951882057243173036464618224518216084115197289578);

			constexpr static T sc = T(3.08990011938602961884790035600227126232046198472673752023023);

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
			using reverse = Daubechies6<T>;
		};

		template<typename T> class Daubechies7 {
		private:
			constexpr static T s1_p0 = 5.09349848430361493158448369309319682065288729718818488170086;

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
				update_s< 0, s1_p0, s2_p1>,
				update_d< -1, s2_m1, s2_m0>,
				update_s< 1, s3_p1, s3_p2>,
				update_d<-3, s4_m3, s4_m2>,
				update_s< 3, s5_p3, s5_p4>,
				update_d<-5, s6_m5, s6_m4>,
				update_s< 5, s7_p5, s7_p6>,
				update_d<-8, s8_m6>,
				scale<sc>
			>;
			constexpr static size_t n_steps = std::tuple_size<steps>::value;

			using type = T;
			using reverse = Daubechies7<T>;
		};

		// Need to forward declare these so they can be used as the reverse of BiorSplines
		template<typename T> class ReverseBiorSpline3_1;
		template<typename T> class ReverseBiorSpline4_2;
		template<typename T> class ReverseBiorSpline2_4;
		template<typename T> class ReverseBiorSpline6_2;

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
			using reverse = ReverseBiorSpline3_1<T>;
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
			using reverse = BiorSpline3_1<T>;
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
			using reverse = ReverseBiorSpline4_2<T>;
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
			using reverse = BiorSpline4_2<T>;
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
			using reverse = ReverseBiorSpline2_4<T>;
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
			using reverse = BiorSpline2_4<T>;
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
			using reverse = ReverseBiorSpline6_2<T>;
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
			using reverse = BiorSpline6_2<T>;
		};

		template<typename WVLT> class LiftingTransform {

		public:
			using type = typename WVLT::type;

			template<typename V>
			static void forward(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, size_t n) {
				step_builder<WVLT, 0, WVLT::n_steps>::forward(s, d, n);
			}

			template<typename V>
			static void inverse(V* WAVELETS_RESTRICT s, V* WAVELETS_RESTRICT d, size_t n) {
				step_builder<WVLT, 0, WVLT::n_steps>::inverse(s, d, n);
			}

		};
	}
	using detail::Haar;
	using detail::Daubechies2;
	using detail::Daubechies3;
	using detail::Daubechies4;
	using detail::Daubechies5;
	using detail::Daubechies6;

	using detail::BiorSpline3_1;
	using detail::ReverseBiorSpline3_1;
	using detail::BiorSpline4_2;
	using detail::ReverseBiorSpline4_2;
	using detail::BiorSpline2_4;
	using detail::ReverseBiorSpline2_4;
	using detail::BiorSpline6_2;
	using detail::ReverseBiorSpline6_2;

	using detail::LiftingTransform;
}

#endif