

#include <array>
#include <stdexcept>
#include <iostream>
#include <typeinfo>
#include <tuple>
#include <numeric>
#include <utility>
#include <span>

#include "lifted.hpp"
#include "test_helpers.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "test_single_levels.cpp"
#include "hwy/foreach_target.h"
#include "lifted-inl.hpp"
// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"

#include "hwy/tests/test_util-inl.h"

// This test is for validating the single level transform drivers
// for forward, inverse, forward_adjoint, and inverse_adjoints, for
// all boundary conditions, and for both an even and odd length input.

HWY_BEFORE_NAMESPACE();
namespace lifted {
namespace detail {
namespace HWY_NAMESPACE{

    template<typename T>
    static auto make_test_jump_table() {

        using FuncType = std::function<void (const BoundaryCondition, T*, T*, const size_t, const size_t)>;

        array< array< array<FuncType, 2>, LIFTED_N_TRANSFORM_TYPES>, LIFTED_TESTING_N_MOCKWAVELETS> jump_table;

        static_for<LIFTED_TESTING_N_MOCKWAVELETS>([&]<size_t IW>(){
            using WVLT = mock_wavelet_from_enum_t<MockWavelet(IW), T>;
            static_for<LIFTED_N_TRANSFORM_TYPES>([&]<size_t IT>(){
                using TF = transform_from_enum_t<Transform(IT)>;
                static_for<2>([&]<size_t IDIR>(){
                    using AX = test_vecdir_from_enum_t<TestVecDir(IDIR)>;
                    jump_table[IW][IT][IDIR] = FuncType{[&](auto&&... args) {
                            FixedTransform<WVLT>::apply(TF(), AX(),
                                std::forward<decltype(args)>(args)...
                            );
                        }
                    };
                });
            });
        });
        return jump_table;
    }

	template<typename T, typename... Args>
	static void test_transform_dispatch(
		const detail::MockWavelet wvlt, const Transform op, const detail::TestVecDir ax,
		Args&&... args
	){
		const static auto test_jump_table = make_test_jump_table<T>();
	
		const size_t iw = static_cast<size_t>(wvlt);
		const size_t it_op = static_cast<size_t>(op);
		const size_t idir = static_cast<size_t>(ax);

		test_jump_table[iw][it_op][idir](std::forward<Args>(args)...);
	}
}
}

namespace HWY_NAMESPACE {

	using prec = double;

	namespace lfd = lifted::detail::HWY_NAMESPACE;
	namespace hn = hwy::HWY_NAMESPACE;

	HWY_NOINLINE void ValidateForwardBackward(
		const detail::test_parameters& params
	){
		const detail::MockWavelet wvlt = std::get<0>(params);
		const BoundaryCondition bc = std::get<1>(params);
		const detail::TestVecDir ax = std::get<2>(params);
		const size_t len = std::get<3>(params);

		const bool is_vec = ax == detail::TestVecDir::Across;

		prec rtol = 1E-6;
		prec atol = 0;
		
		const size_t nv = (is_vec)? hn::Lanes(hn::ScalableTag<prec>()) : 1;

		const size_t N_d = len / 2;
		const size_t N_s = len - N_d;

		const size_t Nt_d = N_d * nv;
		const size_t Nt_s = N_s * nv;

		auto alligned_arr_s = hwy::AllocateAligned<prec>(Nt_s);
		auto alligned_arr_d = hwy::AllocateAligned<prec>(Nt_d);

		auto arr_s = std::span(alligned_arr_s.get(), Nt_s);
		auto arr_d = std::span(alligned_arr_d.get(), Nt_d);

		detail::fill_sin(arr_s, -11.1, 13.4);
		detail::fill_sin(arr_d, -10.2, 12.15);

		auto ref_e = std::vector<prec>(Nt_s);
		auto ref_o = std::vector<prec>(Nt_d);

		detail::fill_sin(ref_e, -11.1, 13.4);
		detail::fill_sin(ref_o, -10.2, 12.15);

		lfd::test_transform_dispatch<prec>(wvlt, Transform::Forward, ax, bc, &arr_s[0], &arr_d[0], N_s, N_d);
		lfd::test_transform_dispatch<prec>(wvlt, Transform::Inverse, ax, bc, &arr_s[0], &arr_d[0], N_s, N_d);
		
		for (size_t i = 0; i < Nt_s; ++i)
			EXPECT_NEAR(ref_e[i], arr_s[i], atol + rtol * std::abs(ref_e[i]));
		
		for (size_t i = 0; i < Nt_d; ++i)
			EXPECT_NEAR(ref_o[i], arr_d[i], atol + rtol * std::abs(ref_o[i]));
	}
	
	HWY_NOINLINE void ValidateAdjointForwardBackward(
		const detail::test_parameters& params
	){
		const detail::MockWavelet wvlt = std::get<0>(params);
		const BoundaryCondition bc = std::get<1>(params);
		const detail::TestVecDir ax = std::get<2>(params);
		const size_t len = std::get<3>(params);

		const bool is_vec = ax == detail::TestVecDir::Across;

		prec rtol = 1E-6;
		prec atol = 0;
		
		const size_t nv = (is_vec)? hn::Lanes(hn::ScalableTag<prec>()) : 1;

		const size_t N_d = len / 2;
		const size_t N_s = len - N_d;

		const size_t Nt_d = N_d * nv;
		const size_t Nt_s = N_s * nv;

		auto alligned_arr_s = hwy::AllocateAligned<prec>(Nt_s);
		auto alligned_arr_d = hwy::AllocateAligned<prec>(Nt_d);

		auto arr_s = std::span(alligned_arr_s.get(), Nt_s);
		auto arr_d = std::span(alligned_arr_d.get(), Nt_d);

		detail::fill_sin(arr_s, -11.1, 13.4);
		detail::fill_sin(arr_d, -10.2, 12.15);

		auto ref_e = std::vector<prec>(Nt_s);
		auto ref_o = std::vector<prec>(Nt_d);

		detail::fill_sin(ref_e, -11.1, 13.4);
		detail::fill_sin(ref_o, -10.2, 12.15);

		lfd::test_transform_dispatch<prec>(wvlt, Transform::ForwardAdjoint, ax, bc, &arr_s[0], &arr_d[0], N_s, N_d);
		lfd::test_transform_dispatch<prec>(wvlt, Transform::InverseAdjoint, ax, bc, &arr_s[0], &arr_d[0], N_s, N_d);
		
		for (size_t i = 0; i < Nt_s; ++i)
			EXPECT_NEAR(ref_e[i], arr_s[i], atol + rtol * std::abs(ref_e[i]));
		
		for (size_t i = 0; i < Nt_d; ++i)
			EXPECT_NEAR(ref_o[i], arr_d[i], atol + rtol * std::abs(ref_o[i]));
	}
	
	HWY_NOINLINE void ValidateForwardAdjoint(
		const detail::test_parameters& params
	){
		const detail::MockWavelet wvlt = std::get<0>(params);
		const BoundaryCondition bc = std::get<1>(params);
		const detail::TestVecDir ax = std::get<2>(params);
		const size_t len = std::get<3>(params);

		const bool is_vec = ax == detail::TestVecDir::Across;

		prec rtol = 1E-6;
		prec atol = 0;
		
		const size_t nv = (is_vec)? hn::Lanes(hn::ScalableTag<prec>()) : 1;

		const size_t N_d = len / 2;
		const size_t N_s = len - N_d;

		const size_t Nt = len * nv;
		const size_t Nt_d = N_d * nv;
		const size_t Nt_s = N_s * nv;

		auto alligned_u = hwy::AllocateAligned<prec>(Nt);
		auto u = std::span(alligned_u.get(), Nt);
		auto u_s = std::span(u.begin(), Nt_s);
		auto u_d = std::span(&u[Nt_s], Nt_d);

		auto alligned_v = hwy::AllocateAligned<prec>(Nt);
		auto v = std::span(alligned_v.get(), Nt);
		auto v_s = std::span(v.begin(), Nt_s);
		auto v_d = std::span(&v[Nt_s], Nt_d);

		auto alligned_ref_u = hwy::AllocateAligned<prec>(Nt);
		auto ref_u = std::span(alligned_ref_u.get(), Nt);
		auto alligned_v_out = hwy::AllocateAligned<prec>(Nt);
		auto v_out = std::span(alligned_v_out.get(), Nt);

		detail::fill_sin(v, -255.0, 442.0);
		detail::fill_sin(ref_u, -200.0, 200.0);

		if(is_vec){
			lfd::deinterleave(detail::Across(), &ref_u[0], &u[0], len);
		}else{
			lfd::deinterleave(detail::Along(), &ref_u[0], &u[0], len);
		}

		lfd::test_transform_dispatch<prec>(wvlt, Transform::Forward, ax, bc, &u_s[0], &u_d[0], N_s, N_d);
		prec v_Fu = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);

		lfd::test_transform_dispatch<prec>(wvlt, Transform::ForwardAdjoint, ax, bc, &v_s[0], &v_d[0], N_s, N_d);

		if(is_vec){
			lfd::interleave(detail::Across(), &v[0], &v_out[0], len);
		}else{
			lfd::interleave(detail::Along(), &v[0], &v_out[0], len);
		}
		prec vFT_u = std::inner_product(v_out.begin(), v_out.end(), ref_u.begin(), 0.0);

		EXPECT_NEAR(v_Fu, vFT_u, atol + rtol * std::abs(vFT_u));
	}

	HWY_NOINLINE void ValidateInverseAdjoint(
		const detail::test_parameters& params
	){
		const detail::MockWavelet wvlt = std::get<0>(params);
		const BoundaryCondition bc = std::get<1>(params);
		const detail::TestVecDir ax = std::get<2>(params);
		const size_t len = std::get<3>(params);

		const bool is_vec = ax == detail::TestVecDir::Across;

		prec rtol = 1E-6;
		prec atol = 0;
		
		const size_t nv = (is_vec)? hn::Lanes(hn::ScalableTag<prec>()) : 1;

		const size_t N_d = len / 2;
		const size_t N_s = len - N_d;

		const size_t Nt = len * nv;
		const size_t Nt_d = N_d * nv;
		const size_t Nt_s = N_s * nv;

		auto alligned_u = hwy::AllocateAligned<prec>(Nt);
		auto u = std::span(alligned_u.get(), Nt);
		auto u_s = std::span(u.begin(), Nt_s);
		auto u_d = std::span(&u[Nt_s], Nt_d);

		auto alligned_v = hwy::AllocateAligned<prec>(Nt);
		auto v = std::span(alligned_v.get(), Nt);
		auto v_s = std::span(v.begin(), Nt_s);
		auto v_d = std::span(&v[Nt_s], Nt_d);

		auto alligned_ref_u = hwy::AllocateAligned<prec>(Nt);
		auto ref_u = std::span(alligned_ref_u.get(), Nt);
		auto alligned_v_out = hwy::AllocateAligned<prec>(Nt);
		auto v_out = std::span(alligned_v_out.get(), Nt);

		detail::fill_sin(v, -255.0, 442.0);
		detail::fill_sin(ref_u, -200.0, 200.0);

		if(is_vec){
			lfd::deinterleave(detail::Across(), &ref_u[0], &u[0], len);
		}else{
			lfd::deinterleave(detail::Along(), &ref_u[0], &u[0], len);
		}

		lfd::test_transform_dispatch<prec>(wvlt, Transform::Inverse, ax, bc, &u_s[0], &u_d[0], N_s, N_d);
		prec v_Fu = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);

		lfd::test_transform_dispatch<prec>(wvlt, Transform::InverseAdjoint, ax, bc, &v_s[0], &v_d[0], N_s, N_d);

		if(is_vec){
			lfd::interleave(detail::Across(), &v[0], &v_out[0], len);
		}else{
			lfd::interleave(detail::Along(), &v[0], &v_out[0], len);
		}
		prec vFT_u = std::inner_product(v_out.begin(), v_out.end(), ref_u.begin(), 0.0);

		EXPECT_NEAR(v_Fu, vFT_u, atol + rtol * std::abs(vFT_u));
	}

	HWY_NOINLINE void ValidateForwardDotInverseAdjoint(
		const detail::test_parameters& params
	){
		const detail::MockWavelet wvlt = std::get<0>(params);
		const BoundaryCondition bc = std::get<1>(params);
		const detail::TestVecDir ax = std::get<2>(params);
		const size_t len = std::get<3>(params);

		const bool is_vec = ax == detail::TestVecDir::Across;

		prec rtol = 1E-6;
		prec atol = 0;
		
		const size_t nv = (is_vec)? hn::Lanes(hn::ScalableTag<prec>()) : 1;

		const size_t N_d = len / 2;
		const size_t N_s = len - N_d;

		const size_t Nt = len * nv;
		const size_t Nt_d = N_d * nv;
		const size_t Nt_s = N_s * nv;

		auto alligned_u = hwy::AllocateAligned<prec>(Nt);
		auto u = std::span(alligned_u.get(), Nt);
		auto u_s = std::span(u.begin(), Nt_s);
		auto u_d = std::span(&u[Nt_s], Nt_d);

		auto alligned_v = hwy::AllocateAligned<prec>(Nt);
		auto v = std::span(alligned_v.get(), Nt);
		auto v_s = std::span(v.begin(), Nt_s);
		auto v_d = std::span(&v[Nt_s], Nt_d);

		detail::fill_sin(v, -255.0, 442.0);
		detail::fill_sin(u, -200.0, 200.0);

		prec v_dot_u = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);

		lfd::test_transform_dispatch<prec>(wvlt, Transform::Forward, ax, bc, &u_s[0], &u_d[0], N_s, N_d);
		lfd::test_transform_dispatch<prec>(wvlt, Transform::InverseAdjoint, ax, bc, &v_s[0], &v_d[0], N_s, N_d);

		prec vw_dot_uw = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);

		EXPECT_NEAR(v_dot_u, vw_dot_uw, atol + rtol * std::abs(v_dot_u));
	}

	HWY_NOINLINE void ValidateInverseDotForwardAdjoint(
		const detail::test_parameters& params
	){
		const detail::MockWavelet wvlt = std::get<0>(params);
		const BoundaryCondition bc = std::get<1>(params);
		const detail::TestVecDir ax = std::get<2>(params);
		const size_t len = std::get<3>(params);

		const bool is_vec = ax == detail::TestVecDir::Across;

		prec rtol = 1E-6;
		prec atol = 0;
		
		const size_t nv = (is_vec)? hn::Lanes(hn::ScalableTag<prec>()) : 1;

		const size_t N_d = len / 2;
		const size_t N_s = len - N_d;

		const size_t Nt = len * nv;
		const size_t Nt_d = N_d * nv;
		const size_t Nt_s = N_s * nv;

		auto alligned_u = hwy::AllocateAligned<prec>(Nt);
		auto u = std::span(alligned_u.get(), Nt);
		auto u_s = std::span(u.begin(), Nt_s);
		auto u_d = std::span(&u[Nt_s], Nt_d);

		auto alligned_v = hwy::AllocateAligned<prec>(Nt);
		auto v = std::span(alligned_v.get(), Nt);
		auto v_s = std::span(v.begin(), Nt_s);
		auto v_d = std::span(&v[Nt_s], Nt_d);

		detail::fill_sin(v, -255.0, 442.0);
		detail::fill_sin(u, -200.0, 200.0);

		prec v_dot_u = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);

		lfd::test_transform_dispatch<prec>(wvlt, Transform::ForwardAdjoint, ax, bc, &u_s[0], &u_d[0], N_s, N_d);
		lfd::test_transform_dispatch<prec>(wvlt, Transform::Inverse, ax, bc, &v_s[0], &v_d[0], N_s, N_d);

		prec vw_dot_uw = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);

		EXPECT_NEAR(v_dot_u, vw_dot_uw, atol + rtol * std::abs(v_dot_u));
	}
}
}
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace lifted{

	class LiftedTestSuite : public hwy::TestWithParamTargetAndT<detail::test_parameters>{};

	HWY_TARGET_INSTANTIATE_TEST_SUITE_P_T(
		LiftedTestSuite,
		::testing::Combine(
			::testing::ValuesIn(detail::mock_wavelet_enum_array),
			::testing::ValuesIn(detail::bc_enum_array),
			::testing::Values(
				detail::TestVecDir::Along, detail::TestVecDir::Across
			),
			::testing::Values(std::size_t(153), std::size_t(510))
		)
	);

	HWY_EXPORT_AND_TEST_P_T(LiftedTestSuite, ValidateForwardBackward);
	HWY_EXPORT_AND_TEST_P_T(LiftedTestSuite, ValidateAdjointForwardBackward);
	HWY_EXPORT_AND_TEST_P_T(LiftedTestSuite, ValidateForwardAdjoint);
	HWY_EXPORT_AND_TEST_P_T(LiftedTestSuite, ValidateInverseAdjoint);
	HWY_EXPORT_AND_TEST_P_T(LiftedTestSuite, ValidateForwardDotInverseAdjoint);
	HWY_EXPORT_AND_TEST_P_T(LiftedTestSuite, ValidateInverseDotForwardAdjoint);
}

#endif