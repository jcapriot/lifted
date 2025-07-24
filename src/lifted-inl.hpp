#if defined(LIFTED_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIFTED_INL_H_
#undef LIFTED_INL_H_
#else
#define LIFTED_INL_H_
#endif

#if !(__cplusplus >= 202002L || _MSVC_LANG+0L >= 202002L)
#error This file requires at least C++20 support.
#endif

#include "lifted-common.hpp"
#include "lifted-ops-inl.hpp"
#include "lifted-steps-inl.hpp"
#include "lifted-drivers-inl.hpp"

#endif