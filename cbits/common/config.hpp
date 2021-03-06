// Copyright (c) 2019, Tom Westerhout
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

// #include <boost/config.hpp>
// #include <boost/current_function.hpp>
#include <complex>
#include <cstddef>
#include <cstdint>

#if !defined(TCM_DEBUG) && !defined(NDEBUG)
#    define TCM_DEBUG 1
#endif

#if !defined(TCM_FORCEINLINE)
#    if defined(_MSC_VER)
#        define TCM_FORCEINLINE __forceinline
#    elif defined(__GNUC__) && __GNUC__ > 3 // Clang also defines __GNUC__ (as 4)
#        define TCM_FORCEINLINE inline __attribute__((__always_inline__))
#    else
#        define TCM_FORCEINLINE inline
#    endif
#endif

#if !defined(TCM_NOINLINE)
#    if defined(_MSC_VER)
#        define TCM_NOINLINE __declspec(noinline)
#    elif defined(__GNUC__) && __GNUC__ > 3 // Clang also defines __GNUC__ (as 4)
#        if defined(__CUDACC__)             // nvcc doesn't always parse __noinline__,
#            define TCM_NOINLINE __attribute__((noinline))
#        else
#            define TCM_NOINLINE __attribute__((__noinline__))
#        endif
#    else
#        define TCM_NOINLINE
#    endif
#endif

#if !defined(TCM_NORETURN)
#    if defined(_MSC_VER)
#        define TCM_NORETURN __declspec(noreturn)
#    elif defined(__GNUC__)
#        define TCM_NORETURN __attribute__((__noreturn__))
#    elif defined(__has_cpp_attribute)
#        if __has_cpp_attribute(noreturn)
#            define TCM_NORETURN [[noreturn]]
#        endif
#    else
#        define TCM_NORETURN
#    endif
#endif

#define TCM_UNREACHABLE __builtin_unreachable()
#define TCM_LIKELY(x) __builtin_expect(x, 1)
#define TCM_UNLIKELY(x) __builtin_expect(x, 0)
#define TCM_EXPORT __attribute__((visibility("default")))
#define TCM_FALLTHROUGH __attribute__((fallthrough))
#define TCM_RESTRICT __restrict__
#define TCM_CURRENT_FUNCTION __PRETTY_FUNCTION__

// #define TCM_NORETURN BOOST_NORETURN
// #define TCM_UNUSED BOOST_ATTRIBUTE_UNUSED

#if defined(__AVX2__)
#    define TCM_HAS_AVX2() 1
#    define TCM_HAS_AVX() 1
#elif defined(__AVX__)
#    define TCM_HAS_AVX2() 0
#    define TCM_HAS_AVX() 1
#elif defined(__SSE4_2__) && defined(__SSE4_1__)
#    define TCM_HAS_AVX2() 0
#    define TCM_HAS_AVX() 0
#else
#    error "unsupported architecture; nqs-playground requires SSE4.1 & SSE4.2"
#endif

// #if defined(BOOST_GCC)
// #    define TCM_GCC BOOST_GCC
// #endif
// #if defined(BOOST_CLANG)
// #    define TCM_CLANG BOOST_CLANG
// #endif
// #if defined(__CUDACC__)
// #    define TCM_NVCC
// #endif

// #if defined(TCM_GCC) || defined(TCM_CLANG)
// #    define TCM_HOT __attribute__((hot))
// #else
// #    define TCM_HOT
// #endif

#define TCM_NAMESPACE tcm
#define TCM_NAMESPACE_BEGIN namespace tcm {
#define TCM_NAMESPACE_END } // namespace tcm
#define TCM_BUG_MESSAGE                                                                            \
    "#####################################################################\n"                      \
    "##    Congratulations, you have found a bug in nqs-playground!     ##\n"                      \
    "##            Please, be so kind to submit it here                 ##\n"                      \
    "##     https://github.com/twesterhout/nqs-playground/issues        ##\n"                      \
    "#####################################################################"
#define TCM_STATIC_ASSERT_BUG_MESSAGE                                                              \
    "Congratulations, you have found a bug in nqs-playground! Please, be "                         \
    "so kind to submit it to https://github.com/twesterhout/nqs-playground/issues."

#if defined(TCM_DEBUG)
#    define TCM_ASSERT(cond, msg)                                                                  \
        (TCM_LIKELY(cond) ? static_cast<void>(0)                                                   \
                          : ::TCM_NAMESPACE::detail::assert_fail(#cond, __FILE__, __LINE__,        \
                                                                 TCM_CURRENT_FUNCTION, msg))
#else
#    define TCM_ASSERT(cond, msg) static_cast<void>(0)
#endif

#if defined(TCM_NVCC)
#    define TCM_HOST __host__
#    define TCM_DEVICE __device__
#    define TCM_GLOBAL __global__
#else
#    define TCM_HOST
#    define TCM_DEVICE
#    define TCM_GLOBAL
#endif

TCM_NAMESPACE_BEGIN

using std::int64_t;
using std::size_t;
using std::uint16_t;
using std::uint64_t;
using real_type    = double;
using complex_type = std::complex<real_type>;

TCM_NAMESPACE_END
