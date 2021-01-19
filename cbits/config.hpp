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

#include <boost/config.hpp>
#include <boost/current_function.hpp>
#include <complex>
#include <cstddef>
#include <cstdint>

#if !defined(TCM_DEBUG) && !defined(NDEBUG)
#    define TCM_DEBUG 1
#endif

#define TCM_FORCEINLINE BOOST_FORCEINLINE
#define TCM_NOINLINE BOOST_NOINLINE
#define TCM_LIKELY(x) BOOST_LIKELY(x)
#define TCM_UNLIKELY(x) BOOST_UNLIKELY(x)
#define TCM_NORETURN BOOST_NORETURN
#define TCM_UNUSED BOOST_ATTRIBUTE_UNUSED
#define TCM_FALLTHROUGH BOOST_FALLTHROUGH
#define TCM_CURRENT_FUNCTION BOOST_CURRENT_FUNCTION
#define TCM_EXPORT BOOST_SYMBOL_EXPORT
#define TCM_IMPORT BOOST_SYMBOL_IMPORT
#define TCM_RESTRICT __restrict__

#if defined(__AVX2__)
#    define TCM_HAS_AVX2() 1
#    define TCM_HAS_AVX() 1
#elif defined(__AVX__)
#    define TCM_HAS_AVX2() 0
#    define TCM_HAS_AVX() 1
#elif defined(__SSE2__) || defined(__x86_64__)
#    define TCM_HAS_AVX2() 0
#    define TCM_HAS_AVX() 0
#else
#    error "unsupported architecture; nqs_playground currently only works on x86_64"
#endif

#if defined(BOOST_GCC)
#    define TCM_GCC BOOST_GCC
#endif
#if defined(BOOST_CLANG)
#    define TCM_CLANG BOOST_CLANG
#endif
#if defined(__CUDACC__)
#    define TCM_NVCC
#endif

#if defined(TCM_GCC) || defined(TCM_CLANG)
#    define TCM_HOT __attribute__((hot))
#else
#    define TCM_HOT
#endif

#define TCM_NAMESPACE tcm
#define TCM_NAMESPACE_BEGIN namespace tcm {
#define TCM_NAMESPACE_END } // namespace tcm
#define TCM_BUG_MESSAGE                                                                            \
    "#####################################################################\n"                      \
    "##    Congratulations, you have found a bug in nqs-playground!     ##\n"                      \
    "##            Please, be so kind to submit it here                 ##\n"                      \
    "##     https://github.com/twesterhout/nqs-playground/issues        ##\n"                      \
    "#####################################################################"

#if defined(TCM_CLANG)
// Clang refuses to display newlines
#    define TCM_STATIC_ASSERT_BUG_MESSAGE                                                          \
        "Congratulations, you have found a bug in nqs-playground! Please, be "                     \
        "so kind to submit it to "                                                                 \
        "https://github.com/twesterhout/nqs-playground/issues."
#else
#    define TCM_STATIC_ASSERT_BUG_MESSAGE "\n" TCM_BUG_MESSAGE
#endif

#define TCM_NOEXCEPT noexcept
#define TCM_CONSTEXPR constexpr

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

#if 0
struct SplitTag {};
struct SerialTag {};
struct ParallelTag {};
struct UnsafeTag {};

constexpr UnsafeTag split_tag;
constexpr UnsafeTag serial_tag;
constexpr UnsafeTag parallel_tag;
constexpr UnsafeTag unsafe_tag;
#endif

TCM_NAMESPACE_END
