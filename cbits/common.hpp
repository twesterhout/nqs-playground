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

#include "config.hpp"
#include "errors.hpp"
#include <boost/align/aligned_allocator.hpp>
#include <boost/align/is_aligned.hpp>
#include <gsl/gsl-lite.hpp>
#include <SG14/inplace_function.h>
#include <torch/types.h>

#if defined(TCM_DEBUG)
#include <c10/core/CPUAllocator.h>
#endif

#include <immintrin.h>
#include <algorithm>
#include <vector>

TCM_NAMESPACE_BEGIN

using torch::optional;
using torch::nullopt;

// -------------------------------- [SIMD] --------------------------------- {{{
namespace detail {
/// Horizontally adds elements of a float4 vector.
///
/// Solution taken from https://stackoverflow.com/a/35270026
TCM_FORCEINLINE auto hadd(__m128 const v) noexcept -> float
{
    __m128 shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

/// Horizontally adds elements of a float8 vector relying only on AVX.
TCM_FORCEINLINE auto hadd(__m256 const v) noexcept -> float
{
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    vlow         = _mm_add_ps(vlow, vhigh);     // add the low 128
    return hadd(vlow);
}
} // namespace detail
// -------------------------------- [SIMD] --------------------------------- }}}

// ------------------------- [detail::make_tensor] ------------------------- {{{
namespace detail {
template <class T> struct ToScalarType {
    static_assert(!std::is_same<T, T>::value, "Type not (yet) supported.");
};

template <> struct ToScalarType<float> {
    static constexpr auto scalar_type() noexcept -> torch::ScalarType
    {
        return torch::kFloat32;
    }
};

template <> struct ToScalarType<int64_t> {
    static constexpr auto scalar_type() noexcept -> torch::ScalarType
    {
        return torch::kInt64;
    }
};

/// Returns an empty one-dimensional tensor of `float` of length `n`.
template <class T, class... Ints>
auto make_tensor(Ints... dims) -> torch::Tensor
{
    static_assert(c10::gAlignment == 64U,
                  "It is assumed that PyTorch tensors are aligned to 64 bytes");
    // TODO(twesterhout): This could overflow if one of `dims` is of type
    // `uint64_t` and is huge.
    auto out = torch::empty({static_cast<int64_t>(dims)...},
                            torch::TensorOptions()
                                .dtype(ToScalarType<T>::scalar_type())
                                .requires_grad(false));
    TCM_ASSERT(out.is_contiguous(), "it is assumed that tensors allocated "
                                    "using `torch::empty` are contiguous");
    TCM_ASSERT(boost::alignment::is_aligned(64U, out.template data<T>()),
               "it is assumed that tensors allocated using `torch::empty` are "
               "aligned to 64-byte boundary");
    return out;
}
} // namespace detail
// ------------------------- [detail::make_tensor] ------------------------- }}}

#if 0
//------------------------------- [compress] ------------------------------- {{{
/// This is very similar to std::unique from libc++ except for the else
/// branch which combines equal values.
template <class ForwardIterator, class EqualFn, class MergeFn>
auto compress(ForwardIterator first, ForwardIterator last, EqualFn equal,
              MergeFn merge) -> ForwardIterator
{
    first =
        std::adjacent_find<ForwardIterator,
                           typename std::add_lvalue_reference<EqualFn>::type>(
            first, last, equal);
    if (first != last) {
        auto i = first;
        merge(*first, std::move(*(++i)));
        for (; ++i != last;) {
            if (!equal(*first, *i)) { *(++first) = std::move(*i); }
            else {
                merge(*first, std::move(*i));
            }
        }
        ++first;
    }
    return first;
}
//------------------------------- [compress] ------------------------------- }}}
#endif

template <class T>
using aligned_vector =
    std::vector<T, boost::alignment::aligned_allocator<T, std::max<size_t>(
                                                              64, alignof(T))>>;

class SpinVector;

using RawForwardT =
    stdext::inplace_function<auto(torch::Tensor const&)->torch::Tensor,
                             /*capacity=*/32, /*alignment=*/8>;
static_assert(sizeof(RawForwardT) == 40, TCM_STATIC_ASSERT_BUG_MESSAGE);

using ForwardT =
    stdext::inplace_function<auto(gsl::span<SpinVector const>)->torch::Tensor,
                             /*capacity=*/32, /*alignment=*/8>;
static_assert(sizeof(ForwardT) == 40, TCM_STATIC_ASSERT_BUG_MESSAGE);

TCM_NAMESPACE_END
