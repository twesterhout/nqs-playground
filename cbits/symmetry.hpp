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

#include <array>
#include <complex>
#include <tuple>
#include <type_traits>

TCM_NAMESPACE_BEGIN

// Permutations {{{
namespace detail {

/// Rounds an integer up to the next power of 2. If \p is already a power of 2,
/// then \p x itself is returned.
///
/// https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
constexpr auto round_up_pow_2(unsigned x) noexcept -> unsigned
{
    x--;
    x |= x >> 1U;
    x |= x >> 2U;
    x |= x >> 4U;
    x |= x >> 8U;
    x |= x >> 16U;
    x++;
    return x;
}

/// Returns base-2 logarithm of \p x rounded down to the nearest integer.
///
/// https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
constexpr auto log2(unsigned x) noexcept -> unsigned
{
    // clang-format off
    unsigned r      = (x > 0xFFFF) << 4U; x >>= r;
    unsigned shift  = (x > 0xFF  ) << 3U; x >>= shift; r |= shift;
             shift  = (x > 0xF   ) << 2U; x >>= shift; r |= shift;
             shift  = (x > 0x3   ) << 1U; x >>= shift; r |= shift;
                                                       r |= (x >> 1U);
    // clang-format on
    return r;
}

/// Performs one step of the Butterfly network. It exchanges bits with distance
/// \p d between them if the corresponding bits in the mask \p m are set.
template <class UInt, class = std::enable_if_t<std::is_integral<UInt>::value
                                               && !std::is_signed<UInt>::value>>
constexpr auto bit_permute_step(UInt const x, UInt const m, unsigned d) noexcept
    -> UInt
{
    auto const y = (x ^ (x >> d)) & m;
    return x ^ y ^ (y << d);
}

/// Forward propagates `x` through a butterfly network specified by `masks`.
template <class UInt, size_t N>
constexpr auto bfly(UInt x, std::array<UInt, N> const& masks) noexcept -> UInt
{
    static_assert(N <= log2(std::numeric_limits<UInt>::digits),
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    for (auto i = 0U; i < N; ++i) {
        x = bit_permute_step(x, masks[i], 1U << i);
    }
    return x;
}

/// Backward propagates `x` through a butterfly network specified by `masks`.
template <class UInt, size_t N>
constexpr auto ibfly(UInt x, std::array<UInt, N> const& masks) noexcept -> UInt
{
    static_assert(N <= log2(std::numeric_limits<UInt>::digits),
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    for (auto i = static_cast<unsigned>(N);
         i-- > 0U;) { // Iterates backwards from N - 1 to 0
        x = bit_permute_step(x, masks[i], 1U << i);
    }
    return x;
}

template <class UInt> struct TCM_IMPORT BenesNetwork {
    static_assert(std::is_integral<UInt>::value && !std::is_signed<UInt>::value,
                  "UInt must be an unsigned integral type.");

    // NOTE: DO NOT CHANGE THE ORDER OF MASKS!
    //
    // This particular order is used in Symmetry's constructor.
    using MasksT = std::array<UInt, log2(std::numeric_limits<UInt>::digits)>;
    MasksT fwd;
    MasksT bwd;

    constexpr auto operator()(UInt const x) const noexcept -> UInt
    {
        return ibfly(bfly(x, fwd), bwd);
    }
};

} // namespace detail
// }}}

// Symmetry {{{
struct TCM_IMPORT Symmetry {
  public:
    using UInt = uint64_t;

  private:
    detail::BenesNetwork<UInt> _permute;
    unsigned                   _sector;
    unsigned                   _periodicity;

  public:
    Symmetry(detail::BenesNetwork<UInt> permute, unsigned sector,
             unsigned periodicity);

    Symmetry(Symmetry const&) noexcept = default;
    Symmetry(Symmetry&&) noexcept      = default;
    auto operator=(Symmetry const&) noexcept -> Symmetry& = default;
    auto operator=(Symmetry&&) noexcept -> Symmetry& = default;

    constexpr auto operator()(UInt const x) const noexcept -> UInt
    {
        return _permute(x);
    }
    constexpr auto sector() const noexcept { return _sector; }
    constexpr auto periodicity() const noexcept { return _periodicity; }
    constexpr auto phase() const noexcept -> double
    {
        return static_cast<double>(_sector) / static_cast<double>(_periodicity);
    }
    auto eigenvalue() const noexcept -> std::complex<double>
    {
        auto const arg = -2.0 * M_PI * phase();
        return std::complex<double>{std::cos(arg), std::sin(arg)};
    }

    using PickleStateT = std::tuple<typename detail::BenesNetwork<UInt>::MasksT,
                                    typename detail::BenesNetwork<UInt>::MasksT,
                                    unsigned, unsigned>;

    // Used to implement pickle support.
    constexpr auto _state_as_tuple() const noexcept -> PickleStateT
    {
        return {_permute.fwd, _permute.bwd, _sector, _periodicity};
    }
}; // }}}

TCM_NAMESPACE_END
