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

#include "bits512.hpp"
#include "errors.hpp"

#include <gsl/gsl-lite.hpp>

#include <array>
#include <complex>
#include <tuple>
#include <type_traits>

TCM_NAMESPACE_BEGIN

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
constexpr auto bit_permute_step(uint64_t const x, uint64_t const m,
                                unsigned const d) noexcept -> uint64_t
{
    auto const y = (x ^ (x >> d)) & m;
    return x ^ y ^ (y << d);
}

/// Forward propagates `x` through a butterfly network specified by `masks`.
constexpr auto bfly(uint64_t x, std::array<uint64_t, 6> const& masks) noexcept
    -> uint64_t
{
    for (auto i = 0U; i < log2(64); ++i) {
        x = bit_permute_step(x, masks[i], 1U << i);
    }
    return x;
}

/// Backward propagates `x` through a butterfly network specified by `masks`.
constexpr auto ibfly(uint64_t x, std::array<uint64_t, 6> const& masks) noexcept
    -> uint64_t
{
    for (auto i = log2(64U); i-- > 0U;) { // Iterates backwards from N - 1 to 0
        x = bit_permute_step(x, masks[i], 1U << i);
    }
    return x;
}

/// Implements one step of the Butterfly network.
///
/// TODO: This implementation is probably quite slow. Profile & optimize it!
template <unsigned Shift>
TCM_FORCEINLINE constexpr auto
inplace_bit_permute_step(bits512& x, bits512 const& m) noexcept -> void
{
    // auto const y = (x ^ (x >> d)) & m;
    // return x ^ y ^ (y << d);

    // This is the simplest possible implementation. We process `x` in words
    // (i.e. 64 bits). Hence, we have two different cases:
    bits512 y = {};
    if constexpr (Shift < 64U) {
        // y := (x ^ (x >> d)) & m
        {
            constexpr auto upper = [](auto const w) { return w >> Shift; };
            constexpr auto lower = [](auto const w) {
                return (w & ((1UL << Shift) - 1UL)) << (64U - Shift);
            };
            for (auto i = 0; i < 7; ++i) {
                y.words[i] =
                    (x.words[i] ^ (lower(x.words[i + 1]) | upper(x.words[i])))
                    & m.words[i];
            }
            y.words[7] = (x.words[7] ^ upper(x.words[7])) & m.words[7];
        }
        // x := x ^ y ^ (y << d)
        {
            constexpr auto upper = [](auto const w) { return w << Shift; };
            constexpr auto lower = [](auto const w) {
                return w >> (64U - Shift);
            };
            x.words[0] ^= y.words[0] ^ upper(y.words[0]);
            for (auto i = 1; i < 8; ++i) {
                x.words[i] ^=
                    y.words[i] ^ (upper(y.words[i]) | lower(y.words[i - 1]));
            }
        }
    }
    else {
        static_assert(Shift % 64U == 0);
        static_assert(Shift / 64U < 8);
        constexpr auto Delta = Shift / 64U;
        // y := (x ^ (x >> d)) & m
        {
            for (auto i = 0U; i < 8U - Delta; ++i) {
                y.words[i] = (x.words[i] ^ x.words[i + Delta]) & m.words[i];
            }
            for (auto i = 8U - Delta; i < 8U; ++i) {
                y.words[i] = (x.words[i] ^ 0UL) & m.words[i];
            }
        }
        // x := x ^ y ^ (y << d)
        {
            for (auto i = 0U; i < Delta; ++i) {
                x.words[i] ^= (y.words[i] ^ 0UL);
            }
            for (auto i = Delta; i < 8U; ++i) {
                x.words[i] ^= y.words[i] ^ y.words[i - Delta];
            }
        }
    }
}

/// Forward propagates `x` through a butterfly network specified by `masks`.
constexpr auto bfly(bits512& x, std::array<bits512, 9> const& masks) noexcept
    -> void
{
#define STEP(i) inplace_bit_permute_step<(1U << i)>(x, masks[i])
    STEP(0);
    STEP(1);
    STEP(2);
    STEP(3);
    STEP(4);
    STEP(5);
    STEP(6);
    STEP(7);
    STEP(8);
#undef STEP
}

/// Backward propagates `x` through a butterfly network specified by `masks`.
constexpr auto ibfly(bits512& x, std::array<bits512, 9> const& masks) noexcept
    -> void
{
#define STEP(i) inplace_bit_permute_step<(1U << i)>(x, masks[i])
    STEP(8);
    STEP(7);
    STEP(6);
    STEP(5);
    STEP(4);
    STEP(3);
    STEP(2);
    STEP(1);
    STEP(0);
#undef STEP
}

// SymmetryBase {{{
struct TCM_IMPORT SymmetryBase {
  private:
    unsigned             _sector;
    unsigned             _periodicity;
    std::complex<double> _eigenvalue;

  public:
    SymmetryBase(unsigned sector, unsigned periodicity);

    SymmetryBase(SymmetryBase const&) noexcept = default;
    SymmetryBase(SymmetryBase&&) noexcept      = default;
    auto operator=(SymmetryBase const&) noexcept -> SymmetryBase& = default;
    auto operator=(SymmetryBase&&) noexcept -> SymmetryBase& = default;

    constexpr auto sector() const noexcept { return _sector; }
    constexpr auto periodicity() const noexcept { return _periodicity; }
    constexpr auto phase() const noexcept -> double
    {
        return static_cast<double>(_sector) / static_cast<double>(_periodicity);
    }
    auto eigenvalue() const noexcept -> std::complex<double>
    {
        return _eigenvalue;
        // auto const arg = -2.0 * M_PI * phase();
        // return std::complex<double>{std::cos(arg), std::sin(arg)};
    }
}; // }}}

struct Symmetry8x64;

namespace v2 {

template <unsigned Bits> struct Symmetry;

template <> struct TCM_EXPORT Symmetry<64> : public SymmetryBase {
  public:
    using StateT = uint64_t;
    using _PickleStateT =
        std::tuple<unsigned, unsigned, std::array<uint64_t, 6>,
                   std::array<uint64_t, 6>>;

    // private:
    std::array<uint64_t, 6> _fwd;
    std::array<uint64_t, 6> _bwd;

  public:
    Symmetry(std::array<uint64_t, 6> const& forward,
             std::array<uint64_t, 6> const& backward, unsigned sector,
             unsigned periodicity);

    Symmetry(Symmetry const&) noexcept = default;
    Symmetry(Symmetry&&) noexcept      = default;
    auto operator=(Symmetry const&) noexcept -> Symmetry& = default;
    auto operator=(Symmetry&&) noexcept -> Symmetry& = default;

    constexpr auto operator()(uint64_t const x) const noexcept -> uint64_t
    {
        return ibfly(bfly(x, _fwd), _bwd);
    }

    auto        _internal_state() const noexcept -> _PickleStateT;
    static auto _from_internal_state(_PickleStateT const&) -> Symmetry;
};

template <> struct TCM_EXPORT Symmetry<512> : public SymmetryBase {
  public:
    using StateT = bits512;

  private:
    std::array<bits512, 9> _fwd;
    std::array<bits512, 9> _bwd;

  public:
    Symmetry(std::array<bits512, 9> const& forward,
             std::array<bits512, 9> const& backward, unsigned sector,
             unsigned periodicity);

    Symmetry(Symmetry const&) noexcept = default;
    Symmetry(Symmetry&&) noexcept      = default;
    auto operator=(Symmetry const&) noexcept -> Symmetry& = default;
    auto operator=(Symmetry&&) noexcept -> Symmetry& = default;

    constexpr auto operator()(bits512 const& x) const noexcept -> bits512
    {
        auto y = x;
        bfly(y, _fwd);
        ibfly(y, _bwd);
        return y;
    }
};

} // namespace v2

struct alignas(64) Symmetry8x64 {
    uint64_t             _fwds[6][8];
    uint64_t             _bwds[6][8];
    unsigned             _sectors[8];
    unsigned             _periodicities[8];
    std::complex<double> _eigenvalues[8];

    Symmetry8x64(gsl::span<v2::Symmetry<64> const> original);

    Symmetry8x64(Symmetry8x64 const&) noexcept = default;
    Symmetry8x64(Symmetry8x64&&) noexcept      = default;
    auto operator=(Symmetry8x64 const&) noexcept -> Symmetry8x64& = default;
    auto operator=(Symmetry8x64&&) noexcept -> Symmetry8x64& = default;

    auto operator()(uint64_t x, uint64_t out[8]) const noexcept -> void;
};

TCM_IMPORT auto full_info(gsl::span<v2::Symmetry<64> const> symmetries,
                          uint64_t                          spin)
    -> std::tuple</*representative=*/uint64_t,
                  /*eigenvalue=*/std::complex<double>, /*norm=*/double>;

TCM_IMPORT auto full_info(gsl::span<v2::Symmetry<512> const> symmetries,
                          bits512 const&                     spin)
    -> std::tuple</*representative=*/bits512,
                  /*eigenvalue=*/std::complex<double>, /*norm=*/double>;

auto representative(gsl::span<Symmetry8x64 const>     symmetries,
                    gsl::span<v2::Symmetry<64> const> other,
                    uint64_t const                    x) noexcept
    -> std::tuple<uint64_t, double, ptrdiff_t>;

template <class... UInts>
TCM_FORCEINLINE constexpr auto flipped(uint64_t const x, UInts... is) noexcept
    -> uint64_t
{
    return x ^ (... | (1UL << is));
}

template <class UInt, class = std::enable_if_t<std::is_unsigned_v<UInt>>>
TCM_FORCEINLINE constexpr auto are_not_aligned(uint64_t const spin,
                                               UInt const     i,
                                               UInt const j) noexcept -> bool
{
    return ((spin >> i) ^ (spin >> j)) & 0x01;
}

namespace detail {
TCM_FORCEINLINE constexpr auto flipped_impl(bits512&) noexcept -> void {}
template <class... UInts>
TCM_FORCEINLINE /*constexpr*/ auto flipped_impl(bits512& x, unsigned const i,
                                                UInts... is) noexcept -> void
{
    auto const chunk = i / 64U;
    auto const rest  = i % 64U;
    x.words[chunk] ^= (1UL << rest);
    flipped_impl(x, is...);
}
} // namespace detail

template <class... UInts>
TCM_FORCEINLINE /*constexpr*/ auto flipped(bits512 const& x,
                                           UInts... is) noexcept -> bits512
{
    auto y = x;
    detail::flipped_impl(y, is...);
    return y;
}

template <class UInt, class = std::enable_if_t<std::is_unsigned_v<UInt>>>
TCM_FORCEINLINE constexpr auto are_not_aligned(bits512 const& spin,
                                               UInt const     i,
                                               UInt const j) noexcept -> bool
{
    auto const fst = spin.words[i / UInt{64}] >> (i % UInt{64});
    auto const snd = spin.words[j / UInt{64}] >> (j % UInt{64});
    return (fst ^ snd) & 0x01;
}

template <class UInt>
TCM_FORCEINLINE constexpr auto gather_bits(uint64_t const spin, UInt const i,
                                           UInt const j) noexcept -> unsigned
{
    auto const fst = (spin >> i) & 1U;
    auto const snd = (spin >> j) & 1U;
    // ============================= IMPORTANT ==============================
    // The order is REALLY important here. This is done to adhere to the
    // definition of cronecker product, i.e. that
    //
    //     kron(A, B) =  A00 B     A01 B     A02 B  ...
    //                   A10 B     A11 B     A12 B  ...
    //                    .
    //                    .
    //                    .
    //
    // In other words, if you change it to `(snd << 1U) | fst` shit will break
    // in really difficult to track ways...
    return (fst << 1U) | snd;
}

template <class UInt>
TCM_FORCEINLINE constexpr auto gather_bits(bits512 const& spin, UInt const i,
                                           UInt const j) noexcept -> unsigned
{
    auto const fst = (spin.words[i / UInt{64}] >> (i % UInt{64})) & 1U;
    auto const snd = (spin.words[j / UInt{64}] >> (j % UInt{64})) & 1U;
    return (fst << 1U) | snd;
}

template <class UInt>
TCM_FORCEINLINE constexpr auto set_bit_to(uint64_t& bits, UInt const i,
                                          bool const value) noexcept -> void
{
    bits = (bits & ~(uint64_t{1} << i)) | (uint64_t{value} << i);
}

template <class UInt>
TCM_FORCEINLINE constexpr auto set_bit_to(bits512& bits, UInt const i,
                                          bool const value) noexcept -> void
{
    set_bit_to(bits.words[i / UInt{64}], i % UInt{64}, value);
}

template <class UInt>
TCM_FORCEINLINE constexpr auto scatter_bits(uint64_t spin, unsigned const bits,
                                            UInt const i, UInt const j) noexcept
    -> uint64_t
{
    set_bit_to(spin, i, (bits >> 1U) & 1U);
    set_bit_to(spin, j, bits & 1U);
    return spin;
}

template <class UInt>
TCM_FORCEINLINE constexpr auto scatter_bits(bits512 spin, unsigned const bits,
                                            UInt const i, UInt const j) noexcept
    -> bits512
{
    set_bit_to(spin, i, (bits >> 1U) & 1U);
    set_bit_to(spin, j, bits & 1U);
    return spin;
}

TCM_NAMESPACE_END
