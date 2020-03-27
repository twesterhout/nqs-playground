// Copyright (c) 2020, Tom Westerhout
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
#include <fmt/format.h>

TCM_NAMESPACE_BEGIN

struct bits512 {
    alignas(64) uint64_t words[8];
};

inline auto operator==(bits512 const& x, bits512 const& y) noexcept -> bool
{
    using std::begin, std::end;
    return std::equal(begin(x.words), end(x.words), begin(y.words));
}

inline auto operator!=(bits512 const& x, bits512 const& y) noexcept -> bool
{
    return !(x == y);
}

inline auto operator<(bits512 const& x, bits512 const& y) noexcept -> bool
{
    for (auto i = 0; i < 8; ++i) {
        if (x.words[i] < y.words[i]) { return true; }
        else if (x.words[i] > y.words[i]) {
            return false;
        }
    }
    return false;
}

inline auto operator>(bits512 const& x, bits512 const& y) noexcept -> bool
{
    return y < x;
}

inline auto operator<=(bits512 const& x, bits512 const& y) noexcept -> bool
{
    return !(x > y);
}

inline auto operator>=(bits512 const& x, bits512 const& y) noexcept -> bool
{
    return !(x < y);
}

namespace detail {
// Compiler-friendly rotate left function. Both GCC and Clang are clever enough
// to replace it with a `rol` instruction.
constexpr auto rotl64(uint64_t n, uint32_t c) noexcept -> uint64_t
{
    constexpr uint32_t mask = 8 * sizeof(n) - 1;
    c &= mask;
    return (n << c) | (n >> ((-c) & mask));
}

constexpr auto fmix64(uint64_t k) noexcept -> uint64_t
{
    k ^= k >> 33U;
    k *= 0xff51afd7ed558ccdLLU;
    k ^= k >> 33U;
    k *= 0xc4ceb9fe1a85ec53LLU;
    k ^= k >> 33U;
    return k;
}

constexpr auto murmurhash3_x64_128(uint64_t const (&words)[8],
                                   uint64_t (&out)[2]) noexcept -> void
{
    constexpr uint64_t c1   = 0x87c37b91114253d5LLU;
    constexpr uint64_t c2   = 0x4cf5ad432745937fLLU;
    constexpr uint32_t seed = 0x208546c8U;
    constexpr int      size = 64;

    uint64_t h1 = seed;
    uint64_t h2 = seed;

    for (auto i = 0; i < size / 16; ++i) {
        auto k1 = words[i * 2 + 0];
        auto k2 = words[i * 2 + 1];

        k1 *= c1;
        k1 = rotl64(k1, 31);
        k1 *= c2;
        h1 ^= k1;

        h1 = rotl64(h1, 27);
        h1 += h2;
        h1 = h1 * 5 + 0x52dce729;

        k2 *= c2;
        k2 = rotl64(k2, 33);
        k2 *= c1;
        h2 ^= k2;

        h2 = rotl64(h2, 31);
        h2 += h1;
        h2 = h2 * 5 + 0x38495ab5;
    }

    // These are useless
    // h1 ^= size;
    // h2 ^= size;

    h1 += h2;
    h2 += h1;

    h1 = fmix64(h1);
    h2 = fmix64(h2);

    h1 += h2;
    h2 += h1;

    out[0] = h1;
    out[1] = h2;
}
} // namespace detail

TCM_NAMESPACE_END

// Formatting {{{
/// Formatting of bits512 using fmtlib facilities
///
/// Used only for error reporting.
namespace fmt {
template <>
struct formatter<::TCM_NAMESPACE::bits512> : formatter<string_view> {
    // parse is inherited from formatter<string_view>.

    template <typename FormatContext>
    auto format(::TCM_NAMESPACE::bits512 const& type, FormatContext& ctx)
    {
        // TODO(twesterhout): This is probably the stupidest possible way to
        // implement it... But since we only use it for debugging and error
        // reporting, who cares?
        auto s = fmt::format("[{}]", fmt::join(type.words, ","));
        return formatter<string_view>::format(s, ctx);
    }
};
} // namespace fmt
// }}}

namespace std {
/// Specialisation of std::hash for SpinVectors to use in QuantumState
template <> struct hash<::TCM_NAMESPACE::bits512> {
    auto operator()(::TCM_NAMESPACE::bits512 const& x) const noexcept -> size_t
    {
        uint64_t out[2];
        ::TCM_NAMESPACE::detail::murmurhash3_x64_128(x.words, out);
        // This part is questionable: should we mix the words in some way or is
        // this good enough...
        return out[0];
    }
};
} // namespace std
