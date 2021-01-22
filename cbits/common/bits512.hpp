// Copyright (c) 2020-2021, Tom Westerhout
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

#include "errors.hpp"
#include <fmt/format.h>
#include <lattice_symmetries/lattice_symmetries.h>

auto operator==(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool;
auto operator!=(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool;
auto operator<(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool;
auto operator>(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool;
auto operator<=(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool;
auto operator>=(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool;

TCM_NAMESPACE_BEGIN

constexpr auto toggle_bit(uint64_t& bits, unsigned const i) noexcept -> void
{
    TCM_ASSERT(i < 64U, "index out of bounds");
    bits ^= uint64_t{1} << uint64_t{i};
}
constexpr auto toggle_bit(ls_bits512& bits, unsigned const i) noexcept -> void
{
    TCM_ASSERT(i < 512U, "index out of bounds");
    return toggle_bit(bits.words[i / 64U], i % 64U);
}
constexpr auto test_bit(uint64_t const bits, unsigned const i) noexcept -> bool
{
    TCM_ASSERT(i < 64U, "index out of bounds");
    return static_cast<bool>((bits >> i) & 1U);
}
constexpr auto test_bit(ls_bits512 const& bits, unsigned const i) noexcept -> bool
{
    TCM_ASSERT(i < 512U, "index out of bounds");
    return test_bit(bits.words[i / 64U], i % 64U);
}
constexpr auto set_zero(uint64_t& bits) noexcept -> void { bits = 0UL; }
constexpr auto set_zero(ls_bits512& bits) noexcept -> void
{
    for (auto& w : bits.words) {
        set_zero(w);
    }
}

TCM_NAMESPACE_END

// Formatting {{{
/// Formatting of bits512 using fmtlib facilities
///
/// Used only for error reporting.
namespace fmt {
template <> struct formatter<ls_bits512> : formatter<string_view> {
    // parse is inherited from formatter<string_view>.

    template <typename FormatContext>
    auto format(ls_bits512 const& type, FormatContext& ctx)
        -> decltype(formatter<string_view>::format(std::declval<std::string const&>(),
                                                   std::declval<FormatContext&>()))
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
