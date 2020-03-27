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
    using std::begin;
    using std::end;
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
        -> decltype(formatter<string_view>::format(std::declval<std::string const&>(), std::declval<FormatContext&>()))
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
    TCM_EXPORT auto operator()(::TCM_NAMESPACE::bits512 const& x) const noexcept -> size_t;
};
} // namespace std
