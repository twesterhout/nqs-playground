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

#include "trim.hpp"

TCM_NAMESPACE_BEGIN

namespace detail {
template <class First, class Middle, class Last>
constexpr auto for_each_line(std::string_view str, First first_fn,
                             Middle middle_fn, Last last_fn) -> void
{
    auto prev = std::string_view::size_type{0};
    auto pos  = str.find_first_of('\n');
    if (pos != std::string_view::npos) {
        first_fn(str.substr(prev, pos - prev));
        prev = pos + 1;
        while ((pos = str.find_first_of('\n', prev))
               != std::string_view::npos) {
            middle_fn(str.substr(prev, pos - prev));
            prev = pos + 1;
        }
    }
    last_fn(str.substr(prev));
}

constexpr auto excess_indent(std::string_view const str) -> size_t
{
    auto       max    = std::string_view::npos;
    auto const update = [&max](auto const line) {
        auto const n = line.find_first_not_of(' ');
        if (n != 0 && n != std::string_view::npos && n < max) { max = n; }
    };
    for_each_line(
        str, [](auto) {}, update, update);
    return max != std::string_view::npos ? max : 0;
}
} // namespace detail

TCM_EXPORT auto trim(std::vector<std::string>& keep_alive, std::string_view raw)
    -> char const*
{
    auto const n = detail::excess_indent(raw);
    if (n == 0) { return raw.data(); }

    std::string out;
    out.reserve(raw.size());
    detail::for_each_line(
        raw,
        [&out](auto const line) {
            if (!line.empty()) {
                out.append(line.data(), line.size());
                out.push_back('\n');
            }
        },
        [&out, n](auto line) {
            line.remove_prefix(std::min(n, line.size()));
            out.append(line.data(), line.size());
            out.push_back('\n');
        },
        [&out, n](auto line) {
            line.remove_prefix(std::min(n, line.size()));
            out.append(line.data(), line.size());
        });
    keep_alive.push_back(std::move(out));
    return keep_alive.back().c_str();
}

TCM_NAMESPACE_END
