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
#include "errors.hpp"
#include <lattice_symmetries/lattice_symmetries.h>
#include <memory>

TCM_NAMESPACE_BEGIN

inline auto check_status_code(ls_error_code const code) -> void
{
    if (TCM_UNLIKELY(code != LS_SUCCESS)) {
        auto deleter = [](auto const* s) { ls_destroy_string(s); };
        auto c_str   = std::unique_ptr<char const, decltype(deleter)>{
            ls_error_to_string(code), deleter};
        TCM_ERROR(
            std::runtime_error,
            fmt::format("lattice_symmetries failed with error code {}: {}",
                        code, c_str.get()));
    }
}

namespace detail {
struct ls_operator_deleter {
    auto operator()(ls_operator* p) const noexcept -> void
    {
        ls_destroy_operator(p);
    }
};
} // namespace detail

struct ls_operator_wrapper
    : std::unique_ptr<ls_operator, detail::ls_operator_deleter> {};

TCM_NAMESPACE_END
