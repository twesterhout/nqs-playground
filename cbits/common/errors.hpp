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
#include <fmt/format.h>

#include <torch/types.h> // torch::ScalarType
#include <string>

TCM_NAMESPACE_BEGIN

namespace detail {
TCM_NORETURN auto assert_fail(char const* expr, char const* file, size_t line, char const* function,
                              std::string const& msg) noexcept -> void;

TCM_NORETURN auto assert_fail(char const* expr, char const* file, size_t line, char const* function,
                              char const* msg) noexcept -> void;

TCM_NORETURN auto failed_to_construct_the_message() noexcept -> void;

auto make_what_message(char const* file, size_t line, char const* func,
                       std::string const& description) -> std::string;
} // namespace detail

/// \brief An alternative to `fmt::format` that never throws.
///
/// If formatting fails, an empty string is returned. If even that fails,
/// `std::terminate` is called. The intended use case are asserts, when one
/// might want to add some info to the message, but when the assert can't
/// throw.
template <class T, class... Ts> auto noexcept_format(T&& fmt, Ts&&... args) noexcept -> std::string
{
    try {
        return fmt::format(std::forward<T>(fmt), std::forward<Ts>(args)...);
    }
    catch (...) {
        try {
            return std::string{};
        }
        catch (...) {
            detail::failed_to_construct_the_message();
        }
    }
}

TCM_NAMESPACE_END

#define TCM_ERROR(ExceptionType, ...)                                                              \
    throw ExceptionType                                                                            \
    {                                                                                              \
        ::TCM_NAMESPACE::detail::make_what_message(__FILE__, static_cast<size_t>(__LINE__),        \
                                                   __PRETTY_FUNCTION__, __VA_ARGS__)               \
    }

#define TCM_CHECK(condition, ExceptionType, ...)                                                   \
    if (TCM_UNLIKELY(!(condition))) { TCM_ERROR(ExceptionType, __VA_ARGS__); }                     \
    do {                                                                                           \
    } while (false)

#define TCM_CHECK_SHAPE(name, arg, ...)                                                            \
    do {                                                                                           \
        auto const _temp_shape_    = arg.sizes();                                                  \
        auto const _temp_expected_ = std::initializer_list<int64_t>(__VA_ARGS__);                  \
        TCM_CHECK(_temp_shape_.size() == _temp_expected_.size(), std::domain_error,                \
                  ::fmt::format("{} has wrong number of dimensions: {}; expected {}", name,        \
                                _temp_shape_.size(), _temp_expected_.size()));                     \
        TCM_CHECK(std::equal(_temp_expected_.begin(), _temp_expected_.end(), _temp_shape_.begin(), \
                             [](auto const _a894, auto const _b475) {                              \
                                 return _a894 == -1 || _a894 == _b475;                             \
                             }),                                                                   \
                  std::domain_error,                                                               \
                  ::fmt::format("{} has wrong shape: {}; expected {}", name,                       \
                                ::fmt::join(_temp_shape_, ", "),                                   \
                                ::fmt::join(_temp_expected_, ", ")));                              \
    } while (false)

#define TCM_CHECK_TYPE(name, arg, type)                                                            \
    TCM_CHECK(arg.scalar_type() == type, std::domain_error,                                        \
              ::fmt::format("{} has wrong type: {}; expected {}", name, arg.scalar_type(), type))

#define TCM_CHECK_CONTIGUOUS(name, arg)                                                            \
    TCM_CHECK(arg.is_contiguous(), std::domain_error,                                              \
              ::fmt::format("expected {} to be contiguous, but it has "                            \
                            "sizes={} and strides={}",                                             \
                            name, ::fmt::join(arg.sizes(), ", "),                                  \
                            ::fmt::join(arg.strides(), ", ")))

// [torch::ScalarType] Formatting {{{
/// Formatting of torch::ScalarType using fmtlib facilities
///
/// Used only for error reporting.
namespace fmt {
template <> struct formatter<::torch::ScalarType> : formatter<string_view> {
    // parse is inherited from formatter<string_view>.

    template <typename FormatContext>
    auto format(::torch::ScalarType const type, FormatContext& ctx)
        -> decltype(formatter<string_view>::format(std::declval<std::string>(),
                                                   std::declval<FormatContext&>()))
    {
        // TODO(twesterhout): Us using c10 is probably not what PyTorch folks had
        // in mind... Suggestions are welcome
        return formatter<string_view>::format(::c10::toString(type), ctx);
    }
};
} // namespace fmt
// [torch::ScalarType] Formatting }}}
