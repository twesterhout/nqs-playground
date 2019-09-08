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

#if defined(TCM_GCC)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wsign-conversion"
#    pragma GCC diagnostic ignored "-Wsign-promo"
#    pragma GCC diagnostic ignored "-Wswitch-default"
#    pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
// #    pragma GCC diagnostic ignored "-Wstrict-overflow"
#elif defined(TCM_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-noreturn"
#    pragma clang diagnostic ignored "-Wsign-conversion"
#    pragma clang diagnostic ignored "-Wswitch-enum"
#    pragma clang diagnostic ignored "-Wundefined-func-template"
#endif
#include <fmt/format.h>
#if defined(TCM_GCC)
#    pragma GCC diagnostic pop
#elif defined(TCM_CLANG)
#    pragma clang diagnostic pop
#endif

#include <torch/types.h> // torch::ScalarType
#include <cstdio>
#include <string>

TCM_NAMESPACE_BEGIN

/// \brief An alternative to `fmt::format` that never throws.
///
/// If formatting fails, an empty string is returned. If even that fails,
/// `std::terminate` is called. The intended use case are asserts, when one
/// might want to add some info to the message, but when the assert can't
/// throw.
template <class T, class... Ts>
auto noexcept_format(T&& fmt, Ts&&... args) noexcept -> std::string
{
    try {
        return fmt::format(std::forward<T>(fmt), std::forward<Ts>(args)...);
    }
    catch (...) {
        try {
            return std::string{};
        }
        catch (...) {
            std::fprintf(
                stderr,
                "Failed to construct the message. Calling terminate...");
            std::terminate();
        }
    }
}

namespace detail {
TCM_NORETURN auto assert_fail(char const* expr, char const* file, size_t line,
                              char const*        function,
                              std::string const& msg) noexcept -> void;

TCM_NORETURN auto assert_fail(char const* expr, char const* file, size_t line,
                              char const* function, char const* msg) noexcept
    -> void;

auto make_what_message(char const* file, size_t line, char const* func,
                       std::string const& description) -> std::string;

constexpr auto is_dim_okay(int64_t const dimension,
                           int64_t const expected) noexcept -> bool
{
    return dimension == expected;
}

constexpr auto is_dim_okay(int64_t const dimension, int64_t const expected_1,
                           int64_t const expected_2) noexcept -> bool
{
    return dimension == expected_1 || dimension == expected_2;
}

constexpr auto is_shape_okay(int64_t const shape,
                             int64_t const expected) noexcept -> bool
{
    return shape == expected;
}

constexpr auto
is_shape_okay(std::tuple<int64_t, int64_t> const& shape,
              std::tuple<int64_t, int64_t> const& expected) noexcept -> bool
{
    return shape == expected;
}

auto make_wrong_dim_msg(int64_t dimension, int64_t expected) -> std::string;
auto make_wrong_dim_msg(int64_t dimension, int64_t expected_1,
                        int64_t expected_2) -> std::string;
auto make_wrong_shape_msg(int64_t const shape, int64_t const expected)
    -> std::string;
auto make_wrong_shape_msg(std::tuple<int64_t, int64_t> const& shape,
                          std::tuple<int64_t, int64_t> const& expected)
    -> std::string;
} // namespace detail

TCM_NAMESPACE_END

#define TCM_ERROR(ExceptionType, ...)                                          \
    throw ExceptionType                                                        \
    {                                                                          \
        ::TCM_NAMESPACE::detail::make_what_message(                            \
            __FILE__, static_cast<size_t>(__LINE__), BOOST_CURRENT_FUNCTION,   \
            __VA_ARGS__)                                                       \
    }

#define TCM_CHECK(condition, ExceptionType, ...)                               \
    if (TCM_UNLIKELY(!(condition))) { TCM_ERROR(ExceptionType, __VA_ARGS__); } \
    do {                                                                       \
    } while (false)

#define TCM_CHECK_DIM(dimension, ...)                                          \
    TCM_CHECK(                                                                 \
        ::TCM_NAMESPACE::detail::is_dim_okay(dimension, __VA_ARGS__),          \
        std::domain_error,                                                     \
        ::TCM_NAMESPACE::detail::make_wrong_dim_msg(dimension, __VA_ARGS__))

#define TCM_CHECK_SHAPE(...)                                                   \
    TCM_CHECK(::TCM_NAMESPACE::detail::is_shape_okay(__VA_ARGS__),             \
              std::domain_error,                                               \
              ::TCM_NAMESPACE::detail::make_wrong_shape_msg(__VA_ARGS__))

#define TCM_CHECK_TYPE(type, expected)                                         \
    TCM_CHECK(type == expected, std::domain_error,                             \
              ::fmt::format("wrong type: {}; expected {}", type, expected))

// [torch::ScalarType] Formatting {{{
/// Formatting of torch::ScalarType using fmtlib facilities
///
/// Used only for error reporting.
namespace fmt {
template <> struct formatter<::torch::ScalarType> : formatter<string_view> {
    // parse is inherited from formatter<string_view>.

    template <typename FormatContext>
    auto format(::torch::ScalarType const type, FormatContext& ctx)
    {
        // TODO(twesterhout): Us using c10 is probably not what PyTorch folks had
        // in mind... Suggestions are welcome
        return formatter<string_view>::format(::c10::toString(type), ctx);
    }
};
} // namespace fmt
  // [torch::ScalarType] Formatting }}}
