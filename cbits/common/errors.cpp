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

#include "errors.hpp"
#include <cstdio>

TCM_NAMESPACE_BEGIN

namespace detail {
TCM_EXPORT auto assert_fail(char const* expr, char const* file,
                            size_t const line, char const* function,
                            std::string const& msg) noexcept -> void
{
    std::fprintf(
        stderr,
        TCM_BUG_MESSAGE
        "\nAssertion failed at %s:%zu: %s: \"%s\" evaluated to false: %s\n",
        file, line, function, expr, msg.c_str());
    std::terminate();
}

TCM_EXPORT auto assert_fail(char const* expr, char const* file,
                            size_t const line, char const* function,
                            char const* msg) noexcept -> void
{
    std::fprintf(
        stderr,
        TCM_BUG_MESSAGE
        "\nAssertion failed at %s:%zu: %s: \"%s\" evaluated to false: %s\n",
        file, line, function, expr, msg);
    std::terminate();
}

TCM_EXPORT auto failed_to_construct_the_message() noexcept -> void
{
    std::fprintf(stderr,
                 "Failed to construct the message. Calling terminate...");
    std::terminate();
}

TCM_EXPORT auto make_what_message(char const* file, size_t const line,
                                  char const*        function,
                                  std::string const& description) -> std::string
{
    return fmt::format("{}:{}: {}: {}", file, line, function, description);
}

} // namespace detail

TCM_NAMESPACE_END
