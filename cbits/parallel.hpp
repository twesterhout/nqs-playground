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

#include <atomic>
#include <type_traits>

#include <omp.h>

TCM_NAMESPACE_BEGIN

namespace detail {
/// Executes `f` for every `x` in `[begin, end)`.
template <class Function, class Int>
TCM_FORCEINLINE auto simple_for_loop(std::true_type, unsigned /*unused*/,
                                     Int begin, Int end, Function f) -> void
{
    for (; begin != end; ++begin) {
        f(begin);
    }
}

/// Similar to the previous overload except that we're given a `factory`
/// function which can produce an `f` rather than `f` itself.
template <class Factory, class Int>
TCM_FORCEINLINE auto simple_for_loop(std::false_type, unsigned worker,
                                     Int begin, Int end, Factory factory)
    -> void
{
    if (begin != end) {
        auto f = factory(worker);
        f(begin);
        for (++begin; begin != end; ++begin) {
            f(begin);
        }
    }
}

template <bool is_eager, class F>
TCM_FORCEINLINE auto parallel_for_impl(int64_t const begin, int64_t const end,
                                       F func, size_t const cutoff,
                                       int const number_threads) -> void
{
    static_assert(std::is_nothrow_copy_constructible<F>::value,
                  "`F` must be nothrow copy constructible to be safely usable "
                  "with OpenMP's firstprivate clause.");
    TCM_ASSERT(number_threads > 0, "invalid number of threads");
    TCM_ASSERT(begin <= end, fmt::format("invalid range [{}, {})", begin, end));
    using IsEager = std::integral_constant<bool, is_eager>;
    using FRef    = std::add_lvalue_reference_t<F>;
    if (static_cast<size_t>(end - begin) <= cutoff) { // Fallback to serial
        simple_for_loop<FRef>(IsEager{}, /*worker=*/0, begin, end, func);
        return;
    }

    std::atomic_flag   err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr err_ptr  = nullptr;
#pragma omp parallel num_threads(number_threads) default(none)                 \
    firstprivate(begin, end, func) shared(err_flag, err_ptr)
    {
        // This is basically a hand-rolled version of OpenMP's schedule(static),
        // but we need it for exception safety
        auto const num_threads = omp_get_num_threads();
        auto const thread_id   = omp_get_thread_num();
        auto const size        = end - begin;
        // TODO(twesterhout): Would moving `%` out of the parallel region help?
        auto const rest       = size % num_threads;
        auto const chunk_size = size / num_threads + (thread_id < rest);
        auto const thread_begin =
            begin + thread_id * chunk_size + (thread_id >= rest) * rest;
        if (thread_begin < end) {
            try {
                simple_for_loop<FRef>(
                    IsEager{}, /*worker=*/static_cast<unsigned>(thread_id),
                    thread_begin, thread_begin + chunk_size, func);
            }
            catch (...) {
                if (!err_flag.test_and_set()) {
                    err_ptr = std::current_exception();
                }
            }
        }
    }
    if (err_ptr != nullptr) { std::rethrow_exception(err_ptr); }
}
} // namespace detail

template <class Factory>
auto parallel_for_lazy(int64_t const begin, int64_t const end, Factory factory,
                       size_t const cutoff = 1, int const num_threads = -1)
    -> void
{
    detail::parallel_for_impl</*is_eager=*/false>(
        begin, end, std::move(factory), cutoff,
        num_threads > 0 ? num_threads : omp_get_max_threads());
}

template <class Function>
auto parallel_for(int64_t const begin, int64_t const end, Function f,
                  size_t const cutoff = 1, int const num_threads = -1) -> void
{
    detail::parallel_for_impl</*is_eager=*/true>(
        begin, end, std::move(f), cutoff,
        num_threads > 0 ? num_threads : omp_get_max_threads());
}

TCM_NAMESPACE_END
