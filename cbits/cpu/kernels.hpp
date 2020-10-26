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

#include "../bits512.hpp"
#include "../tensor_info.hpp"
#include <torch/types.h>

TCM_NAMESPACE_BEGIN

auto dotu_cpu(TensorInfo<std::complex<float> const> const& x,
              TensorInfo<std::complex<float> const> const& y)
    -> std::complex<double>;

auto zanella_jump_rates(torch::Tensor               current_log_prob,
                        torch::Tensor               proposed_log_prob,
                        std::vector<int64_t> const& counts)
    -> std::tuple<torch::Tensor, torch::Tensor>;

template <class scalar_t>
auto tabu_jump_rates(gsl::span<scalar_t const> proposed_log_prob,
                     scalar_t current_log_prob) -> std::vector<scalar_t>;

template <class T>
auto jump_rates_one_avx2(TensorInfo<T> const&, TensorInfo<T const> const&,
                         T) noexcept -> T;
template <class T>
auto jump_rates_one_avx(TensorInfo<T> const&, TensorInfo<T const> const&,
                        T) noexcept -> T;
template <class T>
auto jump_rates_one_sse2(TensorInfo<T> const&, TensorInfo<T const> const&,
                         T) noexcept -> T;

template <class Bits>
auto unpack_cpu(TensorInfo<Bits const> const& spins,
                TensorInfo<float, 2> const&   out) -> void;

auto unpack_one_avx2(bits512 const&, unsigned, float*) noexcept -> void;
auto unpack_one_avx2(uint64_t, unsigned, float*) noexcept -> void;
auto unpack_one_avx(bits512 const&, unsigned, float*) noexcept -> void;
auto unpack_one_avx(uint64_t, unsigned, float*) noexcept -> void;
auto unpack_one_sse2(bits512 const&, unsigned, float*) noexcept -> void;
auto unpack_one_sse2(uint64_t, unsigned, float*) noexcept -> void;

auto bfly(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;
auto bfly_avx(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;
auto bfly_sse2(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;

auto ibfly(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;
auto ibfly_avx(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;
auto ibfly_sse2(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;

auto bfly(uint64_t x, uint64_t out[8], uint64_t const (*masks)[8]) noexcept
    -> void;
auto bfly_avx(uint64_t x, uint64_t out[8], uint64_t const (*masks)[8]) noexcept
    -> void;
auto bfly_sse2(uint64_t x, uint64_t out[8], uint64_t const (*masks)[8]) noexcept
    -> void;

// These are not really used anywhere
auto bfly_avx2(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;
auto bfly_avx2(uint64_t x, uint64_t out[8], uint64_t const (*masks)[8]) noexcept
    -> void;
auto ibfly_avx2(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;

TCM_NAMESPACE_END
