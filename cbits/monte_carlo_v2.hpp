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

#include "random.hpp"
#include "spin.hpp"

TCM_NAMESPACE_BEGIN

struct _Options {
    unsigned number_spins;
    int      magnetisation;
    unsigned number_chains;
    unsigned number_samples;
    unsigned sweep_size;
    unsigned number_discarded;

    _Options(unsigned number_spins, int magnetisation, unsigned number_chains,
             unsigned number_samples, unsigned sweep_size,
             unsigned number_discarded);

    constexpr _Options(_Options const&) noexcept = default;
    constexpr _Options(_Options&&) noexcept      = default;
    constexpr _Options& operator=(_Options const&) noexcept = default;
    constexpr _Options& operator=(_Options&&) noexcept = default;
};

template <class T>
using aligned_vector =
    std::vector<T, boost::alignment::aligned_allocator<T, std::max<size_t>(
                                                              64, alignof(T))>>;

namespace v2 {
auto sample_some(std::string const& filename, _Options const& options)
    -> std::tuple<aligned_vector<SpinVector>, aligned_vector<float>, float>;

auto sample_some(std::function<auto(torch::Tensor const&)->torch::Tensor> state,
                 _Options const& options)
    -> std::tuple<aligned_vector<SpinVector>, aligned_vector<float>, float>;
} // namespace v2

TCM_NAMESPACE_END

