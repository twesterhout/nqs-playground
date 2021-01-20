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

#include "../bits512.hpp"
#include "../tensor_info.hpp"
#include <gsl/gsl-lite.hpp>
#include <lattice_symmetries/lattice_symmetries.h>
#include <torch/types.h>

TCM_NAMESPACE_BEGIN

class ZanellaGenerator {
  private:
    gsl::not_null<ls_spin_basis const*> _basis;

  public:
    ZanellaGenerator(ls_spin_basis const& basis);
    ~ZanellaGenerator();

    auto operator()(torch::Tensor x) const -> std::tuple<torch::Tensor, torch::Tensor>;

    auto max_states() const noexcept -> uint64_t
    {
        auto const number_spins   = ls_get_number_spins(_basis);
        auto const hamming_weight = ls_get_hamming_weight(_basis);
        if (hamming_weight != -1) {
            auto const m = static_cast<unsigned>(hamming_weight);
            return m * (number_spins - m);
        }
        // By construction, number_spins > 1
        return number_spins * (number_spins - 1);
    }

  private:
    auto generate(ls_bits512 const& spin, TensorInfo<ls_bits512, 1> out) const -> unsigned;
};

auto zanella_choose_samples(torch::Tensor weights, int64_t number_samples, double time_step,
                            c10::Device device) -> torch::Tensor;

TCM_NAMESPACE_END
