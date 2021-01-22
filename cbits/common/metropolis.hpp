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

#include "bits512.hpp"
#include "random.hpp"
#include "tensor_info.hpp"
#include <gsl/gsl-lite.hpp>
#include <lattice_symmetries/lattice_symmetries.h>
#include <torch/types.h>

TCM_NAMESPACE_BEGIN

class MetropolisGenerator {
  private:
    gsl::not_null<ls_spin_basis const*> _basis;
    gsl::not_null<RandomGenerator*>     _generator;

  public:
    MetropolisGenerator(ls_spin_basis const& basis,
                        RandomGenerator&     generator = global_random_generator());
    ~MetropolisGenerator();

    MetropolisGenerator(MetropolisGenerator const&)     = default;
    MetropolisGenerator(MetropolisGenerator&&) noexcept = default;
    MetropolisGenerator& operator=(MetropolisGenerator const&) = default;
    MetropolisGenerator& operator=(MetropolisGenerator&&) noexcept = default;

    auto operator()(torch::Tensor x, c10::ScalarType dtype) const
        -> std::tuple<torch::Tensor, torch::Tensor>;

  private:
    template <class scalar_t>
    auto generate(TensorInfo<ls_bits512 const> src, TensorInfo<ls_bits512> dst,
                  TensorInfo<scalar_t> norm) const -> void;
};

TCM_NAMESPACE_END
