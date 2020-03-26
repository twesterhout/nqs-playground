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
#include "symmetry.hpp"
#include "tensor_info.hpp"
#include <gsl/gsl-lite.hpp>
#include <torch/types.h>
#include <memory>

TCM_NAMESPACE_BEGIN

class BasisBase;

class TCM_IMPORT MetropolisKernel {
  private:
    std::shared_ptr<BasisBase const> _basis;
    gsl::not_null<RandomGenerator*>  _generator;

  public:
    MetropolisKernel(std::shared_ptr<BasisBase const> basis,
                     RandomGenerator& generator = global_random_generator());

    MetropolisKernel(MetropolisKernel const&)     = default;
    MetropolisKernel(MetropolisKernel&&) noexcept = default;
    MetropolisKernel& operator=(MetropolisKernel const&) = default;
    MetropolisKernel& operator=(MetropolisKernel&&) noexcept = default;

    auto operator()(torch::Tensor x) const
        -> std::tuple<torch::Tensor, torch::Tensor>;

    auto basis() const noexcept -> std::shared_ptr<BasisBase const>
    {
        return _basis;
    }

  private:
    inline auto kernel_cpu(TensorInfo<bits512 const> const& src_info,
                           TensorInfo<bits512> const&       dst_info,
                           TensorInfo<float> const& norm_info) const -> void;
};

TCM_NAMESPACE_END
