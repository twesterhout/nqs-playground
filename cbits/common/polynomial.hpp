// Copyright (c) 2019-2020, Tom Westerhout
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

#include "accumulator.hpp"
#include <gsl/gsl-lite.hpp>
#include <cmath>
#include <vector>

TCM_NAMESPACE_BEGIN

class Polynomial {
  private:
    QuantumOperator                   _op;
    std::vector<std::complex<double>> _roots;
    uint64_t                          _max_states;
    bool                              _normalising;

  public:
    Polynomial(QuantumOperator op, std::vector<complex_type> roots, bool normalising);
    Polynomial(Polynomial const&)           = delete;
    Polynomial(Polynomial&& other) noexcept = default;
    Polynomial& operator=(Polynomial const&) = delete;
    Polynomial& operator=(Polynomial&&) = delete;

    auto degree() const noexcept -> uint64_t;
    auto roots() const noexcept -> gsl::span<std::complex<double> const>;

    auto operator()(ls_bits512 const& spin, complex_type coeff, gsl::span<ls_bits512> out_spins,
                    gsl::span<complex_type> out_coeffs) const -> uint64_t;

  private:
    struct Buffer;
    auto iteration(complex_type root, Buffer& buffer) const -> void;
};

TCM_NAMESPACE_END
