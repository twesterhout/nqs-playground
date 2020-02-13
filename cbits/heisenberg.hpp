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

#include "common.hpp"
#include "spin_basis.hpp"

#include <boost/align/aligned_allocator.hpp>
#include <gsl/gsl-lite.hpp>

TCM_NAMESPACE_BEGIN

class TCM_IMPORT Heisenberg : public std::enable_shared_from_this<Heisenberg> {
  public:
    using real_type    = double;
    using complex_type = std::complex<real_type>;
    using edge_type    = std::tuple<complex_type, uint16_t, uint16_t>;
    using spec_type =
        std::vector<edge_type,
                    boost::alignment::aligned_allocator<edge_type, 64>>;

  private:
    spec_type                        _edges; ///< Graph edges
    std::shared_ptr<SpinBasis const> _basis;
    unsigned _max_index; ///< The greatest site index present in `_edges`.
                         ///< It is used to detect errors when one tries to
                         ///< apply the hamiltonian to a spin configuration
                         ///< which is too short.
    bool _is_real;
    // mutable Pool _pool;

  public:
    /// Constructs a hamiltonian given graph edges and couplings.
    Heisenberg(spec_type edges, std::shared_ptr<SpinBasis const> basis);

    /// Copy and Move constructors/assignments
    Heisenberg(Heisenberg const&)     = default;
    Heisenberg(Heisenberg&&) noexcept = default;
    Heisenberg& operator=(Heisenberg const&) = default;
    Heisenberg& operator=(Heisenberg&&) noexcept = default;

    /// Returns the number of edges in the graph
    /*constexpr*/ auto size() const noexcept -> size_t { return _edges.size(); }

    /// Returns the greatest index encountered in `_edges`.
    ///
    /// \precondition `size() != 0`
    /*constexpr*/ auto max_index() const noexcept -> size_t
    {
        TCM_ASSERT(!_edges.empty(), "_max_index is not defined");
        return _max_index;
    }

    constexpr auto is_real() const noexcept -> bool { return _is_real; }

    /// Returns a *reference* to graph edges.
    /*constexpr*/ auto edges() const noexcept -> gsl::span<edge_type const>
    {
        return _edges;
    }

    auto basis() const noexcept -> std::shared_ptr<SpinBasis const>
    {
        return _basis;
    }

    template <class Callback>
    __attribute__((visibility("hidden"))) auto
    operator()(SpinBasis::StateT const spin, Callback&& callback) const -> void
    {
        TCM_ASSERT(_edges.empty() || max_index() < _basis->number_spins(),
                   fmt::format("`spin` is too short {}; expected >{}",
                               _basis->number_spins(), max_index()));
        auto const norm = std::get<2>(_basis->full_info(spin));
        TCM_CHECK(norm > 0.0, std::runtime_error,
                  fmt::format("state {} does not belong to the basis", spin));
        auto coeff = complex_type{0, 0};
        for (auto const [coupling, first, second] : edges()) {
            // Heisenberg hamiltonian works more or less like this:
            //
            //     K|↑↑⟩ = J|↑↑⟩
            //     K|↓↓⟩ = J|↓↓⟩
            //     K|↑↓⟩ = -J|↑↓⟩ + 2J|↓↑⟩
            //     K|↓↑⟩ = -J|↓↑⟩ + 2J|↑↓⟩
            //
            // where K is the "kernel". We want to perform
            // |ψ⟩ += c * K|σᵢσⱼ⟩ for each edge (i, j).
            //
            auto const not_aligned =
                ((spin >> first) ^ (spin >> second)) & 0x01;
            if (not_aligned) {
                coeff -= coupling;
                auto [branch_spin, branch_eigenvalue, branch_norm] =
                    _basis->full_info(
                        spin
                        ^ ((uint64_t{1} << first) | (uint64_t{1} << second)));
                if (branch_norm > 0.0) {
                    callback(branch_spin, 2.0 * coupling * branch_norm / norm
                                              * branch_eigenvalue);
                }
            }
            else {
                coeff += coupling;
            }
        }
        callback(spin, coeff);
    }

    auto diag(SpinBasis::StateT const spin) const noexcept -> complex_type
    {
        auto coeff = complex_type{0, 0};
        for (auto const [coupling, first, second] : edges()) {
            auto const not_aligned =
                static_cast<int>(((spin >> first) ^ (spin >> second)) & 0x01);
            coeff += static_cast<real_type>(1 - 2 * not_aligned) * coupling;
        }
        return coeff;
    }

    template <class T, class = std::enable_if_t<
                           std::is_floating_point_v<T> || is_complex_v<T>>>
    auto operator()(gsl::span<T const> x, gsl::span<T> y) const -> void;
};

TCM_NAMESPACE_END
