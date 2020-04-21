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

#include "common.hpp"
#include "spin_basis.hpp"

#include <boost/align/aligned_allocator.hpp>
#include <gsl/gsl-lite.hpp>

TCM_NAMESPACE_BEGIN

struct Interaction {
  public:
    using real_type    = double;
    using complex_type = std::complex<real_type>;
    using edge_type    = std::tuple<uint16_t, uint16_t>;

  private:
    alignas(64) complex_type _matrix[16]; // Stored in column-major order
    std::vector<edge_type> _edges;

    constexpr auto column(unsigned const i) const noexcept
        -> complex_type const*
    {
        return _matrix + i * 4U;
    }

  public:
    Interaction(std::array<std::array<complex_type, 4>, 4> matrix,
                std::vector<edge_type>                     edges);

    Interaction(Interaction const&)     = default;
    Interaction(Interaction&&) noexcept = default;
    auto operator=(Interaction const&) -> Interaction& = default;
    auto operator=(Interaction&&) noexcept -> Interaction& = default;

    auto is_real() const noexcept -> bool;
    auto max_index() const noexcept -> uint16_t;

    /*constexpr*/ auto edges() const noexcept -> gsl::span<edge_type const>
    {
        return _edges;
    }

    constexpr auto matrix() const noexcept -> complex_type const (&)[16]
    {
        return _matrix;
    }

    template <class State>
    auto diag(State const& spin) const noexcept -> complex_type
    {
        auto r = complex_type{0.0, 0.0};
        for (auto const [i, j] : _edges) {
            auto const k = gather_bits(spin, i, j);
            r += column(k)[k];
        }
        return r;
    }

    template <class State, class Callback>
    auto operator()(State const& spin, complex_type& diagonal,
                    Callback off_diag) const
        noexcept(noexcept(std::declval<Callback&>()(
            std::declval<State>(), std::declval<complex_type>()))) -> void
    {
        for (auto const [i, j] : _edges) {
            auto const  k    = gather_bits(spin, i, j);
            auto const* data = column(k);
            auto        n    = 0U;

            for (; n < k; ++n) {
                if (data[n] != 0.0) {
                    off_diag(scatter_bits(spin, n, i, j), data[n]);
                }
            }
            diagonal += data[n++];
            for (; n < 4; ++n) {
                if (data[n] != 0.0) {
                    off_diag(scatter_bits(spin, n, i, j), data[n]);
                }
            }
        }
    }
};

class TCM_IMPORT Operator : public std::enable_shared_from_this<Operator> {
  public:
    using real_type    = Interaction::real_type;
    using complex_type = Interaction::complex_type;

  private:
    std::vector<Interaction>         _interactions;
    std::shared_ptr<BasisBase const> _basis;
    bool                             _is_real;

  public:
    /// Constructs a hamiltonian given graph edges and couplings.
    Operator(std::vector<Interaction>         interactions,
             std::shared_ptr<BasisBase const> basis);

    /// Copy and Move constructors/assignments
    Operator(Operator const&)     = default;
    Operator(Operator&&) noexcept = default;
    Operator& operator=(Operator const&) = default;
    Operator& operator=(Operator&&) noexcept = default;

    constexpr auto is_real() const noexcept -> bool { return _is_real; }

    auto basis() const noexcept -> std::shared_ptr<BasisBase const>
    {
        return _basis;
    }

    auto interactions() const noexcept -> gsl::span<Interaction const>
    {
        return _interactions;
    }

  private:
    template <class Basis, class State, class Callback>
    auto call_impl(State const& spin, Callback&& callback) const -> void
    {
        auto const* basis = static_cast<Basis const*>(_basis.get());
        auto const  norm  = std::get<2>(basis->full_info(spin));
        TCM_CHECK(norm > 0.0, std::runtime_error,
                  fmt::format("state does not belong to the basis"));
        auto       diagonal = complex_type{0, 0};
        auto const off_diag = [basis, norm_x = norm, &callback](auto const& x,
                                                                auto const& c) {
            auto [y, eigenvalue_y, norm_y] = basis->full_info(x);
            if (norm_y > 0.0) {
                callback(y, c * norm_y / norm_x * eigenvalue_y);
            }
        };
        for (auto const& interaction : _interactions) {
            interaction(spin, diagonal, std::cref(off_diag));
        }
        callback(spin, diagonal);
    }

  public:
    template <class Callback>
    auto operator()(bits512 const& spin, Callback&& callback) const -> void
    {
        auto const& basis = *_basis;
        if (typeid(basis) == typeid(SmallSpinBasis)) {
            return call_impl<SmallSpinBasis>(spin,
                                             std::forward<Callback>(callback));
        }
        if (typeid(basis) == typeid(BigSpinBasis)) {
            return call_impl<BigSpinBasis>(spin,
                                           std::forward<Callback>(callback));
        }
        TCM_ERROR(std::runtime_error,
                  fmt::format(
                      "invalid basis: {}; not yet supported by the Hamiltonian",
                      typeid(basis).name()));
    }

    template <class StateT,
              class = std::enable_if_t<
                  std::is_same_v<StateT,
                                 uint64_t> || std::is_same_v<StateT, bits512>>>
    auto diag(StateT const& spin) const noexcept -> complex_type
    {
        using std::begin, std::end;
        return std::accumulate(begin(_interactions), end(_interactions),
                               complex_type{0.0, 0.0},
                               [&spin](auto const& r, auto const& interaction) {
                                   return r + interaction.diag(spin);
                               });
    }

    template <class T, class = std::enable_if_t<
                           std::is_floating_point_v<T> || is_complex_v<T>>>
    auto operator()(gsl::span<T const> x, gsl::span<T> y) const -> void;

    template <class T>
    auto _to_sparse() const -> std::tuple<torch::Tensor, torch::Tensor>;
};

TCM_NAMESPACE_END
