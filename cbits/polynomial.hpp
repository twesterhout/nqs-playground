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

#include "bits512.hpp"
#include "errors.hpp"

#include <boost/align/is_aligned.hpp>
#include <torch/types.h>
#include <flat_hash_map/bytell_hash_map.hpp>

#include <cmath>
#include <memory>
#include <vector>

TCM_NAMESPACE_BEGIN

class QuantumState
    : public ska::bytell_hash_map<bits512, std::complex<double>> {
  public:
    using base = ska::bytell_hash_map<bits512, std::complex<double>>;
    using base::value_type;

    using base::base;

    QuantumState(QuantumState const&) = default;
    QuantumState(QuantumState&&)      = default;
    QuantumState& operator=(QuantumState const&) = delete;
    QuantumState& operator=(QuantumState&&) = delete;

    /// Performs `|ψ⟩ := |ψ⟩ + c|σ⟩`.
    ///
    /// \param value A pair `(c, |σ⟩)`.
    TCM_FORCEINLINE TCM_HOT auto
    operator+=(typename base::value_type const& value) -> QuantumState&;

    friend inline auto swap(QuantumState& x, QuantumState& y) -> void;
};

auto keys(QuantumState const& state) -> torch::Tensor;
auto values(QuantumState const&, bool only_real = true) -> torch::Tensor;

auto QuantumState::operator+=(typename base::value_type const& value)
    -> QuantumState&
{
    TCM_ASSERT(std::isfinite(value.second.real())
                   && std::isfinite(value.second.imag()),
               fmt::format("invalid coefficient {} + {}j", value.second.real(),
                           value.second.imag()));
    auto& c = static_cast<base&>(*this)[value.first];
    c += value.second;
    return *this;
}

auto swap(QuantumState& x, QuantumState& y) -> void
{
    using std::swap;
    using base = QuantumState::base;
    static_cast<base&>(x).swap(static_cast<base&>(y));
}

#if 0
template <class State>
auto keys(QuantumState<State> const& psi) -> aligned_vector<State>
{
    using std::begin, std::end;
    aligned_vector<State> spins;
    spins.reserve(psi.size());
    std::transform(begin(psi), end(psi), std::back_inserter(spins),
                   [](auto const& item) { return item.first; });
    return spins;
}

template <class State>
auto values(QuantumState<State> const& psi, bool only_real) -> torch::Tensor
{
    using std::begin, std::end;
    if (only_real) {
        auto  coeffs = ::TCM_NAMESPACE::detail::make_tensor<float>(psi.size());
        auto* data   = reinterpret_cast<float*>(coeffs.data_ptr());
        std::transform(begin(psi), end(psi), data, [](auto const& item) {
            return static_cast<float>(item.second.real());
        });
        return coeffs;
    }
    else {
        auto coeffs =
            ::TCM_NAMESPACE::detail::make_tensor<float>(psi.size(), 2);
        auto* data = reinterpret_cast<std::complex<float>*>(coeffs.data_ptr());
        std::transform(begin(psi), end(psi), data,
                       [](auto const& item) { return item.second; });
        return coeffs;
    }
}
#endif

// Polynomial {{{
template <class Hamiltonian> class Polynomial {
  private:
    QuantumState _current;
    QuantumState _old;
    /// Hamiltonian which knows how to perform `|ψ⟩ += c * H|σ⟩`.
    std::shared_ptr<Hamiltonian const> _hamiltonian;
    /// List of roots A.
    std::vector<complex_type> _roots;
    bool                      _normalising;

  public:
    /// Constructs the polynomial given the hamiltonian and a list or terms.
    TCM_NOINLINE Polynomial(std::shared_ptr<Hamiltonian const> hamiltonian,
                            std::vector<complex_type> roots, bool normalising);

    Polynomial(Polynomial const&)           = delete;
    Polynomial(Polynomial&& other) noexcept = default;
    Polynomial& operator=(Polynomial const&) = delete;
    Polynomial& operator=(Polynomial&&) = delete;

    inline auto degree() const noexcept -> size_t;
    inline auto hamiltonian() const noexcept
        -> std::shared_ptr<Hamiltonian const>;

    /// Applies the polynomial to state `|ψ⟩ = coeff * |spin⟩`.
    TCM_NOINLINE auto operator()(bits512 const& spin, complex_type coeff = 1.0)
        -> QuantumState const&;

    /// Applies the polynomial to a state.
    TCM_NOINLINE auto operator()(QuantumState const& state)
        -> QuantumState const&;

  private:
    TCM_FORCEINLINE auto apply_hamiltonian(complex_type   coeff,
                                           bits512 const& spin,
                                           QuantumState&  state) const -> void;

    TCM_FORCEINLINE auto iteration(complex_type root, QuantumState& current,
                                   QuantumState const& old) const -> void;

    template <size_t Offset>
    TCM_FORCEINLINE auto kernel() -> QuantumState const&;
}; // }}}

// Polynomial IMPLEMENTATION {{{
template <class Hamiltonian>
Polynomial<Hamiltonian>::Polynomial(
    std::shared_ptr<Hamiltonian const> hamiltonian,
    std::vector<complex_type> roots, bool const normalising)
    : _current{}
    , _old{}
    , _hamiltonian{std::move(hamiltonian)}
    , _roots{std::move(roots)}
    , _normalising{normalising}
{
    TCM_CHECK(_hamiltonian != nullptr, std::invalid_argument,
              "hamiltonian must not be nullptr (or None)");
    TCM_CHECK(!_roots.empty(), std::invalid_argument,
              "zero-degree polynomials are not supported");
    auto const estimated_size =
        std::min(static_cast<size_t>(std::round(
                     std::pow(_hamiltonian->size() / 2, _roots.size()))),
                 size_t{16384});
    _old.reserve(estimated_size);
    _current.reserve(estimated_size);
}

template <class Hamiltonian>
auto Polynomial<Hamiltonian>::degree() const noexcept -> size_t
{
    return _roots.size();
}

template <class Hamiltonian>
auto Polynomial<Hamiltonian>::hamiltonian() const noexcept
    -> std::shared_ptr<Hamiltonian const>
{
    return _hamiltonian;
}

template <class Hamiltonian>
auto Polynomial<Hamiltonian>::operator()(bits512 const& spin,
                                         complex_type   coeff)
    -> QuantumState const&
{
    TCM_CHECK(std::isfinite(coeff.real()) && std::isfinite(coeff.imag()),
              std::runtime_error,
              fmt::format("invalid coefficient {} + {}j; expected a finite "
                          "(i.e. either normal, subnormal or zero)",
                          coeff.real(), coeff.imag()));
    // `|_old⟩ := - coeff * root|spin⟩`
    _old.clear();
    _old.emplace(spin, -coeff * _roots[0]);
    // `|_old⟩ += coeff * H|spin⟩`
    apply_hamiltonian(coeff, spin, _old);
    return kernel<1>();
}

template <class Hamiltonian>
auto Polynomial<Hamiltonian>::operator()(QuantumState const& state)
    -> QuantumState const&
{
    if (std::addressof(state) == std::addressof(_old)) { return kernel<0>(); }
    iteration(_roots[0], /*current=*/_old, /*old=*/state);
    return kernel<1>();
}

template <class Hamiltonian>
auto Polynomial<Hamiltonian>::apply_hamiltonian(complex_type   coeff,
                                                bits512 const& spin,
                                                QuantumState&  state) const
    -> void
{
    (*_hamiltonian)(spin, [coeff, &state](auto const& x, auto const c) {
        state += {x, c * coeff};
    });
}

template <class Hamiltonian>
auto Polynomial<Hamiltonian>::iteration(complex_type        root,
                                        QuantumState&       current,
                                        QuantumState const& old) const -> void
{
    TCM_ASSERT(current.empty(), "precondition violated");
    if (_normalising) {
        auto norm = real_type{0};
        // Performs `|current⟩ := (H - root)|old⟩ / ‖old‖₂` in two steps:
        // 1) `|current⟩ := - root|old⟩`
        for (auto const& item : old) {
            current.emplace(item.first, -root * item.second);
            norm += std::norm(item.second);
        }
        // `|current⟩ := |current⟩ / ‖old‖₂`
        auto const scale = real_type{1} / std::sqrt(norm);
        for (auto& item : current) {
            item.second *= scale;
        }
        // 2) `|current⟩ += H |old⟩ / ‖old‖₂`
        for (auto const& item : old) {
            apply_hamiltonian(item.second * scale, item.first, current);
        }
    }
    else {
        // Performs `|current⟩ := (H - root)|old⟩` in two steps:
        // 1) `|current⟩ := - root|old⟩`
        for (auto const& item : old) {
            current.emplace(item.first, -root * item.second);
        }
        // 2) `|current⟩ += H |old⟩`
        for (auto const& item : old) {
            apply_hamiltonian(item.second, item.first, current);
        }
    }
}

template <class Hamiltonian>
template <size_t Offset>
auto Polynomial<Hamiltonian>::kernel() -> QuantumState const&
{
    using std::swap;
    for (auto i = Offset; i < _roots.size(); ++i) {
        // `|_current⟩ := (H - root)|_old⟩`
        iteration(_roots[i], _current, _old);
        // |_old⟩ := |_current⟩, but to not waste allocated memory, we use
        // `swap + clear` instead.
        swap(_old, _current);
        _current.clear();
    }
    return _old;
}
// }}}

TCM_NAMESPACE_END
