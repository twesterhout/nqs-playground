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
#include "parallel.hpp"

#include <libcuckoo/cuckoohash_map.hh>

#include <cmath>
#include <memory>
#include <vector>

TCM_NAMESPACE_BEGIN

namespace v2 {
class QuantumState {
  private:
    using table_type = libcuckoo::cuckoohash_map<bits512, std::complex<double>>;

    table_type                              _table;
    std::optional<table_type::locked_table> _locked_table;

  public:
    QuantumState();
    QuantumState(QuantumState const&) = default;
    QuantumState(QuantumState&&)      = default;
    QuantumState& operator=(QuantumState const&) = delete;
    QuantumState& operator=(QuantumState&&) = delete;

    /// Performs `|ψ⟩ := |ψ⟩ + c|σ⟩`.
    ///
    /// \param value A pair `(c, |σ⟩)`.
    auto operator+=(std::pair<bits512 const, std::complex<double>> const& item)
        -> QuantumState&;

    auto norm() const -> double;
    auto empty() const noexcept -> bool { return _table.empty(); }
    auto freeze() -> void;
    auto unfreeze() -> void;
    auto clear() -> void;

    auto unsafe_locked_table() const -> table_type::locked_table const&;

    template <class Function>
    auto parallel_for(Function fn, uint64_t chunk_size, int num_threads) const
        -> void;

    template <class Function> auto for_each(Function&& fn) const -> void;

    friend auto swap(QuantumState& x, QuantumState& y) -> void;
};

template <class Function>
auto QuantumState::parallel_for(Function fn, uint64_t const chunk_size,
                                int num_threads) const -> void
{
    TCM_CHECK(_locked_table.has_value(), std::runtime_error,
              "table is not frozen");
    if (_locked_table->empty()) { return; }
    if (num_threads <= 0) { num_threads = omp_get_max_threads(); }

    omp_task_handler task_handler;
#pragma omp parallel default(none) num_threads(num_threads)                    \
    firstprivate(fn, chunk_size) shared(task_handler)
    {
#pragma omp single nowait
        {
            auto const count = _locked_table->size() / chunk_size;
            auto       first = _locked_table->cbegin();
            for (auto i = uint64_t{0}; i < count; ++i) {
                auto next = first;
                for (auto _j = uint64_t{0}; _j < chunk_size; ++_j) {
                    ++next;
                }
                task_handler.submit([first, next, fn]() { fn(first, next); });
                first = next;
            }
            fn(first, _locked_table->cend());
        }
    }
    task_handler.check_errors();
}

template <class Function>
auto QuantumState::for_each(Function&& fn) const -> void
{
    TCM_CHECK(_locked_table.has_value(), std::runtime_error,
              "table is not frozen");
    for (auto const& item : (*_locked_table)) {
        fn(item.first, item.second);
    }
}
} // namespace v2

#if 0
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
#endif

// Polynomial {{{
template <class Hamiltonian> class Polynomial {
  private:
    v2::QuantumState _current;
    v2::QuantumState _old;
    /// Hamiltonian which knows how to perform `|ψ⟩ += c * H|σ⟩`.
    std::shared_ptr<Hamiltonian const> _hamiltonian;
    /// List of roots A.
    std::vector<complex_type> _roots;
    bool                      _normalising;
    int                       _num_threads;

  public:
    /// Constructs the polynomial given the hamiltonian and a list or terms.
    TCM_NOINLINE Polynomial(std::shared_ptr<Hamiltonian const> hamiltonian,
                            std::vector<complex_type> roots, bool normalising,
                            int num_threads = -1);

    Polynomial(Polynomial const&)           = delete;
    Polynomial(Polynomial&& other) noexcept = default;
    Polynomial& operator=(Polynomial const&) = delete;
    Polynomial& operator=(Polynomial&&) = delete;

    inline auto degree() const noexcept -> size_t;
    inline auto hamiltonian() const noexcept
        -> std::shared_ptr<Hamiltonian const>;

    /// Applies the polynomial to state `|ψ⟩ = coeff * |spin⟩`.
    TCM_NOINLINE auto operator()(bits512 const& spin, complex_type coeff = 1.0)
        -> v2::QuantumState const&;

    /// Applies the polynomial to a state.
    TCM_NOINLINE auto operator()(v2::QuantumState const& state)
        -> v2::QuantumState const&;

  private:
    TCM_FORCEINLINE auto apply_hamiltonian(complex_type      coeff,
                                           bits512 const&    spin,
                                           v2::QuantumState& state) const
        -> void;

    TCM_FORCEINLINE auto iteration(complex_type root, v2::QuantumState& current,
                                   v2::QuantumState& old) const -> void;

    template <size_t Offset>
    TCM_FORCEINLINE auto kernel() -> v2::QuantumState const&;
}; // }}}

// Polynomial IMPLEMENTATION {{{
template <class Hamiltonian>
Polynomial<Hamiltonian>::Polynomial(
    std::shared_ptr<Hamiltonian const> hamiltonian,
    std::vector<complex_type> roots, bool const normalising,
    int const num_threads)
    : _current{}
    , _old{}
    , _hamiltonian{std::move(hamiltonian)}
    , _roots{std::move(roots)}
    , _normalising{normalising}
    , _num_threads{num_threads > 0 ? num_threads : omp_get_max_threads()}
{
    TCM_CHECK(_hamiltonian != nullptr, std::invalid_argument,
              "hamiltonian must not be nullptr (or None)");
    TCM_CHECK(!_roots.empty(), std::invalid_argument,
              "zero-degree polynomials are not supported");
    // auto const estimated_size =
    //     std::min(static_cast<size_t>(std::round(
    //                  std::pow(_hamiltonian->size() / 2, _roots.size()))),
    //              size_t{16384});
    // _old.reserve(estimated_size);
    // _current.reserve(estimated_size);
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
    -> v2::QuantumState const&
{
    TCM_CHECK(std::isfinite(coeff.real()) && std::isfinite(coeff.imag()),
              std::runtime_error,
              fmt::format("invalid coefficient {} + {}j; expected a finite "
                          "(i.e. either normal, subnormal or zero)",
                          coeff.real(), coeff.imag()));
    _old.unfreeze();
    _current.unfreeze();
    // `|_old⟩ := - coeff * root|spin⟩`
    _old.clear();
    _old += {spin, -coeff * _roots[0]};
    // `|_old⟩ += coeff * H|spin⟩`
    apply_hamiltonian(coeff, spin, _old);
    return kernel<1>();
}

template <class Hamiltonian>
auto Polynomial<Hamiltonian>::operator()(v2::QuantumState const& state)
    -> v2::QuantumState const&
{
    _old.unfreeze();
    _current.unfreeze();
    if (std::addressof(state) == std::addressof(_old)) { return kernel<0>(); }
    iteration(_roots[0], /*current=*/_old, /*old=*/state);
    return kernel<1>();
}

template <class Hamiltonian>
auto Polynomial<Hamiltonian>::apply_hamiltonian(complex_type      coeff,
                                                bits512 const&    spin,
                                                v2::QuantumState& state) const
    -> void
{
    (*_hamiltonian)(spin, [coeff, &state](auto const& x, auto const c) {
        state += {x, c * coeff};
    });
}

template <class Hamiltonian>
auto Polynomial<Hamiltonian>::iteration(complex_type      root,
                                        v2::QuantumState& current,
                                        v2::QuantumState& old) const -> void
{
    constexpr auto chunk_size = 256UL;
    TCM_ASSERT(current.empty(), "precondition violated");
    auto scale = real_type{1};
    old.freeze();
    if (_normalising) { scale = real_type{1} / std::sqrt(old.norm()); }
    // Performs `|current⟩ := (H - root)|old⟩` in two steps:
    // 1) `|current⟩ := - root|old⟩ / ‖old‖₂`
    // 2) `|current⟩ += H |old⟩ / ‖old‖₂`
    auto const fn = [this, &current, root, scale](auto first, auto last) {
        for (; first != last; ++first) {
            current += {first->first, -scale * root * first->second};
            apply_hamiltonian(first->second * scale, first->first, current);
        }
    };
    old.parallel_for(fn, chunk_size, _num_threads);
    old.unfreeze();
}

template <class Hamiltonian>
template <size_t Offset>
auto Polynomial<Hamiltonian>::kernel() -> v2::QuantumState const&
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
    _old.freeze();
    return _old;
}
// }}}

TCM_NAMESPACE_END
