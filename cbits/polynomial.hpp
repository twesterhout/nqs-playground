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

#include "config.hpp"
#include "errors.hpp"
#include "spin.hpp"

#include <torch/script.h>
#include <flat_hash_map/bytell_hash_map.hpp>

#include <memory>
#include <vector>

TCM_NAMESPACE_BEGIN

/// \brief Explicit representation of a quantum state `|ψ⟩`.
class QuantumState // {{{
    : public ska::bytell_hash_map<SpinVector, complex_type> {
  public:
    using base = ska::bytell_hash_map<SpinVector, complex_type>;
    using base::value_type;

    static_assert(alignof(base::value_type) == 16, "");
    static_assert(sizeof(base::value_type) == 32, "");
    static_assert(std::is_trivially_destructible<base::value_type>::value,
                  "\n" TCM_BUG_MESSAGE);
    // NOTE: std::complex is not trivially copyable, which is a shame...
    // static_assert(std::is_trivially_copyable<base::value_type>::value,
    //               "\n" TCM_BUG_MESSAGE);

    using base::base;

    QuantumState(QuantumState const&) = default;
    QuantumState(QuantumState&&)      = default;
    QuantumState& operator=(QuantumState const&) = delete;
    QuantumState& operator=(QuantumState&&) = delete;

    /// Performs `|ψ⟩ := |ψ⟩ + c|σ⟩`.
    ///
    /// \param value A pair `(c, |σ⟩)`.
    TCM_FORCEINLINE TCM_HOT auto
                    operator+=(std::pair<complex_type, SpinVector> const& value)
        -> QuantumState&
    {
        TCM_ASSERT(std::isfinite(value.first.real())
                       && std::isfinite(value.first.imag()),
                   fmt::format("Invalid coefficient ({}, {})",
                               value.first.real(), value.first.imag()));
        auto& c = static_cast<base&>(*this)[value.second];
        c += value.first;
        return *this;
    }

    friend auto swap(QuantumState& x, QuantumState& y) -> void
    {
        using std::swap;
        static_cast<base&>(x).swap(static_cast<base&>(y));
    }
}; // }}}

auto keys(QuantumState const&) -> aligned_vector<SpinVector>;
auto values(QuantumState const&, bool only_real = true) -> torch::Tensor;
auto items(QuantumState const&, bool only_real = true)
    -> std::pair<aligned_vector<SpinVector>, torch::Tensor>;

/// \brief Represents the Heisenberg Hamiltonian.
class Heisenberg // {{{
    : public std::enable_shared_from_this<Heisenberg> {
  public:
    using edge_type = std::tuple<real_type, uint16_t, uint16_t>;
    using spec_type =
        std::vector<edge_type,
                    boost::alignment::aligned_allocator<edge_type, 64>>;

  private:
    spec_type _edges;     ///< Graph edges
    unsigned  _max_index; ///< The greatest site index present in `_edges`.
                          ///< It is used to detect errors when one tries to
                          ///< apply the hamiltonian to a spin configuration
                          ///< which is too short.

  public:
    /// Constructs a hamiltonian given graph edges and couplings.
    Heisenberg(spec_type edges);

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

    /// Returns a *reference* to graph edges.
    /*constexpr*/ auto edges() const noexcept -> gsl::span<edge_type const>
    {
        return _edges;
    }

    /// Performs `|ψ⟩ += c * H|σ⟩`.
    ///
    /// \param coeff Coefficient `c`
    /// \param spin  Spin configuration `|σ⟩`
    /// \param psi   State `|ψ⟩`
    ///
    /// \precondition `coeff` is finite, i.e.
    ///               `isfinite(coeff.real()) && isfinite(coeff.imag())`.
    /// \preconfition When `size() != 0`, `max_index() < spin.size()`.
    TCM_FORCEINLINE TCM_HOT auto operator()(complex_type const coeff,
                                            SpinVector const   spin,
                                            QuantumState& psi) const -> void
    {
        TCM_ASSERT(std::isfinite(coeff.real()) && std::isfinite(coeff.imag()),
                   fmt::format("invalid coefficient ({}, {}); expected a "
                               "finite complex number",
                               coeff.real(), coeff.imag()));
        TCM_ASSERT(_edges.empty() || max_index() < spin.size(),
                   fmt::format("`spin` is too short {}; expected >{}",
                               spin.size(), max_index()));
        auto c = complex_type{0, 0};
        for (auto const& edge : edges()) {
            real_type coupling;
            uint16_t  first, second;
            std::tie(coupling, first, second) = edge;
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
            auto const aligned = spin[first] == spin[second];
            // sign == 1.0 when aligned == true and sign == -1.0
            // when aligned == false
            auto const sign = static_cast<real_type>(-1 + 2 * aligned);
            c += sign * coeff * coupling;
            if (!aligned) {
                psi += {real_type{2} * coeff * coupling,
                        spin.flipped({first, second})};
            }
        }
        psi += {c, spin};
    }

  private:
    /// Finds the largest index used in `_edges`.
    ///
    /// \precondition Range must not be empty.
    template <class Iter, class = std::enable_if_t<std::is_same<
                              typename std::iterator_traits<Iter>::value_type,
                              edge_type>::value> /**/>
    static auto find_max_index(Iter begin, Iter end) -> unsigned
    {
        TCM_ASSERT(begin != end, "Range is empty");
        // This implementation is quite inefficient, but it's not on the hot
        // bath, so who cares ;)
        auto max_index = std::max(std::get<1>(*begin), std::get<2>(*begin));
        ++begin;
        for (; begin != end; ++begin) {
            max_index = std::max(
                max_index, std::max(std::get<1>(*begin), std::get<2>(*begin)));
        }
        return max_index;
    }
}; // }}}

// [Polynomial] {{{
///
///
///
class Polynomial {
  private:
    QuantumState _current;
    QuantumState _old;
    /// Hamiltonian which knows how to perform `|ψ⟩ += c * H|σ⟩`.
    std::shared_ptr<Heisenberg const> _hamiltonian;
    /// List of roots A.
    std::vector<complex_type> _roots;
    bool _normalising;

  public:
    /// Constructs the polynomial given the hamiltonian and a list or terms.
    Polynomial(std::shared_ptr<Heisenberg const> hamiltonian,
               std::vector<complex_type> roots, bool normalising);

    Polynomial(Polynomial const&)           = delete;
    Polynomial(Polynomial&& other) noexcept = default;
    Polynomial& operator=(Polynomial const&) = delete;
    Polynomial& operator=(Polynomial&&) = delete;

    inline auto degree() const noexcept -> size_t;

    /// Applies the polynomial to state `|ψ⟩ = coeff * |spin⟩`.
    TCM_HOT auto operator()(complex_type coeff, SpinVector spin)
        -> QuantumState const&;

    /// Applies the polynomial to state.
    TCM_HOT auto operator()(QuantumState const& state) -> QuantumState const&;

  private:
    auto iteration(complex_type root, QuantumState& current,
                   QuantumState const& old) const -> void;

    template <size_t Offset> auto kernel() -> QuantumState const&;

#if 0
    template <class Map>
    TCM_NOINLINE auto save_results(Map const&                        map,
                                   torch::optional<real_type> const& eps)
        -> void;
#endif

#if 0
    template <class Map> static auto check_all_real(Map const& map) -> void
    {
        constexpr auto eps       = static_cast<real_type>(2e-3);
        auto           norm_full = real_type{0};
        auto           norm_real = real_type{0};
        for (auto const& item : map) {
            norm_full += std::norm(item.second);
            norm_real += std::norm(item.second.real());
        }
        if (norm_full >= (real_type{1} + eps) * norm_real) {
            throw std::runtime_error{
                "Polynomial contains complex coefficients: |P| >= (1 + eps) * "
                "|Re[P]|: |"
                + std::to_string(norm_full) + "| >= (1 + " + std::to_string(eps)
                + ") * |" + std::to_string(norm_real) + "|"};
        }
    }
#endif
};

inline auto Polynomial::degree() const noexcept -> size_t
{
    return _roots.size();
}

#if 0
template <class Map>
auto Polynomial::save_results(Map const& map, optional<real_type> const& eps)
    -> void
{
    // TODO(twesterhout): We might be seriously wasting memory here (i.e.
    // using ~5% of the allocated storage).
    auto const size = map.size();
    _basis.clear();
    _basis.reserve(size);
    if (!_coeffs.defined()) { _coeffs = detail::make_tensor<float>(size); }
    else if (_coeffs.size(0) != static_cast<int64_t>(size)) {
        _coeffs.resize_({static_cast<int64_t>(size)});
    }

    auto i        = int64_t{0};
    auto accessor = _coeffs.packed_accessor<float, 1>();
    if (eps.has_value()) {
        for (auto const& item : map) {
            if (std::abs(item.second.real()) >= *eps) {
                _basis.emplace_back(item.first);
                accessor[i++] = static_cast<float>(item.second.real());
            }
        }
        TCM_ASSERT(_basis.size() == static_cast<size_t>(i), "");
        _coeffs.resize_(i);
    }
    else {
        for (auto const& item : map) {
            _basis.emplace_back(item.first);
            accessor[i++] = static_cast<float>(item.second.real());
        }
        TCM_ASSERT(i == static_cast<int64_t>(size), "");
    }
}
#endif
// [Polynomial] }}}

#if 0
// [VarAccumulator] {{{
struct VarAccumulator {
    using value_type = real_type;

  private:
    size_t     _count;
    value_type _mean;
    value_type _M2;

  public:
    constexpr VarAccumulator() noexcept : _count{0}, _mean{0}, _M2{0} {}

    constexpr VarAccumulator(VarAccumulator const&) noexcept = default;
    constexpr VarAccumulator(VarAccumulator&&) noexcept      = default;
    constexpr VarAccumulator&
                              operator=(VarAccumulator const&) noexcept = default;
    constexpr VarAccumulator& operator=(VarAccumulator&&) noexcept = default;

    constexpr auto operator()(value_type const x) noexcept -> void
    {
        ++_count;
        auto const delta = x - _mean;
        _mean += delta / _count;
        _M2 += delta * (x - _mean);
    }

    constexpr auto count() const noexcept -> size_t { return _count; }

    constexpr auto mean() const -> value_type
    {
        TCM_CHECK(_count > 0, std::runtime_error,
                  fmt::format("mean of 0 samples is not defined"));
        return _mean;
    }

    constexpr auto variance() const -> value_type
    {
        TCM_CHECK(_count > 1, std::runtime_error,
                  fmt::format("sample variance of {} samples is not defined",
                              _count));
        return _M2 / (_count - 1);
    }

    constexpr auto merge(VarAccumulator const& other) noexcept
        -> VarAccumulator&
    {
        if (_count != 0 || other._count != 0) {
            auto const sum = _mean * _count + other._mean * other._count;
            _count += other._count;
            _mean = sum / _count;
            _M2 += other._M2;
        }
        return *this;
    }
};

static_assert(std::is_trivially_copyable<VarAccumulator>::value, "");
static_assert(std::is_trivially_destructible<VarAccumulator>::value, "");
// }}}
#endif

#if 0
// [PolynomialState] {{{
class PolynomialState {
    struct Worker {
      private:
        ForwardT                         _forward;
        gsl::not_null<Polynomial const*> _polynomial;
        torch::Tensor                    _buffer;
        size_t                           _batch_size;
        size_t                           _num_spins;

        static_assert(
            std::is_nothrow_move_constructible<decltype(_forward)>::value
                && std::is_nothrow_move_assignable<decltype(_forward)>::value,
            TCM_STATIC_ASSERT_BUG_MESSAGE);
        static_assert(
            std::is_nothrow_move_constructible<decltype(_polynomial)>::value
                && std::is_nothrow_move_assignable<
                       decltype(_polynomial)>::value,
            TCM_STATIC_ASSERT_BUG_MESSAGE);

      public:
        Worker(ForwardT f, Polynomial const& p, size_t const batch_size,
               size_t const num_spins)
            : _forward{std::move(f)}
            , _polynomial{std::addressof(p)}
            , _buffer{detail::make_tensor<float>(batch_size, num_spins)}
            , _batch_size{batch_size}
            , _num_spins{num_spins}
        {
            // Access the memory to make sure it belongs to us
            // if (batch_size * num_spins != 0) { *_buffer.data<float>() = 0.0f; }
        }

        Worker(Worker const&)     = delete;
        Worker(Worker&&) noexcept = default;
        Worker& operator=(Worker const&) = delete;
        // torch::Tensor is not noexcept assignable, so we delete the assignment
        // altogether. Otherwise, we'll have trouble with OpenMP.
        Worker& operator=(Worker&&) = delete;

        auto operator()(int64_t batch_index) -> float;

        constexpr auto batch_size() const noexcept -> size_t
        {
            return _batch_size;
        }

        constexpr auto number_spins() const noexcept -> size_t
        {
            return _num_spins;
        }

      private:
        auto forward_propagate_batch(size_t i) -> float;
        auto forward_propagate_rest(size_t i) -> float;
    };

  private:
    Polynomial _poly;
    std::vector<Worker, boost::alignment::aligned_allocator<Worker, 64>>
                   _workers;
    VarAccumulator _poly_time;
    VarAccumulator _psi_time;

  public:
    /// Creates a state with one worker
    PolynomialState(ForwardT psi, Polynomial poly,
                    std::tuple<size_t, size_t> dim)
        : _poly{std::move(poly)}, _workers{}, _poly_time{}, _psi_time{}
    {
        _workers.emplace_back(std::move(psi), _poly, std::get<0>(dim),
                              std::get<1>(dim));
    }

    PolynomialState(std::vector<ForwardT> psis, Polynomial poly,
                    std::tuple<size_t, size_t> dim)
        : _poly{std::move(poly)}, _workers{}, _poly_time{}, _psi_time{}
    {
        TCM_CHECK(psis.size() <= max_number_workers(), std::runtime_error,
                  fmt::format("too many workers specified: {}; expected <={}",
                              psis.size(), max_number_workers()));
        _workers.reserve(psis.size());
        for (auto i = size_t{0}; i < psis.size(); ++i) {
            _workers.emplace_back(std::move(psis[i]), _poly, std::get<0>(dim),
                                  std::get<1>(dim));
        }
    }

    static constexpr auto max_number_workers() noexcept -> size_t { return 32; }

    auto operator()(SpinVector) -> float;

    auto time_poly() const -> std::pair<real_type, real_type>;
    auto time_psi() const -> std::pair<real_type, real_type>;
};
// }}}
#endif

auto load_forward_fn(std::string const& filename) -> ForwardT;
// auto load_forward_fn(std::string const& filename, size_t count)
//     -> std::vector<ForwardT>;

auto bind_heisenberg(pybind11::module) -> void;
auto bind_explicit_state(pybind11::module m) -> void;
auto bind_polynomial(pybind11::module) -> void;

TCM_NAMESPACE_END
