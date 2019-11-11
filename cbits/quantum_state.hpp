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
#include "config.hpp"
#include "errors.hpp"

#include <boost/align/is_aligned.hpp>
#include <gsl/gsl-lite.hpp>
#include <torch/extension.h>
#include <vectorclass/version2/vectorclass.h>
#include <flat_hash_map/bytell_hash_map.hpp>

#include <cmath>
#include <memory>
#include <vector>

extern "C" { // From Intel SVML (libsvml.so)
__m256 __svml_cexpf8(__m256);
}

TCM_NAMESPACE_BEGIN

namespace v2 {
namespace detail {

    // QuantumState {{{
    template <class State>
    class QuantumState : public ska::bytell_hash_map<State, complex_type> {
      public:
        using base = ska::bytell_hash_map<State, complex_type>;
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

        template <class S>
        friend auto swap(QuantumState<S>& x, QuantumState<S>& y) -> void;
    };

    template <class State>
    inline auto keys(QuantumState<State> const&) -> aligned_vector<State>;

    template <class State>
    inline auto values(QuantumState<State> const&, bool only_real = true)
        -> torch::Tensor;
    // }}}

    // QuantumState IMPLEMENTATION {{{
    template <class State>
    auto QuantumState<State>::operator+=(typename base::value_type const& value)
        -> QuantumState&
    {
        TCM_ASSERT(std::isfinite(value.second.real())
                       && std::isfinite(value.second.imag()),
                   fmt::format("invalid coefficient {} + {}j",
                               value.second.real(), value.second.imag()));
        auto& c = static_cast<base&>(*this)[value.first];
        c += value.second;
        return *this;
    }

    template <class State>
    auto swap(QuantumState<State>& x, QuantumState<State>& y) -> void
    {
        using std::swap;
        using base = typename QuantumState<State>::base;
        static_cast<base&>(x).swap(static_cast<base&>(y));
    }

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
            auto coeffs =
                ::TCM_NAMESPACE::detail::make_tensor<float>(psi.size());
            auto* data = reinterpret_cast<float*>(coeffs.data_ptr());
            std::transform(begin(psi), end(psi), data, [](auto const& item) {
                return static_cast<float>(item.second.real());
            });
            return coeffs;
        }
        else {
            auto coeffs =
                ::TCM_NAMESPACE::detail::make_tensor<float>(psi.size(), 2);
            auto* data =
                reinterpret_cast<std::complex<float>*>(coeffs.data_ptr());
            std::transform(begin(psi), end(psi), data,
                           [](auto const& item) { return item.second; });
            return coeffs;
        }
    }
    // }}}

    // Polynomial {{{
    template <class State, class Hamiltonian> class Polynomial {
      private:
        QuantumState<State> _current;
        QuantumState<State> _old;
        /// Hamiltonian which knows how to perform `|ψ⟩ += c * H|σ⟩`.
        std::shared_ptr<Hamiltonian const> _hamiltonian;
        /// List of roots A.
        std::vector<complex_type> _roots;
        bool                      _normalising;

      public:
        /// Constructs the polynomial given the hamiltonian and a list or terms.
        TCM_NOINLINE Polynomial(std::shared_ptr<Hamiltonian const> hamiltonian,
                                std::vector<complex_type>          roots,
                                bool                               normalising);

        Polynomial(Polynomial const&)           = delete;
        Polynomial(Polynomial&& other) noexcept = default;
        Polynomial& operator=(Polynomial const&) = delete;
        Polynomial& operator=(Polynomial&&) = delete;

        inline auto degree() const noexcept -> size_t;
        inline auto hamiltonian() const noexcept
            -> std::shared_ptr<Hamiltonian const>;

        /// Applies the polynomial to state `|ψ⟩ = coeff * |spin⟩`.
        TCM_NOINLINE auto operator()(complex_type coeff, State spin)
            -> QuantumState<State> const&;

        /// Applies the polynomial to state.
        TCM_NOINLINE auto operator()(QuantumState<State> const& state)
            -> QuantumState<State> const&;

      private:
        TCM_FORCEINLINE auto
        apply_hamiltonian(complex_type coeff, State spin,
                          QuantumState<State>& state) const;

        TCM_FORCEINLINE auto iteration(complex_type               root,
                                       QuantumState<State>&       current,
                                       QuantumState<State> const& old) const
            -> void;

        template <size_t Offset>
        TCM_FORCEINLINE auto kernel() -> QuantumState<State> const&;
    }; // }}}

    // Polynomial IMPLEMENTATION {{{
    template <class State, class Hamiltonian>
    Polynomial<State, Hamiltonian>::Polynomial(
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

    template <class State, class Hamiltonian>
    auto Polynomial<State, Hamiltonian>::degree() const noexcept -> size_t
    {
        return _roots.size();
    }

    template <class State, class Hamiltonian>
    auto Polynomial<State, Hamiltonian>::hamiltonian() const noexcept
        -> std::shared_ptr<Hamiltonian const>
    {
        return _hamiltonian;
    }

    template <class State, class Hamiltonian>
    auto Polynomial<State, Hamiltonian>::operator()(complex_type coeff,
                                                    State const  spin)
        -> QuantumState<State> const&
    {
        TCM_CHECK(std::isfinite(coeff.real()) && std::isfinite(coeff.imag()),
                  std::runtime_error,
                  fmt::format("invalid coefficient {} + {}j; expected a finite "
                              "(i.e. either normal, subnormal or zero)",
                              coeff.real(), coeff.imag()));
#if 0
    TCM_CHECK(_hamiltonian->max_index() < spin.size(), std::out_of_range,
              fmt::format("spin configuration too short {}; expected >{}",
                          spin.size(), _hamiltonian->max_index()));
#endif
        // `|_old⟩ := - coeff * root|spin⟩`
        _old.clear();
        _old.emplace(spin, -coeff * _roots[0]);
        // `|_old⟩ += coeff * H|spin⟩`
        // (*_hamiltonian)(coeff, spin, _old);
        apply_hamiltonian(coeff, spin, _old);
        return kernel<1>();
    }

    template <class State, class Hamiltonian>
    auto
    Polynomial<State, Hamiltonian>::operator()(QuantumState<State> const& state)
        -> QuantumState<State> const&
    {
        if (std::addressof(state) == std::addressof(_old)) {
            return kernel<0>();
        }
        iteration(_roots[0], /*current=*/_old, /*old=*/state);
        return kernel<1>();
    }

    template <class State, class Hamiltonian>
    auto Polynomial<State, Hamiltonian>::apply_hamiltonian(
        complex_type coeff, State spin, QuantumState<State>& state) const
    {
        (*_hamiltonian)(spin, [coeff, &state](auto const x, auto const c) {
            state += {x, c * coeff};
        });
    }

    template <class State, class Hamiltonian>
    auto Polynomial<State, Hamiltonian>::iteration(
        complex_type root, QuantumState<State>& current,
        QuantumState<State> const& old) const -> void
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
                // (*_hamiltonian)(item.second * scale, item.first, current);
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
                // (*_hamiltonian)(item.second, item.first, current);
                apply_hamiltonian(item.second, item.first, current);
            }
        }
    }

    template <class State, class Hamiltonian>
    template <size_t Offset>
    auto Polynomial<State, Hamiltonian>::kernel() -> QuantumState<State> const&
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

    // ForwardPropagator {{{
    template <class State> struct ForwardPropagator {
      private:
        aligned_vector<State>        _spins;
        aligned_vector<complex_type> _coeffs;
        size_t                       _count;
        size_t                       _batch_size;

        inline auto coeffs() const noexcept -> gsl::span<complex_type const>;

        static_assert(std::is_nothrow_move_assignable<State>::value,
                      TCM_STATIC_ASSERT_BUG_MESSAGE);

      public:
        TCM_NOINLINE explicit ForwardPropagator(unsigned batch_size);

        constexpr auto batch_size() const noexcept -> size_t;
        constexpr auto full() const noexcept -> bool;
        constexpr auto empty() const noexcept -> bool;
        inline auto push(State spin, complex_type coeff) TCM_NOEXCEPT -> void;
        inline auto fill() TCM_NOEXCEPT -> void;
        inline auto clear() noexcept -> void;

        template <class ForwardFn>
        inline auto run(ForwardFn&& fn)
            -> std::pair<gsl::span<complex_type const>, torch::Tensor>;
    }; // }}}

    // ForwardPropagator IMPLEMENTATION {{{
    template <class State>
    ForwardPropagator<State>::ForwardPropagator(unsigned const batch_size)
        : _spins{}, _coeffs{}, _count{0}, _batch_size{batch_size}
    {
        TCM_CHECK(
            batch_size > 0, std::invalid_argument,
            fmt::format("invalid batch size: {}; expected a positive integer",
                        batch_size));
        _spins.resize(_batch_size, State{});
        constexpr auto NaN = std::numeric_limits<real_type>::quiet_NaN();
        _coeffs.resize(_batch_size, complex_type{NaN, NaN});
    }

    template <class State>
    auto ForwardPropagator<State>::coeffs() const noexcept
        -> gsl::span<complex_type const>
    {
        TCM_ASSERT(_coeffs.size() == batch_size(),
                   "ForwardPropagator is in an invalid state");
        return _coeffs;
    }

    template <class State>
    auto ForwardPropagator<State>::clear() noexcept -> void
    {
        using std::begin, std::end;
        std::fill(begin(_spins), end(_spins), State{});
        constexpr auto NaN = std::numeric_limits<real_type>::quiet_NaN();
        std::fill(begin(_coeffs), end(_coeffs), complex_type{NaN, NaN});
        _count = 0;
    }

    template <class State>
    constexpr auto ForwardPropagator<State>::batch_size() const noexcept
        -> size_t
    {
        return _batch_size;
    }

    template <class State>
    constexpr auto ForwardPropagator<State>::full() const noexcept -> bool
    {
        TCM_ASSERT(_count <= _batch_size, "precondition violated");
        return _count == _batch_size;
    }

    template <class State>
    constexpr auto ForwardPropagator<State>::empty() const noexcept -> bool
    {
        TCM_ASSERT(_count <= _batch_size, "precondition violated");
        return _count == 0;
    }

    template <class State>
    auto ForwardPropagator<State>::push(State        spin,
                                        complex_type coeff) TCM_NOEXCEPT -> void
    {
        TCM_ASSERT(!full(), "buffer is full");
        _spins[_count]  = std::move(spin);
        _coeffs[_count] = std::move(coeff);
        ++_count;
    }

    template <class State>
    auto ForwardPropagator<State>::fill() TCM_NOEXCEPT -> void
    {
        TCM_ASSERT(!empty(), "precondition violated");
        auto spin = _spins[_count - 1];
        for (; _count < _batch_size; ++_count) {
            _spins[_count]  = spin;
            _coeffs[_count] = complex_type{0, 0};
        }
        TCM_ASSERT(full(), "postcondition violated");
    }

    template <class State>
    template <class ForwardFn>
    auto ForwardPropagator<State>::run(ForwardFn&& fn)
        -> std::pair<gsl::span<complex_type const>, torch::Tensor>
    {
        TCM_ASSERT(full(), "batch is not yet filled");
        TCM_ASSERT(_spins.size() == _batch_size, "precondition violated");
        auto output = std::forward<ForwardFn>(fn)(_spins);
        TCM_CHECK_SHAPE("output tensor", output,
                        {static_cast<int64_t>(_batch_size), 2});
        TCM_CHECK_CONTIGUOUS("output tensor", output);
        _count = 0;
        return {coeffs(), std::move(output)};
    }
    // }}}

    // Utilities for Accumulator {{{
    // Returns `max{Re[x] for x in xs}`
    inline auto max_real(gsl::span<std::complex<float> const> xs) -> float
    {
        auto        chunks = xs.size() / 8;
        auto        rest   = xs.size() % 8;
        auto const* data   = xs.data();
        auto const  load   = [](std::complex<float> const* p) -> vcl::Vec8f {
            auto const x = _mm256_load_ps(reinterpret_cast<float const*>(p));
            auto const y =
                _mm256_load_ps(reinterpret_cast<float const*>(p + 4));
            return _mm256_shuffle_ps(x, y, _MM_SHUFFLE(2, 0, 2, 0));
        };
        auto scalar_max = -std::numeric_limits<float>::infinity();
        if (chunks != 0) {
            auto max = load(data);
            for (--chunks, data += 8; chunks != 0; --chunks, data += 8) {
                max = vcl::maximum(max, load(data));
            }
            scalar_max = vcl::horizontal_max(max);
        }
        for (; rest != 0; --rest, ++data) {
            scalar_max = std::max(scalar_max, data->real());
        }

        auto const expected = [xs]() {
            auto it =
                std::max_element(xs.begin(), xs.end(), [](auto a, auto b) {
                    TCM_ASSERT(!std::isnan(a.real()) && !std::isnan(b.real()),
                               "");
                    return a.real() < b.real();
                });
            return it != xs.end() ? it->real()
                                  : -std::numeric_limits<float>::infinity();
        }();
        TCM_ASSERT(expected == scalar_max,
                   noexcept_format("max_real is broken: {} != {}", expected,
                                   scalar_max));
        return scalar_max;
    }

    // Unconjugated dot product
    inline auto dotu(gsl::span<complex_type const>        xs,
                     gsl::span<std::complex<float> const> ys) TCM_NOEXCEPT
        -> complex_type
    {
        using std::begin, std::end;
        TCM_ASSERT(xs.size() == ys.size(), "dimensions don't match");
        return std::inner_product(
            begin(xs), end(xs), begin(ys), complex_type{0, 0}, std::plus<>{},
            [](auto const& x, auto const& y) {
                return std::conj(x) * static_cast<complex_type>(y);
            });
    }

    /// Computes xs <- exp(xs - k)
    inline auto exp_min_const(gsl::span<std::complex<float>> xs, float const k)
        -> void
    {
#if 0
    auto const factor = vcl::Vec8f{k, 0.0f, k, 0.0f, k, 0.0f, k, 0.0f};
    auto       chunks = xs.size() / 4;
    auto       rest   = xs.size() % 4;
    auto*      data   = reinterpret_cast<float*>(xs.data());

    vcl::Vec8f x;
    for (; chunks != 0; --chunks, data += 8) {
        x.load_a(data);
        x = __svml_cexpf8(x - factor);
        x.store_a(data);
    }
    x.load_partial(2 * rest, data);
    x = __svml_cexpf8(x - factor);
    x.store_partial(2 * rest, data);
#else
        std::for_each(xs.begin(), xs.end(),
                      [k](auto& x) { x = std::exp(x - k); });
#endif
    }
    // }}}

    // Accumulator {{{
    namespace detail {
        template <class State> struct Accumulator {
          private:
            struct state_type {
              public:
                complex_type sum;
                real_type    scale;

              public:
                constexpr state_type() noexcept : sum{0, 0}, scale{0} {}
                constexpr state_type(real_type const k) noexcept
                    : sum{0, 0}, scale{k}
                {}

                constexpr state_type(state_type const&) noexcept = default;
                constexpr state_type(state_type&&) noexcept      = default;
                constexpr auto operator=(state_type const&) noexcept
                    -> state_type&     = default;
                constexpr auto operator=(state_type&&) noexcept
                    -> state_type&     = default;

                auto rescale(real_type const k) TCM_NOEXCEPT -> void
                {
                    TCM_ASSERT(k >= scale, "there is no point in downscaling");
                    sum *= std::exp(scale - k);
                    scale = k;
                }
            };

            class output_type {
                using buffer_type = gsl::span<std::complex<float>>;
                buffer_type _out;
                size_t      _index;

              public:
                constexpr output_type(buffer_type buffer) noexcept
                    : _out{buffer}, _index{0}
                {}

                constexpr output_type(output_type const&) noexcept = default;
                constexpr output_type(output_type&&) noexcept      = default;
                constexpr auto operator=(output_type const&) noexcept
                    -> output_type&    = default;
                constexpr auto operator=(output_type&&) noexcept
                    -> output_type&    = default;

                auto operator()(state_type& state) TCM_NOEXCEPT -> void
                {
                    TCM_ASSERT(_index < _out.size(), "output buffer is full");
                    _out[_index++] = static_cast<buffer_type::value_type>(
                        state.scale + std::log(state.sum));
                    state.sum = complex_type{0, 0};
                }
            };

            ForwardPropagator<State> _forward;
            output_type              _store;
            state_type               _state;
            std::vector<size_t>      _counts;

          public:
            Accumulator(unsigned                       batch_size,
                        gsl::span<std::complex<float>> out);

            auto reset(gsl::span<std::complex<float>> out) TCM_NOEXCEPT -> void;

            template <class ForwardFn, class Iterator>
            auto operator()(ForwardFn fn, Iterator first, Iterator last)
                -> void;

            template <class ForwardFn> auto finalize(ForwardFn fn) -> void;

          private:
            template <class ForwardFn>
            TCM_NOINLINE auto process_batch(ForwardFn fn) -> void;
        }; // }}}

        // Accumulator IMPLEMENTATION {{{
        template <class State>
        Accumulator<State>::Accumulator(
            unsigned const batch_size, gsl::span<std::complex<float>> const out)
            : _forward{batch_size}, _store{out}, _state{}, _counts{}
        {
            _counts.reserve(_forward.batch_size());
        }

        template <class State>
        auto Accumulator<State>::reset(gsl::span<std::complex<float>> out)
            TCM_NOEXCEPT -> void
        {
            _forward.clear();
            _store = output_type{out};
            _state = state_type{};
            _counts.clear();
        }

        template <class State>
        template <class ForwardFn, class Iterator>
        auto Accumulator<State>::operator()(ForwardFn fn, Iterator first,
                                            Iterator last) -> void
        {
            TCM_ASSERT(!_forward.full(), "precondition violated");
            _counts.push_back(0);
            for (; first != last; ++first) {
                _forward.push(first->first, first->second);
                ++_counts.back();
                if (_forward.full()) { process_batch(fn); }
            }
            TCM_ASSERT(!_forward.full(), "postcondition violated");
        }

        template <class State>
        template <class ForwardFn>
        auto Accumulator<State>::finalize(ForwardFn fn) -> void
        {
            TCM_ASSERT(!_forward.full(), "precondition violated");
            if (_forward.empty()) {
                _store(_state);
                return;
            }
            _counts.push_back(0);
            _forward.fill();
            process_batch(std::move(fn));
            TCM_ASSERT(_forward.empty(), "postcondition violated");
        }

        template <class State>
        template <class ForwardFn>
        auto Accumulator<State>::process_batch(ForwardFn fn) -> void
        {
            using std::swap;
            TCM_ASSERT(!_counts.empty(), "precondition violated");
            TCM_ASSERT(_forward.full(), "precondition violated");
            auto const result = _forward.run(std::move(fn));
            auto const coeff  = result.first;
            auto const y      = gsl::span<std::complex<float>>{
                reinterpret_cast<std::complex<float>*>(
                    result.second.data_ptr()),
                result.first.size()};

            {
                auto const k = max_real(y);
                TCM_CHECK(!std::isnan(k), std::runtime_error,
                          "NaN encountered in neural network output");
                if (k >= _state.scale) { _state.rescale(k); }
            }
            exp_min_const(y, _state.scale);

            auto offset = size_t{0};
            for (auto j = size_t{0}; j < _counts.size() - 1; offset +=
                                                             _counts[j++]) {
                _state.sum += dotu(coeff.subspan(offset, _counts[j]),
                                   y.subspan(offset, _counts[j]));
                _store(_state);
            }
            _state.sum += dotu(coeff.subspan(offset), y.subspan(offset));

            // Throw away all _counts except for the last which we set to 0
            _counts.resize(1);
            _counts[0] = 0;
            TCM_ASSERT(_forward.empty(), "postcondition violated");
        }
    } // namespace detail
    // }}}

    // PolynomialState {{{
    template <class State, class Hamiltonian> class PolynomialState {
        using _Polynomial = Polynomial<State, Hamiltonian>;
      public:
        using StateT       = State;
        using HamiltonianT = Hamiltonian;

      private:
        detail::Accumulator<State>   _accum;
        ForwardT<State>              _fn;
        std::shared_ptr<_Polynomial> _poly;

      public:
        PolynomialState(std::shared_ptr<_Polynomial> polynomial,
                        ForwardT<State> fn, unsigned batch_size);

        PolynomialState(PolynomialState const&)     = default;
        PolynomialState(PolynomialState&&) noexcept = default;
        auto operator=(PolynomialState const&) -> PolynomialState& = default;
        auto operator           =(PolynomialState&&) noexcept
            -> PolynomialState& = default;

        auto operator()(gsl::span<State const> spins) -> torch::Tensor;
    }; // }}}

    // PolynomialState IMPLEMENTATION {{{
    template <class State, class Hamiltonian>
    PolynomialState<State, Hamiltonian>::PolynomialState(
        std::shared_ptr<_Polynomial> polynomial, ForwardT<State> fn,
        unsigned batch_size)
        : _accum{batch_size, {}}
        , _fn{std::move(fn)}
        , _poly{std::move(polynomial)}
    {}

    template <class State, class Hamiltonian>
    auto PolynomialState<State, Hamiltonian>::operator()(
        gsl::span<State const> spins) -> torch::Tensor
    {
        auto out = ::TCM_NAMESPACE::detail::make_tensor<float>(spins.size(), 2);
        _accum.reset(gsl::span<std::complex<float>>{
            reinterpret_cast<std::complex<float>*>(out.data_ptr()),
            spins.size()});

        for (auto const& s : spins) {
            auto const& state = (*_poly)(real_type{1}, s);
            _accum(std::cref(_fn), state.begin(), state.end());
        }
        _accum.finalize(std::cref(_fn));
        return out;
    }
    // }}}

} // namespace detail
} // namespace v2

TCM_NAMESPACE_END
