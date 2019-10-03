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

#include "polynomial.hpp"

TCM_NAMESPACE_BEGIN

namespace detail {
struct ForwardPropagator {
  private:
    aligned_vector<SpinVector>          _spins;
    aligned_vector<std::complex<float>> _coeffs;
    size_t                              _count;
    size_t                              _batch_size;

    inline auto coeffs() const noexcept -> gsl::span<std::complex<float> const>;

  public:
    explicit ForwardPropagator(std::pair<size_t, size_t> input_shape);

    /*constexpr*/ auto clear() noexcept -> void;
    constexpr auto batch_size() const noexcept -> size_t;
    constexpr auto full() const noexcept -> bool;
    constexpr auto empty() const noexcept -> bool;
    inline auto    push(SpinVector const&   spin,
                        std::complex<float> coeff) TCM_NOEXCEPT -> void;
    inline auto    fill() TCM_NOEXCEPT -> void;

    template <class ForwardFn>
    inline auto run(ForwardFn&& fn)
        -> std::pair<gsl::span<std::complex<float> const>, torch::Tensor>;
};

struct Accumulator {
  private:
    struct state_type {
      public:
        std::complex<float> sum;
        float               scale;

      public:
        constexpr state_type() noexcept : sum{0.0f}, scale{0.0f} {}
        constexpr state_type(float const k) noexcept : sum{0.0f}, scale{k} {}

        constexpr state_type(state_type const&) noexcept = default;
        constexpr state_type(state_type&&) noexcept      = default;
        constexpr auto operator=(state_type const&) noexcept
            -> state_type&     = default;
        constexpr auto operator=(state_type&&) noexcept
            -> state_type&     = default;

        auto rescale(float const k) TCM_NOEXCEPT -> void
        {
            TCM_ASSERT(k >= scale, "there is no point in downscaling");
            sum *= std::exp(scale - k);
            scale = k;
        }
    };

    class output_type {
        gsl::span<std::complex<float>> _out;
        size_t                         _index;

      public:
        constexpr output_type(gsl::span<std::complex<float>> buffer) noexcept
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
            _out[_index++] = state.scale + std::log(state.sum);
            state.sum      = 0.0f;
        }
    };

    ForwardPropagator   _forward;
    output_type         _store;
    state_type          _state;
    std::vector<size_t> _counts;

  public:
    Accumulator(std::pair<size_t, size_t> const input_shape,
                gsl::span<std::complex<float>>  out);

    inline auto reset(gsl::span<std::complex<float>> out) TCM_NOEXCEPT -> void;

    template <class ForwardFn, class Iterator>
    auto operator()(ForwardFn fn, Iterator first, Iterator last) -> void;

    template <class ForwardFn> auto finalize(ForwardFn fn) -> void;

  private:
    template <class ForwardFn> auto process_batch(ForwardFn fn) -> void;
};
} // namespace detail

class PolynomialStateV2 {
    std::shared_ptr<Polynomial> _poly;
    ForwardT                    _fn;
    detail::Accumulator         _accum;

  public:
    PolynomialStateV2(std::shared_ptr<Polynomial> polynomial, ForwardT fn,
                      std::pair<size_t, size_t> input_shape);

    PolynomialStateV2(PolynomialStateV2 const&)     = default;
    PolynomialStateV2(PolynomialStateV2&&) noexcept = default;
    auto operator=(PolynomialStateV2 const&) -> PolynomialStateV2& = default;
    auto operator             =(PolynomialStateV2&&) noexcept
        -> PolynomialStateV2& = default;

    auto operator()(gsl::span<SpinVector const> spins) -> torch::Tensor;
};

TCM_NAMESPACE_END
