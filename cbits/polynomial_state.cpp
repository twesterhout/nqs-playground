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

#include "polynomial_state.hpp"

#include <boost/align/is_aligned.hpp>
#include <mkl_cblas.h>
#include <torch/extension.h>
#include <vectorclass/version2/vectorclass.h>

#if defined(TCM_GCC)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wundef"
#    pragma GCC diagnostic ignored "-Wold-style-cast"
#elif defined(TCM_CLANG)
#    pragma clang diagnostic push
#endif
#include <pybind11/stl.h>
#if defined(TCM_GCC)
#    pragma GCC diagnostic pop
#elif defined(TCM_CLANG)
#    pragma clang diagnostic pop
#endif

extern "C" {
__m256 __svml_cexpf8(__m256);
}

TCM_NAMESPACE_BEGIN

namespace detail {

ForwardPropagator::ForwardPropagator(std::pair<size_t, size_t> input_shape)
    : _spins{}, _coeffs{}, _count{0}, _batch_size{input_shape.first}
{
    TCM_CHECK(
        input_shape.first > 0, std::invalid_argument,
        fmt::format(
            "invalid input_shape: [{}, {}]; expected a positive batch size",
            input_shape.first, input_shape.second));
    TCM_CHECK(input_shape.second > 0, std::invalid_argument,
              fmt::format("invalid input_shape: [{}, {}]; expected a "
                          "positive system size",
                          input_shape.first, input_shape.second));
    _spins.resize(input_shape.first, SpinVector{});
    constexpr auto NaN = std::numeric_limits<float>::quiet_NaN();
    _coeffs.resize(input_shape.first, std::complex<float>{NaN, NaN});
}

auto ForwardPropagator::coeffs() const noexcept
    -> gsl::span<std::complex<float> const>
{
    TCM_ASSERT(_coeffs.size() == batch_size(),
               "ForwardPropagator is in an invalid state");
    return _coeffs;
}

auto ForwardPropagator::clear() noexcept -> void
{
    using std::begin, std::end;
    std::fill(begin(_spins), end(_spins), SpinVector{});
    constexpr auto NaN = std::numeric_limits<float>::quiet_NaN();
    std::fill(begin(_coeffs), end(_coeffs), std::complex<float>{NaN, NaN});
    _count = 0;
}

constexpr auto ForwardPropagator::batch_size() const noexcept -> size_t
{
    return _batch_size;
}

constexpr auto ForwardPropagator::full() const noexcept -> bool
{
    TCM_ASSERT(_count <= _batch_size, "precondition violated");
    return _count == _batch_size;
}

constexpr auto ForwardPropagator::empty() const noexcept -> bool
{
    TCM_ASSERT(_count <= _batch_size, "precondition violated");
    return _count == 0;
}

auto ForwardPropagator::push(SpinVector const&   spin,
                             std::complex<float> coeff) TCM_NOEXCEPT -> void
{
    TCM_ASSERT(!full(), "buffer is full");
    _spins[_count]  = spin;
    _coeffs[_count] = coeff;
    ++_count;
}

auto ForwardPropagator::fill() TCM_NOEXCEPT -> void
{
    TCM_ASSERT(!empty(), "precondition violated");
    auto spin = _spins[_count - 1];
    for (; _count < _batch_size; ++_count) {
        _spins[_count]  = spin;
        _coeffs[_count] = 0.0f;
    }
    TCM_ASSERT(full(), "postcondition violated");
}

template <class ForwardFn>
auto ForwardPropagator::run(ForwardFn&& fn)
    -> std::pair<gsl::span<std::complex<float> const>, torch::Tensor>
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

namespace {
auto max_real(gsl::span<std::complex<float> const> xs) -> float
{
    auto        chunks = xs.size() / 8;
    auto        rest   = xs.size() % 8;
    auto const* data   = xs.data();
    auto const  load   = [](std::complex<float> const* p) -> vcl::Vec8f {
        auto const x = _mm256_load_ps(reinterpret_cast<float const*>(p));
        auto const y = _mm256_load_ps(reinterpret_cast<float const*>(p + 4));
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
                return a.real() < b.real();
            });
        return it != xs.end() ? it->real()
                              : -std::numeric_limits<float>::infinity();
    }();
    TCM_ASSERT(
        expected == scalar_max,
        noexcept_format("max_real is broken: {} != {}", expected, scalar_max));
    return scalar_max;
}

auto dotu(gsl::span<std::complex<float> const> xs,
          gsl::span<std::complex<float> const> ys) TCM_NOEXCEPT
    -> std::complex<float>
{
    TCM_ASSERT(xs.size() == ys.size(), "dimensions don't match");
    std::complex<float> r;
    cblas_cdotu_sub(xs.size(), xs.data(), 1, ys.data(), 1, &r);
    return r;
}

/// Computes xs <- exp(xs - k)
auto exp_min_const(gsl::span<std::complex<float>> xs, float const k) -> void
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
    std::for_each(xs.begin(), xs.end(), [k](auto& x) { x = std::exp(x - k); });
#endif
}
} // namespace

Accumulator::Accumulator(std::pair<size_t, size_t> const input_shape,
                         gsl::span<std::complex<float>>  out)
    : _forward{input_shape}, _store{out}, _state{}, _counts{}
{
    _counts.reserve(_forward.batch_size());
}

auto Accumulator::reset(gsl::span<std::complex<float>> out) TCM_NOEXCEPT -> void
{
    _forward.clear();
    _store = output_type{out};
    _state = state_type{};
    _counts.clear();
}

template <class ForwardFn, class Iterator>
auto Accumulator::operator()(ForwardFn fn, Iterator first, Iterator last)
    -> void
{
    TCM_ASSERT(!_forward.full(), "precondition violated");
    _counts.push_back(0);
    for (; first != last; ++first) {
        _forward.push(first->first,
                      static_cast<std::complex<float>>(first->second));
        ++_counts.back();
        if (_forward.full()) {
            process_batch(fn);
            TCM_ASSERT(_forward.empty(), "");
        }
    }
    TCM_ASSERT(!_forward.full(), "postcondition violated");
}

template <class ForwardFn> auto Accumulator::finalize(ForwardFn fn) -> void
{
    TCM_ASSERT(!_forward.full(), "precondition violated");
    if (_forward.empty()) {
        _store(_state);
        return;
    }
    _counts.push_back(0);
    _forward.fill();
    process_batch(std::move(fn));
    TCM_ASSERT(_forward.empty(), "");
}

template <class ForwardFn> auto Accumulator::process_batch(ForwardFn fn) -> void
{
    using std::swap;
    TCM_ASSERT(!_counts.empty(), "precondition violated");
    TCM_ASSERT(_forward.full(), "precondition violated");
    auto const result = _forward.run(std::move(fn));
    auto const coeff  = result.first;
    auto const y      = gsl::span<std::complex<float>>{
        reinterpret_cast<std::complex<float>*>(result.second.data_ptr()),
        result.first.size()};

    {
        auto const k = max_real(y);
        TCM_CHECK(!std::isnan(k), std::runtime_error,
                  "NaN encountered in neural network output");
        if (k >= _state.scale) { _state.rescale(k); }
    }
    exp_min_const(y, _state.scale);

    auto offset = size_t{0};
    for (auto j = size_t{0}; j < _counts.size() - 1; offset += _counts[j++]) {
        _state.sum += dotu(coeff.subspan(offset, _counts[j]),
                           y.subspan(offset, _counts[j]));
        _store(_state);
    }
    _state.sum += dotu(coeff.subspan(offset), y.subspan(offset));

    // Throw away all _counts except for the last which we set to 0
    _counts.resize(1);
    _counts[0] = 0;
}

} // namespace detail

PolynomialStateV2::PolynomialStateV2(std::shared_ptr<Polynomial> polynomial,
                                     ForwardT                    fn,
                                     std::pair<size_t, size_t>   input_shape)
    : _poly{std::move(polynomial)}, _fn{std::move(fn)}, _accum{input_shape, {}}
{}

auto PolynomialStateV2::operator()(gsl::span<SpinVector const> spins)
    -> torch::Tensor
{
    auto out = detail::make_tensor<float>(spins.size(), 2);
    _accum.reset(gsl::span<std::complex<float>>{
        reinterpret_cast<std::complex<float>*>(out.data_ptr()), spins.size()});
    for (auto const& s : spins) {
        auto const& state = (*_poly)(1.0f, s);
        _accum(std::cref(_fn), state.begin(), state.end());
    }
    _accum.finalize(std::cref(_fn));
    return out;
}

TCM_NAMESPACE_END
