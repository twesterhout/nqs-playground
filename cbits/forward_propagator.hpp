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

#include <torch/types.h>

#include <future>
#include <queue>
#include <vector>

TCM_NAMESPACE_BEGIN

namespace detail {

struct TaskBuilder {
  public:
    struct Task {
        v2::ForwardT          psi;
        torch::Tensor         spins;
        torch::Tensor         coeffs;
        std::vector<uint64_t> counts;
        bool                  complete;

        auto operator()() const
            -> std::tuple<float, bool, std::vector<std::complex<float>>>;
    };

  private:
    uint64_t             _i;
    uint64_t*            _spins_ptr;
    std::complex<float>* _coeffs_ptr;
    uint64_t             _batch_size;
    Task                 _next_task;

    auto prepare(v2::ForwardT fn) -> void;

  public:
    TaskBuilder(v2::ForwardT psi, uint64_t batch_size);

    auto start()
    {
        TCM_ASSERT(!full(), "buffer is full");
        if (!empty()) { _next_task.counts.push_back(0); }
        _next_task.complete = false;
    }

    template <class R,
              class = std::enable_if_t<std::is_same<R, float>::value
                                       || std::is_same<R, double>::value>>
    auto add(uint64_t const spin, std::complex<R> const coeff)
    {
        TCM_ASSERT(!full(), "buffer is full");
        _spins_ptr[_i]  = spin;
        _coeffs_ptr[_i] = static_cast<std::complex<float>>(coeff);
        ++_next_task.counts.back();
        ++_i;
    }

    auto add_junk() -> void;

    auto finish() -> void
    {
        _next_task.complete = true;
        if (empty()) { _next_task.counts.push_back(0); }
    }

    auto submit(bool prepare_next = true) -> Task;

    constexpr auto empty() const noexcept -> bool
    {
        TCM_ASSERT(_i <= _batch_size, "precondition violated");
        return _i == 0;
    }

    constexpr auto full() const noexcept -> bool
    {
        TCM_ASSERT(_i <= _batch_size, "precondition violated");
        return _i == _batch_size;
    }
};

class Accumulator {

  private:
    struct state_type {
      private:
        std::complex<float> _sum;
        float               _scale;

      public:
        constexpr state_type(std::complex<float> const sum   = {0.0f, 0.0f},
                             float const               scale = 0.0f) noexcept
            : _sum{sum}, _scale{scale}
        {}
        constexpr state_type(state_type const&) noexcept = default;
        constexpr state_type(state_type&&) noexcept      = default;
        constexpr state_type& operator=(state_type const&) noexcept = default;
        constexpr state_type& operator=(state_type&&) noexcept = default;

        auto operator+=(state_type const& other) -> state_type&
        {
            if (other._scale >= _scale) {
                _sum *= std::exp(_scale - other._scale);
                _sum += other._sum;
                _scale = other._scale;
            }
            else {
                _sum += other._sum * std::exp(other._scale - _scale);
            }
            return *this;
        }

        auto get_log() const noexcept -> std::complex<float>
        {
            return _scale + std::log(_sum);
        }
    };

    class output_type {
        using buffer_type = gsl::span<std::complex<float>>;
        buffer_type _out;
        size_t      _index;

      public:
        constexpr output_type(buffer_type const buffer) noexcept
            : _out{buffer}, _index{0}
        {}
        constexpr output_type(output_type const&) noexcept = default;
        constexpr output_type(output_type&&) noexcept      = default;
        constexpr output_type& operator=(output_type const&) noexcept = default;
        constexpr output_type& operator=(output_type&&) noexcept = default;

        auto operator()(buffer_type::value_type const value) TCM_NOEXCEPT
            -> void
        {
            TCM_ASSERT(_index < _out.size(), "output buffer is full");
            _out[_index++] = value;
        }
    };

    using result_type = std::invoke_result_t<TaskBuilder::Task>;
    using future_type = std::future<result_type>;

    TaskBuilder             _builder;
    output_type             _store;
    state_type              _state;
    std::queue<future_type> _futures;

  public:
    Accumulator(v2::ForwardT fn, gsl::span<std::complex<float>> out,
                unsigned batch_size);

    auto reset(gsl::span<std::complex<float>> out) -> void;

    template <class Iterator>
    auto operator()(Iterator first, Iterator last) -> void;

    template <class ForEach> auto operator()(ForEach&& for_each) -> void;

    auto finalize() -> void;

  private:
    inline auto drain_if_needed() -> void;
    auto        drain(unsigned count) -> void;
    auto        process(result_type result) -> void;
};

auto Accumulator::drain_if_needed() -> void
{
    constexpr auto soft_max = 32U;
    constexpr auto hard_max = 64U;
    TCM_ASSERT(_futures.size() <= hard_max, "");
    if (_futures.size() == hard_max) { drain(hard_max - soft_max); }
}

template <class Iterator>
auto Accumulator::operator()(Iterator first, Iterator last) -> void
{
    TCM_ASSERT(!_builder.full(), "precondition violated");
    _builder.start();
    for (; first != last; ++first) {
        _builder.add(first->first, first->second);
        if (_builder.full()) {
            drain_if_needed();
            _futures.push(std::async(_builder.submit()));
        }
    }
    _builder.finish();
    TCM_ASSERT(!_builder.full(), "postcondition violated");
}

template <class ForEach>
auto Accumulator::operator()(ForEach&& for_each) -> void
{
    TCM_ASSERT(!_builder.full(), "precondition violated");
    _builder.start();
    std::forward<ForEach>(for_each)([this](auto const spin, auto const coeff) {
        _builder.add(spin, coeff);
        if (_builder.full()) {
            drain_if_needed();
            _futures.push(std::async(_builder.submit()));
        }
    });
    _builder.finish();
    TCM_ASSERT(!_builder.full(), "postcondition violated");
}

} // namespace detail

TCM_NAMESPACE_END
