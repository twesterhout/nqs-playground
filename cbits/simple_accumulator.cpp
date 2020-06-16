// Copyright (c) 2020, Tom Westerhout
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

#include "simple_accumulator.hpp"
#include "dotu.hpp"
#include "errors.hpp"
#include "parallel.hpp"
#include "tensor_info.hpp"

#include <future>
#include <queue>

TCM_NAMESPACE_BEGIN

struct Task {
    v2::ForwardT          psi;
    torch::Tensor         spins;
    torch::Tensor         coeffs;
    std::vector<uint64_t> counts;
    unsigned              size;
    bool                  complete;

    Task(v2::ForwardT forward, uint64_t batch_size, bool pin_memory);
    Task(Task const&) = delete;
    Task(Task&&)      = default;
    auto operator=(Task const&) -> Task& = delete;
    auto operator=(Task &&) -> Task& = delete;

    auto operator()() const
        -> std::tuple<float, bool, std::vector<std::complex<float>>>;

    auto add(gsl::span<bits512 const>      spins,
             gsl::span<complex_type const> coeffs) -> void;
};

TCM_NOINLINE Task::Task(v2::ForwardT forward, uint64_t const batch_size,
                        bool const pin_memory)
    : psi{std::move(forward)}
    , spins{}
    , coeffs{}
    , counts{}
    , size{0}
    , complete{true}
{
    auto const common_options = torch::TensorOptions{}
                                    .device(c10::DeviceType::CPU)
                                    .pinned_memory(pin_memory);
    spins  = torch::empty({static_cast<int64_t>(batch_size), 8L},
                         common_options.dtype(torch::kInt64));
    coeffs = torch::empty({static_cast<int64_t>(batch_size), 2L},
                          common_options.dtype(torch::kFloat32));
}

TCM_NOINLINE auto Task::operator()() const
    -> std::tuple<float, bool, std::vector<std::complex<float>>>
{
    torch::NoGradGuard no_grad;

    auto       output     = this->psi(spins);
    auto const batch_size = this->spins.size(0);
    TCM_CHECK_SHAPE("output tensor", output, {batch_size, 2});
    auto real = torch::narrow(output, /*dim=*/1, /*start=*/0, /*length=*/1);
    auto imag = torch::narrow(output, /*dim=*/1, /*start=*/1, /*length=*/1);

    auto const scale = torch::max(real).item<float>();
    TCM_CHECK(!std::isnan(scale), std::runtime_error,
              "NaN encountered in neural network output");

    // The following computes complex-valued exp of (output - scale) in-place
    real -= scale;
    real.exp_();
    auto cos = torch::cos(imag);
    torch::sin_(imag);
    imag *= real;
    real *= cos;

    std::vector<std::complex<float>> results;
    results.reserve(this->counts.size());

    auto offset = int64_t{0};
    for (auto j = 0U; j < this->counts.size();
         offset += static_cast<int64_t>(this->counts[j++])) {
        auto const slice = [this, j, offset](auto const& t) {
            return torch::narrow(
                t, /*dim=*/0, /*start=*/offset,
                /*length=*/static_cast<int64_t>(this->counts[j]));
        };
        auto const r = this->counts[j] > 0 ? static_cast<std::complex<float>>(
                           dotu(slice(this->coeffs), slice(output)))
                                           : std::complex<float>{0.0f, 0.0f};
        results.push_back(r);
    }

    return {scale, complete, std::move(results)};
}

TCM_NOINLINE auto Task::add(gsl::span<bits512 const>      more_spins,
                            gsl::span<complex_type const> more_coeffs) -> void
{
    TCM_ASSERT(size
                   == std::accumulate(std::begin(counts), std::end(counts),
                                      uint64_t{0}, std::plus<void>{}),
               "");
    TCM_CHECK(more_spins.size() == more_coeffs.size(), std::invalid_argument,
              fmt::format("spins and coeffs have different lengths: {} != {}",
                          more_spins.size(), more_coeffs.size()));
    auto const batch_size = static_cast<uint64_t>(spins.size(0));
    TCM_CHECK(
        size + more_spins.size() <= batch_size, std::invalid_argument,
        fmt::format("buffer overflow: trying to add {} more elements to a "
                    "buffer of size {} which already contains {} elements",
                    more_spins.size(), batch_size, size));

    std::memcpy(reinterpret_cast<bits512*>(spins.data_ptr<int64_t>()) + size,
                more_spins.data(), more_spins.size() * sizeof(bits512));
    std::transform(
        more_coeffs.data(), more_coeffs.data() + more_coeffs.size(),
        reinterpret_cast<std::complex<float>*>(coeffs.data_ptr<float>()) + size,
        [](auto const& x) { return static_cast<std::complex<float>>(x); });
    // std::memcpy(
    //     reinterpret_cast<std::complex<float>*>(coeffs.data_ptr<float>()) + size,
    //     more_coeffs.data(), more_coeffs.size() * sizeof(std::complex<float>));
    counts.push_back(more_spins.size());
    size += more_spins.size();
}

class Writer {
    struct state_type {
      private:
        std::complex<double> _sum;
        double               _scale;

      public:
        constexpr state_type(std::complex<double> const sum   = {0.0f, 0.0f},
                             double const               scale = 0.0f) noexcept
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

        auto get_log() const noexcept
        {
            return static_cast<std::complex<float>>(_scale + std::log(_sum));
        }
    };

    aligned_vector<std::complex<float>> _buffer;
    std::optional<state_type>           _partial;

  public:
    Writer() noexcept : _buffer{}, _partial{std::nullopt} {}
    Writer(Writer const&)     = delete;
    Writer(Writer&&) noexcept = default;
    Writer& operator=(Writer const&) = delete;
    Writer& operator=(Writer&&) noexcept = default;

    auto reset() -> aligned_vector<std::complex<float>>;

    auto operator()(std::invoke_result_t<Task> const& result) -> void;
};

TCM_NOINLINE auto Writer::reset() -> aligned_vector<std::complex<float>>
{
    TCM_CHECK(!_partial.has_value(), std::runtime_error,
              "precondition violated");
    aligned_vector<std::complex<float>> temp;
    std::swap(temp, _buffer);
    return temp;
}

TCM_NOINLINE auto Writer::operator()(std::invoke_result_t<Task> const& result)
    -> void
{
    auto const [scale, complete, values] = std::move(result);
    TCM_CHECK(!values.empty(), std::runtime_error, "precondition violated");
    auto i = uint64_t{0};
    // First value
    if (_partial.has_value()) {
        (*_partial) += state_type{values[i], scale};
        ++i;
        // If there are more values, we know that `_partial` is complete.
        // Alternatively, if we're processing the last value, but
        // `complete==true`, then `_partial` is also complete.
        if (i != values.size() || complete) {
            _buffer.push_back(_partial->get_log());
            _partial.reset();
        }
        if (i == values.size()) { return; }
    }
    for (; i < values.size() - 1; ++i) {
        _buffer.push_back(state_type{values[i], scale}.get_log());
    }
    if (complete) { _buffer.push_back(state_type{values[i], scale}.get_log()); }
    else {
        _partial.emplace(values[i], scale);
    }
}

class Accumulator {
  private:
    using result_type = std::invoke_result_t<Task>;
    using future_type = std::future<result_type>;
    // using async_type  = stdext::inplace_function<auto(Task &&)->future_type>;

    v2::ForwardT            _fn;
    unsigned                _batch_size;
    c10::Device const       _device;
    std::optional<Task>     _partial_task;
    std::queue<future_type> _futures;
    Writer                  _writer;

  public:
    Accumulator(v2::ForwardT fn, unsigned batch_size, c10::Device device);

    auto reset() -> aligned_vector<std::complex<float>>;

    auto operator()(gsl::span<bits512 const>      spins,
                    gsl::span<complex_type const> coeffs) -> void;

  private:
    auto submit(Task task) -> void;
    auto submit_final() -> void;
    auto drain_if_needed() -> void;
    auto drain(unsigned count) -> void;
};

TCM_NOINLINE Accumulator::Accumulator(v2::ForwardT fn, unsigned batch_size,
                                      c10::Device device)
    : _fn{std::move(fn)}
    , _batch_size{batch_size}
    , _device{device}
    , _partial_task{std::nullopt}
    , _futures{}
    , _writer{}
{
    TCM_CHECK(_fn, std::invalid_argument, "forward function must not be None");
    TCM_CHECK(batch_size > 0, std::invalid_argument,
              "batch_size must be positive");
}

TCM_NOINLINE auto Accumulator::reset() -> aligned_vector<std::complex<float>>
{
    submit_final();
    drain(_futures.size());
    return _writer.reset();
}

TCM_NOINLINE auto Accumulator::operator()(gsl::span<bits512 const>      spins,
                                          gsl::span<complex_type const> coeffs)
    -> void
{
    if (spins.empty()) { return; }
    if (!_partial_task.has_value()) {
        _partial_task.emplace(_fn, _batch_size, !_device.is_cpu());
    }
    auto const count =
        std::min<uint64_t>(_batch_size - _partial_task->size, spins.size());
    _partial_task->add(spins.subspan(0, count), coeffs.subspan(0, count));
    _partial_task->complete = count == spins.size();
    if (_partial_task->size == _batch_size) {
        submit(*std::move(_partial_task));
        _partial_task.reset();
    }
    operator()(spins.subspan(count), coeffs.subspan(count));
}

TCM_NOINLINE auto Accumulator::submit(Task task) -> void
{
    if (!_device.is_cpu()) {
        task.spins  = task.spins.to(task.spins.options().device(_device),
                                   /*non_blocking=*/true, /*copy=*/false);
        task.coeffs = task.coeffs.to(task.coeffs.options().device(_device),
                                     /*non_blocking=*/true, /*copy=*/false);
    }
    drain_if_needed();
    auto future = detail::global_thread_pool().enqueue(std::move(task));
    _futures.push(std::move(future));
}

TCM_NOINLINE auto Accumulator::submit_final() -> void
{
    if (_partial_task.has_value()) {
        TCM_CHECK(_partial_task->complete, std::runtime_error,
                  "precondition violated");
        _partial_task->spins = _partial_task->spins.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/_partial_task->size);
        _partial_task->coeffs = _partial_task->coeffs.narrow(
            /*dim=*/0, /*start=*/0, /*length=*/_partial_task->size);
        submit(*std::move(_partial_task));
        _partial_task.reset();
    }
}

auto Accumulator::drain_if_needed() -> void
{
    constexpr auto soft_max = 32U;
    constexpr auto hard_max = 64U;
    TCM_ASSERT(_futures.size() <= hard_max, "");
    if (_futures.size() == hard_max) { drain(hard_max - soft_max); }
}

TCM_NOINLINE auto Accumulator::drain(unsigned const count) -> void
{
    TCM_ASSERT(count <= _futures.size(), "");
    for (auto i = 0U; i < count; ++i) {
        auto future = std::move(_futures.front());
        _futures.pop();
        try {
            auto value = future.get();
            _writer(value);
        }
        catch (...) {
            // Drain the rest of the queue since we don't know how to recover
            // anyway. We ignore all remaining exceptions and rethrow the first
            // one.
            auto exception_ptr = std::current_exception();
            for (; !_futures.empty(); _futures.pop()) {
                try {
                    _futures.front().get();
                }
                catch (...) {
                }
            }
            std::rethrow_exception(exception_ptr);
        }
    }
}

template <class Allocator>
auto vector_to_tensor(std::vector<std::complex<float>, Allocator> vector)
    -> torch::Tensor
{
    using VectorT = std::vector<std::complex<float>, Allocator>;
    auto data     = vector.data();
    auto size     = static_cast<int64_t>(vector.size());
    auto deleter  = [original =
                        std::make_unique<VectorT>(std::move(vector)).release()](
                       void* p) mutable {
        if (p != original->data()) {
            detail::assert_fail(
                "false", __FILE__, __LINE__, TCM_CURRENT_FUNCTION,
                fmt::format("Trying to delete wrong pointer: {} != {}", p,
                            static_cast<void*>(original->data())));
        }
        std::default_delete<std::vector<std::complex<float>, Allocator>>{}(
            original);
    };
    return torch::from_blob(data, {size, 2L}, std::move(deleter),
                            torch::TensorOptions{}.dtype(torch::kFloat32));
}

namespace {
auto apply_impl(torch::Tensor spins, OperatorT op, v2::ForwardT psi,
                uint64_t const max_states, uint32_t const batch_size,
                c10::Device const device) -> torch::Tensor
{
    torch::NoGradGuard no_grad;

    auto const spins_info = obtain_tensor_info<bits512 const>(spins);
    aligned_vector<bits512>      temp_spins(max_states);
    aligned_vector<complex_type> temp_coeffs(max_states);

    Accumulator acc{std::move(psi), batch_size, device};
    for (auto i = int64_t{0}; i < spins_info.size(); ++i) {
        auto const written = op(spins_info[i], 1.0, temp_spins, temp_coeffs);
        acc(gsl::span<bits512 const>{temp_spins.data(), written},
            gsl::span<complex_type const>{temp_coeffs.data(), written});
    }
    return vector_to_tensor(acc.reset());
}
} // namespace

TCM_EXPORT auto apply(torch::Tensor spins, OperatorT op, v2::ForwardT psi,
                      uint64_t max_states, uint32_t batch_size,
                      int32_t num_threads) -> torch::Tensor
{
    torch::NoGradGuard no_grad;
    auto const         device = spins.device();
    if (!device.is_cpu()) { spins = spins.cpu(); }

    auto const chunk_size =
        std::max(128L, spins.size(0) / (10 * omp_get_max_threads()));
    auto chunks = torch::split(spins, chunk_size, /*dim=*/0);
    std::vector<torch::Tensor> outputs(chunks.size());

    if (num_threads <= 0) { num_threads = omp_get_max_threads(); }
    std::atomic_flag   error_flag    = ATOMIC_FLAG_INIT;
    std::exception_ptr exception_ptr = nullptr;
#pragma omp parallel for num_threads(num_threads)                              \
    schedule(dynamic, 1) default(none)                                         \
        shared(error_flag, exception_ptr, chunks, outputs, op, psi)            \
            firstprivate(max_states, batch_size, device)
    for (auto i = uint64_t{0}; i < chunks.size(); ++i) {
        try {
            outputs[i] =
                apply_impl(chunks[i], op, psi, max_states, batch_size, device);
        }
        catch (...) {
            if (!error_flag.test_and_set()) {
                exception_ptr = std::current_exception();
            }
        }
    };
    if (exception_ptr) { std::rethrow_exception(exception_ptr); }

    auto out = torch::cat(outputs);
    if (!device.is_cpu()) { out = out.to(out.options().device(device)); }
    return out;
}

TCM_NAMESPACE_END
