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

#include "accumulator.hpp"
#include "errors.hpp"
#include "parallel.hpp"
#include "tensor_info.hpp"
#include <gsl/gsl-lite.hpp>
#include <torch/types.h>
#include <future>
#include <queue>

TCM_NAMESPACE_BEGIN

struct Task {
    ForwardT              forward;
    torch::Tensor         spins;
    torch::Tensor         coeffs;
    std::vector<uint64_t> counts;
    c10::Device           device;
    bool                  complete;

    auto operator()() const -> std::tuple</*coeffs=*/torch::Tensor, /*complete=*/bool>;
};

namespace {
auto extract_scalar(char const* name, torch::Tensor const& x) -> double
{
    switch (x.scalar_type()) {
    case torch::kFloat32: return static_cast<double>(x.item<float>());
    case torch::kFloat64: return x.item<double>();
    default:
        TCM_ERROR(std::runtime_error,
                  fmt::format("{} has wrong type: {}; expected either Float or Double", name,
                              x.scalar_type()));
    }
}
} // namespace

TCM_NOINLINE auto Task::operator()() const -> std::tuple<torch::Tensor, bool>
{
    torch::NoGradGuard no_grad;

    auto const output = forward(spins.to(device));
    if (output.dim() > 1) { output.squeeze_(/*dim=*/1); }
    TCM_ASSERT(!torch::any(torch::isnan(output)).item<bool>(), "");
    auto const dtype = output.dtype();
    auto       real  = torch::real(output);
    auto const scale = torch::max(real).item();
    real -= scale;
    output.exp_();

    auto       results = torch::zeros({static_cast<int64_t>(counts.size())},
                                torch::TensorOptions{}.device(torch::kCPU).dtype(dtype));
    auto const cs      = coeffs.to(coeffs.options().device(device).dtype(dtype));
    TCM_ASSERT(!torch::any(torch::isnan(cs)).item<bool>(), "");

    auto offset = int64_t{0};
    for (auto j = 0U; j < counts.size(); offset += static_cast<int64_t>(counts[j++])) {
        auto const slice = [this, j, offset](auto const& t) {
            return torch::narrow(t, /*dim=*/0, /*start=*/offset,
                                 /*length=*/static_cast<int64_t>(counts[j]));
        };
        if (counts[j] > 0) {
            results[static_cast<int64_t>(j)] = torch::dot(slice(cs), slice(output));
        }
    }
    TCM_ASSERT(!torch::any(torch::isnan(results)).item<bool>(), "");

    results.log_();
    torch::real(results) += scale;
    return {std::move(results), complete};
}

class Buffer {
    ForwardT              _forward;
    torch::Tensor         _spins;
    torch::Tensor         _coeffs;
    std::vector<uint64_t> _counts;
    int64_t               _offset;

    // General information
    uint64_t _batch_size;        // How many spin configurations to include in a Task
    uint64_t _max_required_size; // Maximal number of spin configurations
                                 // which can be generated at a time
    c10::Device _device;         // On which device is _forward located

    auto make_spins() const -> torch::Tensor;
    auto make_coeffs() const -> torch::Tensor;

  public:
    using future_type = std::future<std::invoke_result_t<Task>>;

    Buffer(ForwardT forward, uint64_t max_required_size, uint64_t batch_size, c10::Device device);
    Buffer(Buffer const&) = delete;
    Buffer(Buffer&&)      = delete;
    auto operator=(Buffer const&) -> Buffer& = delete;
    auto operator=(Buffer&&) -> Buffer& = delete;

    auto submit(ls_bits512 const& x, OperatorT const& op)
        -> std::vector<std::future<std::invoke_result_t<Task>>>;
    auto submit_final() -> std::optional<std::future<std::invoke_result_t<Task>>>;
};

Buffer::Buffer(ForwardT forward, uint64_t max_required_size, uint64_t batch_size,
               c10::Device device)
    : _forward{std::move(forward)}
    , _counts{}
    , _offset{0}
    , _batch_size{batch_size}
    , _max_required_size{max_required_size}
    , _device{device}
{
    _spins  = make_spins();
    _coeffs = make_coeffs();
}

auto Buffer::make_spins() const -> torch::Tensor
{
    return torch::zeros({static_cast<int64_t>(_batch_size + _max_required_size), 8},
                        torch::TensorOptions{}.device(torch::kCPU).dtype(torch::kInt64));
}

auto Buffer::make_coeffs() const -> torch::Tensor
{
    return torch::empty({static_cast<int64_t>(_batch_size + _max_required_size)},
                        torch::TensorOptions{}.device(torch::kCPU).dtype(torch::kComplexDouble));
}

namespace {
auto split_counts(std::vector<uint64_t> counts, uint64_t const batch_size)
    -> std::tuple<std::vector<std::pair<std::vector<uint64_t>, bool>>,
                  std::optional<std::pair<std::vector<uint64_t>, bool>>>
{
    std::vector<std::pair<std::vector<uint64_t>, bool>> chunks;
    std::pair<std::vector<uint64_t>, bool>              chunk       = {{}, {}};
    auto                                                chunk_count = uint64_t{0};

    for (auto i = uint64_t{0}; i < counts.size(); ++i) {
        auto current_count = counts[i];
        if (current_count == 0) {
            chunk.first.push_back(0);
            chunk.second = true;
        }

        while (chunk_count + current_count >= batch_size) {
            chunk.first.push_back(batch_size - chunk_count);
            current_count -= batch_size - chunk_count;
            chunk.second = current_count == 0;
            chunks.push_back(std::move(chunk));
            // Reset chunk
            chunk       = {{}, {}};
            chunk_count = 0;
        }
        if (current_count != 0) {
            chunk.first.push_back(current_count);
            chunk.second = true;
            chunk_count += current_count;
        }
    }
    std::optional<std::pair<std::vector<uint64_t>, bool>> rest{std::nullopt};
    if (!chunk.first.empty()) { rest = std::move(chunk); }
    return {std::move(chunks), std::move(rest)};
}
} // namespace

auto Buffer::submit(ls_bits512 const& x, OperatorT const& op)
    -> std::vector<std::future<std::invoke_result_t<Task>>>
{
    TCM_ASSERT(_offset < static_cast<int64_t>(_batch_size), "");
    auto const written =
        op(x, 1.0,
           gsl::span{static_cast<ls_bits512*>(_spins.data_ptr()) + _offset, _max_required_size},
           gsl::span{static_cast<std::complex<double>*>(_coeffs.data_ptr()) + _offset,
                     _max_required_size});
    _offset += static_cast<int64_t>(written);
    _counts.push_back(written);

    std::vector<std::future<std::invoke_result_t<Task>>> futures;
    if (_offset < static_cast<int64_t>(_batch_size)) { return futures; }

    auto [chunks, rest] = split_counts(_counts, _batch_size);
    auto i              = uint64_t{0};
    for (; i < chunks.size(); ++i) {
        auto const start  = static_cast<int64_t>(i * _batch_size);
        auto const length = static_cast<int64_t>(_batch_size);
        TCM_ASSERT(start + length <= _batch_size + _max_required_size, "");
        auto future = async(Task{_forward,
                                 torch::narrow(_spins, /*dim=*/0, /*start=*/start,
                                               /*length=*/length),
                                 torch::narrow(_coeffs, /*dim=*/0, /*start=*/start,
                                               /*length=*/length),
                                 std::move(chunks[i].first), _device, chunks[i].second});
        futures.push_back(std::move(future));
    }
    if (rest.has_value()) {
        _counts = std::move(rest->first);
        _offset = static_cast<int64_t>(
            std::accumulate(std::begin(_counts), std::end(_counts), uint64_t{0}));
        auto const start = static_cast<int64_t>(i * _batch_size);
        TCM_ASSERT(start + _offset <= _batch_size + _max_required_size, "");

        auto new_spins = make_spins();
        torch::narrow(new_spins, /*dim=*/0, /*start=*/0,
                      /*length=*/_offset)
            .copy_(torch::narrow(_spins, /*dim=*/0, /*start=*/start,
                                 /*length=*/_offset));
        _spins = std::move(new_spins);

        auto new_coeffs = make_coeffs();
        torch::narrow(new_coeffs, /*dim=*/0, /*start=*/0,
                      /*length=*/_offset)
            .copy_(torch::narrow(_coeffs, /*dim=*/0, /*start=*/start,
                                 /*length=*/_offset));
        _coeffs = std::move(new_coeffs);
    }
    else {
        _counts = {};
        _offset = {};
        _spins  = make_spins();
        _coeffs = make_coeffs();
    }
    TCM_ASSERT(_offset < static_cast<int64_t>(_batch_size), "");
    return futures;
}

auto Buffer::submit_final() -> std::optional<std::future<std::invoke_result_t<Task>>>
{
    if (_counts.empty()) { return std::nullopt; }
    auto future =
        async(Task{_forward, torch::narrow(_spins, /*dim=*/0, /*start=*/0, /*length=*/_offset),
                   torch::narrow(_coeffs, /*dim=*/0, /*start=*/0, /*length=*/_offset),
                   std::move(_counts), _device,
                   /*complete=*/true});
    _spins  = torch::Tensor{};
    _coeffs = torch::Tensor{};
    _offset = 0;
    return std::optional{std::move(future)};
}

namespace {
auto log_plus_log(std::complex<double> const& log_acc,
                  std::complex<double> const& log_term) noexcept -> std::complex<double>
{
    auto scale = std::max(log_acc.real(), log_term.real());
    return scale + std::log(std::exp(log_acc - scale) + std::exp(log_term - scale));
}
} // namespace

class Accumulator {
    using future_type = Buffer::future_type;

    Buffer                  _buffer;
    OperatorT               _operator;
    std::queue<future_type> _queue;

    std::vector<std::complex<double>> _output;
    bool                              _complete;

  public:
    Accumulator(ForwardT forward, OperatorT op, uint64_t max_required_size, uint64_t batch_size,
                c10::Device device);

    auto operator()(torch::Tensor spins) -> torch::Tensor;

  private:
    auto drain(uint64_t count) -> void;

    auto process_future(future_type future) -> void;
    auto process_future(std::optional<future_type> future) -> void;
    auto process_futures(std::vector<future_type> futures) -> void;

    auto process_result(std::invoke_result_t<Task> result) -> void;
    template <class T>
    auto process_result_helper(torch::TensorAccessor<T, 1> const& coeffs, bool complete) -> void;
};

Accumulator::Accumulator(ForwardT forward, OperatorT op, uint64_t max_required_size,
                         uint64_t batch_size, c10::Device device)
    : _buffer{std::move(forward), max_required_size, batch_size, device}
    , _operator{std::move(op)}
    , _queue{}
    , _output{}
    , _complete{true}
{}

auto Accumulator::drain(uint64_t count) -> void
{
    for (auto i = uint64_t{0}; i < count && !_queue.empty(); ++i) {
        auto future = std::move(_queue.front());
        _queue.pop();
        try {
            process_result(future.get());
        }
        catch (...) {
            // Drain the rest of the queue since we don't know how to recover
            // anyway. We ignore all remaining exceptions and rethrow the first
            // one.
            auto exception_ptr = std::current_exception();
            for (; !_queue.empty(); _queue.pop()) {
                try {
                    _queue.front().get();
                }
                catch (...) {
                }
            }
            std::rethrow_exception(exception_ptr);
        }
    }
}

auto Accumulator::process_future(future_type future) -> void
{
    constexpr auto hard_max = 16U;
    constexpr auto soft_max = 8U;
    _queue.push(std::move(future));
    if (_queue.size() >= hard_max) { drain(_queue.size() - soft_max); }
}

auto Accumulator::process_future(std::optional<future_type> future) -> void
{
    if (future.has_value()) { process_future(*std::move(future)); }
}

auto Accumulator::process_futures(std::vector<future_type> futures) -> void
{
    for (auto& future : futures) {
        process_future(std::move(future));
    }
}

template <class T>
auto Accumulator::process_result_helper(torch::TensorAccessor<T, 1> const& coeffs,
                                        bool const                         complete) -> void
{
    TCM_ASSERT(coeffs.size(0) > 0, "");
    auto i = int64_t{0};
    if (!_complete) {
        TCM_ASSERT(!_output.empty(), "");
        _output.back() = log_plus_log(_output.back(), static_cast<std::complex<double>>(coeffs[i]));
        ++i;
    }
    _output.reserve(_output.size() + static_cast<uint64_t>(coeffs.size(0) - i));
    for (; i < coeffs.size(0); ++i) {
        _output.push_back(static_cast<std::complex<double>>(coeffs[i]));
    }
    _complete = complete;
}

auto Accumulator::process_result(std::invoke_result_t<Task> result) -> void
{
    auto const& [coeffs, complete] = result;
    switch (coeffs.scalar_type()) {
    case torch::ScalarType::ComplexFloat:
        process_result_helper<c10::complex<float>>(coeffs.accessor<c10::complex<float>, 1>(),
                                                   complete);
        break;
    case torch::ScalarType::ComplexDouble:
        process_result_helper<c10::complex<double>>(coeffs.accessor<c10::complex<double>, 1>(),
                                                    complete);
        break;
    default:
        TCM_ERROR(std::runtime_error,
                  fmt::format("expected either ComplexFloat or ComplexDouble, but got {}",
                              coeffs.scalar_type()));
    } // end switch
}

namespace {
template <class Allocator>
auto vector_to_tensor(std::vector<std::complex<double>, Allocator> vector) -> torch::Tensor
{
    using VectorT = std::vector<std::complex<double>, Allocator>;
    auto data     = vector.data();
    auto size     = static_cast<int64_t>(vector.size());
    auto deleter  = [original =
                        std::make_unique<VectorT>(std::move(vector)).release()](void* p) mutable {
        if (p != original->data()) {
            detail::assert_fail("false", __FILE__, __LINE__, TCM_CURRENT_FUNCTION,
                                fmt::format("Trying to delete wrong pointer: {} != {}", p,
                                            static_cast<void*>(original->data())));
        }
        std::default_delete<std::vector<std::complex<double>, Allocator>>{}(original);
    };
    return torch::from_blob(data, {size}, std::move(deleter),
                            torch::TensorOptions{}.dtype(torch::ScalarType::ComplexDouble));
}
} // namespace

auto Accumulator::operator()(torch::Tensor spins) -> torch::Tensor
{
    auto const spins_shape = spins.sizes();
    auto const spins_dim   = spins_shape.size();
    TCM_CHECK(
        spins_dim == 2 && spins_shape[1] == 8, std::domain_error,
        fmt::format("spins has wrong shape: [{}]; expected [?, 8]", fmt::join(spins_shape, ", ")));
    auto const device = spins.device();
    spins             = spins.to(spins.options().device(torch::kCPU));
    auto accessor     = spins.accessor<int64_t, 2>();
    for (auto i = int64_t{0}; i < accessor.size(0); ++i) {
        auto const row = accessor[i];

        ls_bits512 x;
        for (auto j = 0; j < 8; ++j) {
            x.words[j] = static_cast<uint64_t>(row[j]); // Yes overflows are fine
        }
        process_futures(_buffer.submit(x, _operator));
    }
    process_future(_buffer.submit_final());
    drain(_queue.size());
    TCM_ASSERT(_queue.empty(), "");
    auto r = vector_to_tensor(std::move(_output));
    return r.to(r.options().device(device));
}

TCM_EXPORT auto apply(torch::Tensor spins, OperatorT op, ForwardT psi, uint64_t max_required_size,
                      uint32_t batch_size) -> torch::Tensor
{
    Accumulator acc{std::move(psi), std::move(op), max_required_size, batch_size, spins.device()};
    return acc(std::move(spins));
}

TCM_NAMESPACE_END
