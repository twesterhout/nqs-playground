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

#include "../common.hpp"
#include "../errors.hpp"
#include "../parallel.hpp"
#include "../tensor_info.hpp"
#include <gsl/gsl-lite.hpp>
#include <torch/types.h>
#include <future>
#include <queue>

TCM_NAMESPACE_BEGIN
namespace v2 {

struct Task {
    ForwardT              forward;
    torch::Tensor         spins;
    torch::Tensor         coeffs;
    std::vector<uint64_t> counts;
    c10::Device           device;
    bool                  complete;

    auto operator()() const -> std::tuple</*scale=*/double, /*complete=*/bool,
                                          /*coeffs=*/torch::Tensor>;
};

inline auto extract_scalar(char const* name, torch::Tensor const& x) -> double
{
    switch (x.scalar_type()) {
    case torch::kFloat32: return static_cast<double>(x.item<float>());
    case torch::kFloat64: return x.item<double>();
    default:
        TCM_ERROR(std::runtime_error,
                  fmt::format(
                      "{} has wrong type: {}; expected either Float or Double",
                      name, x.scalar_type()));
    }
}

TCM_NOINLINE auto Task::operator()() const
    -> std::tuple<double, bool, torch::Tensor>
{
    torch::NoGradGuard no_grad;

    auto const output = forward(spins.to(device));
    auto const dtype  = output.dtype();
    auto       real   = torch::real(output);
    auto const scale  = torch::max(real);
    real -= scale;
    output.exp_();

    auto results =
        torch::zeros({static_cast<int64_t>(counts.size())},
                     torch::TensorOptions{}.device(torch::kCPU).dtype(dtype));
    auto const cs = coeffs.to(coeffs.options().device(device).dtype(dtype));

    auto offset = int64_t{0};
    for (auto j = 0U; j < counts.size(); offset +=
                                         static_cast<int64_t>(counts[j++])) {
        auto const slice = [this, j, offset](auto const& t) {
            return torch::narrow(t, /*dim=*/0, /*start=*/offset,
                                 /*length=*/static_cast<int64_t>(counts[j]));
        };
        if (counts[j] > 0) {
            results[static_cast<int64_t>(j)] =
                torch::dot(slice(cs), slice(output));
        }
    }

    return {extract_scalar("scale", scale), complete, std::move(results)};
}

class Buffer {
    ForwardT              _forward;
    torch::Tensor         _spins;
    torch::Tensor         _coeffs;
    std::vector<uint64_t> _counts;
    int64_t               _offset;

    // General information
    uint64_t _batch_size; // How many spin configurations to include in a Task
    uint64_t _max_required_size; // Maximal number of spin configurations
                                 // which can be generated at a time
    c10::Device _device;         // On which device is _forward located

    auto make_spins() const -> torch::Tensor;
    auto make_coeffs() const -> torch::Tensor;

  public:
    using generate_fn_type = stdext::inplace_function<
        auto(gsl::span<bits512>, gsl::span<complex_type>)->uint64_t>;
    using future_type = std::future<std::invoke_result_t<Task>>;

    Buffer(v2::ForwardT forward, uint64_t max_required_size,
           uint64_t batch_size, c10::Device device);
    Buffer(Buffer const&) = delete;
    Buffer(Buffer&&)      = delete;
    auto operator=(Buffer const&) -> Buffer& = delete;
    auto operator=(Buffer&&) -> Buffer& = delete;

    auto submit(generate_fn_type op)
        -> std::vector<std::future<std::invoke_result_t<Task>>>;
    auto submit_final()
        -> std::optional<std::future<std::invoke_result_t<Task>>>;
};

Buffer::Buffer(v2::ForwardT forward, uint64_t max_required_size,
               uint64_t batch_size, c10::Device device)
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
    return torch::empty(
        {static_cast<int64_t>(_batch_size + _max_required_size), 8},
        torch::TensorOptions{}.device(torch::kCPU).dtype(torch::kInt64));
}

auto Buffer::make_coeffs() const -> torch::Tensor
{
    return torch::empty(
        {static_cast<int64_t>(_batch_size + _max_required_size)},
        torch::TensorOptions{}
            .device(torch::kCPU)
            .dtype(torch::kComplexDouble));
}

auto split_counts(std::vector<uint64_t> counts, uint64_t const batch_size)
    -> std::vector<std::pair<std::vector<uint64_t>, bool>>
{
    std::vector<std::pair<std::vector<uint64_t>, bool>> chunks;
    std::pair<std::vector<uint64_t>, bool>              chunk = {{}, {}};
    auto chunk_count                                          = uint64_t{0};

    for (auto i = uint64_t{0}; i < counts.size(); ++i) {
        auto current_count = counts[i];

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
    if (!chunk.first.empty()) { chunks.push_back(std::move(chunk)); }
    return chunks;
}

auto Buffer::submit(generate_fn_type op)
    -> std::vector<std::future<std::invoke_result_t<Task>>>
{
    TCM_ASSERT(_offset < static_cast<int64_t>(_batch_size), "");
    auto const written = op(
        gsl::span{static_cast<bits512*>(_spins.data_ptr()), _max_required_size},
        gsl::span{static_cast<std::complex<double>*>(_coeffs.data_ptr()),
                  _max_required_size});
    _offset += static_cast<int64_t>(written);
    _counts.push_back(written);

    std::vector<std::future<std::invoke_result_t<Task>>> futures;
    if (_offset < static_cast<int64_t>(_batch_size)) { return futures; }

    auto chunks = split_counts(_counts, _batch_size);
    TCM_ASSERT(!chunks.empty(), "");
    for (auto i = uint64_t{0}; i < chunks.size() - 1; ++i) {
        auto const start  = static_cast<int64_t>(i * _batch_size);
        auto const length = static_cast<int64_t>(_batch_size);
        auto       future =
            async(Task{_forward,
                       torch::narrow(_spins, /*dim=*/0, /*start=*/start,
                                     /*length=*/length),
                       torch::narrow(_coeffs, /*dim=*/0, /*start=*/start,
                                     /*length=*/length),
                       std::move(chunks[i].first), _device, chunks[i].second});
        futures.push_back(std::move(future));
    }
    {
        _counts = std::move(chunks.back().first);
        _offset = static_cast<int64_t>(std::accumulate(
            std::begin(_counts), std::end(_counts), uint64_t{0}));
        auto const start =
            static_cast<int64_t>((chunks.size() - 1) * _batch_size);

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
    return futures;
}

auto Buffer::submit_final()
    -> std::optional<std::future<std::invoke_result_t<Task>>>
{
    if (_counts.empty()) { return std::nullopt; }
    auto future = async(
        Task{_forward,
             torch::narrow(_spins, /*dim=*/0, /*start=*/0, /*length=*/_offset),
             torch::narrow(_coeffs, /*dim=*/0, /*start=*/0, /*length=*/_offset),
             std::move(_counts), _device,
             /*complete=*/true});
    _spins  = torch::Tensor{};
    _coeffs = torch::Tensor{};
    _offset = 0;
    return std::optional{std::move(future)};
}

class Accumulator {
    using future_type = Buffer::future_type;

    Buffer                  _buffer;
    OperatorT               _operator;
    std::queue<future_type> _queue;

  public:
    Accumulator(v2::ForwardT forward, OperatorT op, uint64_t max_required_size,
                uint64_t batch_size, c10::Device device);

    auto operator()(torch::Tensor const& spins) -> torch::Tensor;
};

Accumulator::Accumulator(v2::ForwardT forward, OperatorT op,
                         uint64_t max_required_size, uint64_t batch_size,
                         c10::Device device)
    : _buffer{std::move(forward), max_required_size, batch_size, device}
    , _operator{std::move(op)}
    , _queue{}
{}

} // namespace v2
TCM_NAMESPACE_END
