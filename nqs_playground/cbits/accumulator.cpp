// Copyright (c) 2020-2021, Tom Westerhout
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
#include "parallel.hpp"
#include "hedley.h"
#include <torch/types.h>
#include <future>

namespace tcm {
namespace {

struct Task {
    ForwardT             forward;
    torch::Tensor        spins;
    torch::Tensor        coeffs;
    std::vector<int64_t> counts;

    auto operator()() const -> torch::Tensor;
};

auto _check_forward_result(torch::Tensor const& output, int64_t const expected_batch_size,
                           c10::Device const expected_device) -> void
{
    if (HEDLEY_UNLIKELY(output.dim() != 1 || output.size(0) != expected_batch_size)) {
        std::ostringstream msg;
        msg << "forward_fn produced a tensor of wrong shape: " << output.sizes()
            << "; expected a 1D tensor of shape [" << expected_batch_size << "]";
        throw std::runtime_error{msg.str()};
    }
    if (HEDLEY_UNLIKELY(!output.is_complex())) {
        std::ostringstream msg;
        msg << "forward_fn produced a tensor of wrong dtype: " << output.scalar_type()
            << "; expected a complex tensor";
        throw std::runtime_error{msg.str()};
    }
    if (HEDLEY_UNLIKELY(output.device() != expected_device)) {
        std::ostringstream msg;
        msg << "forward_fn produced a tensor on the wrong device: " << output.device()
            << "; expected a tensor on " << expected_device;
        throw std::runtime_error{msg.str()};
    }
}

HEDLEY_NEVER_INLINE auto Task::operator()() const -> torch::Tensor
{
    torch::NoGradGuard no_grad;

    auto const device = spins.device();
    auto output = forward(spins);
    if (output.dim() > 1) { output.squeeze_(/*dim=*/1); }
    _check_forward_result(output, spins.size(0), device);
    // Convert the neural network output to std::complex<double> (from most
    // likely std::complex<float> to avoid the loss of precision when we do
    // exponentiation and other stuff)
    output = output.to(at::kComplexDouble);
    // Compute exp(output), but making sure it does not overflow
    auto       real  = torch::real(output);
    auto const scale = torch::max(real); // .item();
    real -= scale;
    output.exp_();

    auto       results = torch::empty({static_cast<int64_t>(counts.size())},
                                torch::TensorOptions{}.device(device).dtype(at::kComplexDouble));
    auto const cs      = coeffs.to(coeffs.options().device(device));
    // TCM_ASSERT(!torch::any(torch::isnan(cs)).item<bool>(), "");
    auto offset = int64_t{0};
    for (auto j = 0U; j < counts.size(); offset += counts[j++]) {
        if (HEDLEY_UNLIKELY(counts[j] <= 0)) {
            std::ostringstream msg;
            msg << "counts[" << j << "] = " << counts[j]
                << ", but expected a positive integer. This could be caused by the fact that "
                   "your lattice_symmetries.Operator contains a zero row. In that case, adding "
                   "a small diagonal correction might help.";
            throw std::runtime_error{msg.str()};
        }
        auto const slice_fn = [this, j, offset](auto const& t) {
            return torch::narrow(t, /*dim=*/0, /*start=*/offset, /*length=*/counts[j]);
        };
        results[j] = torch::dot(slice_fn(cs), slice_fn(output));
    }

    results.log_();
    torch::real(results) += scale;
    return results;
}

auto allocate_tensors_for_outer_batch(int64_t const batch_size, int64_t const max_required_size, bool const pin_memory)
    -> std::tuple<torch::Tensor, torch::Tensor, std::vector<int64_t>>
{
    auto spins = torch::empty(std::initializer_list<int64_t>{batch_size * max_required_size, 8L},
                              torch::TensorOptions{}.device(torch::kCPU).dtype(torch::kInt64).pinned_memory(pin_memory));
    std::memset(spins.data_ptr(), 0, sizeof(int64_t) * static_cast<size_t>(spins.numel()));
    auto coeffs = torch::empty(std::initializer_list<int64_t>{batch_size * max_required_size},
                               torch::TensorOptions{}.device(torch::kCPU).dtype(torch::kComplexDouble).pinned_memory(pin_memory));
    auto counts = std::vector<int64_t>(static_cast<size_t>(batch_size));
    return std::make_tuple(std::move(spins), std::move(coeffs), std::move(counts));
}

HEDLEY_NEVER_INLINE auto process_chunk(ls_bits512 const* spins, int64_t const batch_size, ls_operator const& op,
                   ForwardT fn, c10::Device const device) -> Task
{
    auto [other_spins, other_coeffs, counts] = allocate_tensors_for_outer_batch(
        batch_size, static_cast<int64_t>(ls_operator_max_buffer_size(&op)), device.type() == torch::kCUDA);
    auto const written = ls_batched_operator_apply(
        &op,
        static_cast<uint64_t>(batch_size),
        spins,
        static_cast<ls_bits512 *>(other_spins.data_ptr()),
        static_cast<std::complex<double>*>(other_coeffs.data_ptr()),
        reinterpret_cast<uint64_t*>(counts.data())
    );
    auto const slice_fn = [size = static_cast<int64_t>(written), device](auto t) {
        return t.narrow(/*dim=*/0, /*start=*/0, /*length=*/size).to(t.options().device(device), /*non_blocking=*/false);
    };
    return Task{
        std::move(fn),
        slice_fn(std::move(other_spins)),
        slice_fn(std::move(other_coeffs)),
        std::move(counts)
    };
}

auto log_apply(ls_bits512 const* spins, int64_t const count, ls_operator const& op,
               ForwardT fn, c10::Device const device, int64_t const batch_size) -> torch::Tensor
{
    std::optional<ThreadPool> pool{std::nullopt};
    if (device.type() != torch::kCPU) { pool.emplace(); }
    std::queue<std::future<std::invoke_result_t<Task>>> futures;
    // futures.reserve(static_cast<size_t>((count + batch_size - 1) / batch_size));
    std::vector<torch::Tensor> results;
    results.reserve(static_cast<size_t>((count + batch_size - 1) / batch_size));

    auto const make_future = [&pool](auto&& task) {
        if (pool.has_value()) {
            return pool->enqueue(std::forward<decltype(task)>(task));
        }
        return sync_task(std::forward<decltype(task)>(task));
    };
    // This is a hack to initialize CUcontext needed by Halide
    if (device.type() != torch::kCPU) {
        auto task = [device]() {
            auto dummy_tensor = torch::zeros(std::initializer_list<int64_t>{1},
                    torch::TensorOptions{}.device(device));
        };
        auto f = make_future(std::move(task));
        f.get();
    }
    auto i = int64_t{0};
    for (; i + batch_size <= count; i += batch_size) {
        auto task = process_chunk(spins + i, batch_size, op, fn, device);
        auto f = make_future(std::move(task));
        if (futures.size() > 1) {
            results.emplace_back(futures.front().get());
            futures.pop();
        }
        futures.emplace(std::move(f));
        // futures.back().wait();
    }
    if (i != count) {
        auto task = process_chunk(spins + i, count - i, op, fn, device);
        auto f = make_future(std::move(task));
        if (futures.size() > 1) {
            results.emplace_back(futures.front().get());
            futures.pop();
        }
        futures.emplace(std::move(f));
        // futures.back().wait();
    }

    // std::vector<torch::Tensor> results;
    // results.reserve(futures.size());
    while (!futures.empty()) {
        results.emplace_back(futures.front().get());
        futures.pop();
    }
    // for (auto& f : futures) {
    //     results.emplace_back(f.get());
    // }
    return torch::cat(results, /*dim=*/0);
}

auto _check_spins_tensor(torch::Tensor const& spins) -> void
{
    if (HEDLEY_UNLIKELY(spins.scalar_type() != torch::kInt64)) {
        std::ostringstream msg;
        msg << "'spins' tensor has wrong dtype: " << c10::toString(spins.scalar_type())
            << "; expected Long";
        throw std::invalid_argument{msg.str()};
    }
    if (HEDLEY_UNLIKELY(spins.dim() != 2 || spins.size(0) == 0 || spins.size(1) != 8)) {
        std::ostringstream msg;
        msg << "'spins' tensor has wrong shape: " << spins.sizes()
            << "; expected a non-empty 2D tensor of shape [_, 8]";
        throw std::invalid_argument{msg.str()};
    }
    if (HEDLEY_UNLIKELY(!spins.is_contiguous())) {
        std::ostringstream msg;
        msg << "'spins' tensor has wrong strides: " << spins.strides()
            << "; expected a contiguous tensor";
        throw std::invalid_argument{msg.str()};
    }
}

} // namespace

HEDLEY_PUBLIC auto log_apply(torch::Tensor spins, ls_operator const& op,
                             ForwardT fn, int64_t const batch_size) -> torch::Tensor
{
    _check_spins_tensor(spins);
    if (HEDLEY_UNLIKELY(batch_size <= 0)) {
        std::ostringstream msg;
        msg << "invalide 'batch_size': " << batch_size
            << "; expected a positive integer";
        throw std::invalid_argument{msg.str()};
    }
    auto const device = spins.device();
    spins             = spins.to(spins.options().device(torch::kCPU));
    return log_apply(
        static_cast<ls_bits512 const*>(spins.data_ptr()),
        spins.size(0),
        op,
        std::move(fn),
        device,
        batch_size
    );
}

} // namespace tcm
