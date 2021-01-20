// Copyright (c) 2021, Tom Westerhout
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

#include "zanella.hpp"
#include "../errors.hpp"

TCM_NAMESPACE_BEGIN

// Zanella process
//
//
// zanella_process(current_state, log_prob_fn, generator_fn):
//     current_log_prob <- log_prob_fn(current_state)
//     states <- empty((number_samples, number_chains, 8))
//     log_probs <- empty((number_samples, number_chains))
//     weights <- empty((number_samples, number_chains))
//
//     states[0] <- current_state
//     log_prob[0] <- current_log_prob
//     current_state <- states[0]
//     current_log_prob <- log_prob[0]
//     current_weight <- weights[0]
//
//     iteration = 0
//     discard = True
//     while True:
//         possible_state, counts <- generator_fn(current_state) # CPU
//         possible_log_prob <- log_prob_fn(possible_state) # GPU
//         jump_rates, jump_rates_sum = zanella_jump_rates(current_log_prob, possible_log_prob, counts) # CPU
//         current_weight <- zanella_waiting_time(jump_rates_sum) # CPU
//
//         ++iteration
//         if discard:
//             if iteration == number_discarded:
//                 iteration <- 0
//                 discard <- False
//         else:
//             if iteration == number_samples:
//                 break
//             current_state <- states[iteration]
//             current_log_prob <- log_prob[iteration]
//             current_weight <- weights[iteration]
//
//         indices <- zanella_next_state_index(jump_rates, jump_rates_sum, counts, device)
//         current_state <- possible_state[indices]
//         current_log_prob <- possible_log_prob[indices]
//     return states, log_prob, weights

namespace {
auto check_basis(ls_spin_basis const& basis) -> ls_spin_basis const&
{
    TCM_CHECK(ls_get_number_spins(&basis) > 1, std::invalid_argument,
              "'ZanellaGenerator' uses binary flips, system size must thus be at least 2");
    return basis;
}
} // namespace

TCM_EXPORT ZanellaGenerator::ZanellaGenerator(ls_spin_basis const& basis)
    : _basis{ls_copy_spin_basis(&check_basis(basis))}
{}

TCM_EXPORT ZanellaGenerator::~ZanellaGenerator()
{
    ls_destroy_spin_basis(const_cast<ls_spin_basis*>(static_cast<ls_spin_basis const*>(_basis)));
}

TCM_EXPORT auto ZanellaGenerator::operator()(torch::Tensor x) const
    -> std::tuple<torch::Tensor, torch::Tensor>
{
    auto       pin_memory = false;
    auto const device     = x.device();
    auto const batch_size = x.size(0);
    TCM_CHECK(batch_size > 0, std::invalid_argument, "expected 'x' to contain a least one row");
    if (device.type() != torch::DeviceType::CPU) {
        pin_memory = true;
        x          = x.to(x.options().device(torch::DeviceType::CPU),
                 /*non_blocking=*/true);
    }
    TCM_CHECK(ls_get_hamming_weight(_basis) != -1, std::runtime_error,
              "ZanellaGenerator currently only supports bases with fixed magnetisation");
    auto const max_possible_states = static_cast<int64_t>(max_states());
    auto const options = torch::TensorOptions{}.dtype(torch::kInt64).pinned_memory(pin_memory);
    auto       y =
        torch::zeros(std::initializer_list<int64_t>{batch_size, max_possible_states, 8L}, options);
    auto counts = torch::empty(std::initializer_list<int64_t>{batch_size}, options);

    auto const x_info      = tensor_info<ls_bits512 const>(x, "x");
    auto const y_info      = tensor_info<ls_bits512, 2>(y);
    auto const counts_info = tensor_info<int64_t>(counts);

#pragma omp parallel for
    for (auto i = 0L; i < x_info.size(); ++i) {
        counts_info[i] = generate(x_info[i], row(y_info, i));
    }
    auto const size = *std::max_element(counts_info.data, counts_info.data + batch_size);

    y = torch::narrow(y, /*dim=*/1, /*start=*/0, /*length=*/size);
    if (device.type() != torch::DeviceType::CPU) {
        y      = y.to(y.options().device(device), /*non_blocking=*/pin_memory);
        counts = counts.to(counts.options().device(device), /*non_blocking=*/pin_memory);
    }
    return {std::move(y), std::move(counts)};
}

TCM_EXPORT auto ZanellaGenerator::generate(ls_bits512 const&         spin,
                                           TensorInfo<ls_bits512, 1> out) const -> unsigned
{
    auto const number_spins = ls_get_number_spins(_basis);
    ls_bits512 repr;
    std::fill(std::begin(repr.words), std::end(repr.words), uint64_t{0});
    double norm  = 0.0;
    auto   count = 0U;
    for (auto i = 0U; i < number_spins - 1; ++i) {
        for (auto j = i + 1U; j < number_spins; ++j) {
            if (test_bit(spin, i) != test_bit(spin, j)) {
                std::complex<double> dummy;
                auto                 possible = spin;
                toggle_bit(possible, i);
                toggle_bit(possible, j);
                ls_get_state_info(_basis, &possible, &repr, &dummy, &norm);
                if (norm > 0) { out[count++] = repr; }
            }
        }
    }
    TCM_CHECK(out.stride() == 1, std::runtime_error, "expected 'out' to be contiguous");
    std::sort(out.data, out.data + count);
    count = std::unique(out.data, out.data + count) - out.data;
    return count;
}

TCM_EXPORT auto zanella_choose_samples(torch::Tensor weights, int64_t const number_samples,
                                       double const time_step, c10::Device const device)
    -> torch::Tensor
{
    TCM_CHECK(weights.dim() == 1, std::invalid_argument,
              fmt::format("weights has wrong shape: [{}]; expected it to be a vector",
                          fmt::join(weights.sizes(), ", ")));
    TCM_CHECK(weights.device().type() == c10::DeviceType::CPU, std::invalid_argument,
              "weights must reside on the CPU");

    auto const pinned_memory = device != c10::DeviceType::CPU;
    auto       indices       = torch::empty(
        {number_samples}, torch::TensorOptions{}.dtype(torch::kInt64).pinned_memory(pinned_memory));
    if (number_samples == 0) { return indices.to(device); }

    AT_DISPATCH_FLOATING_TYPES(weights.scalar_type(), "zanella_choose_samples", [&] {
        auto weights_info = tensor_info<scalar_t const>(weights);
        auto indices_info = tensor_info<int64_t>(indices);
        auto time         = 0.0;
        auto index        = int64_t{0};
        indices_info[0]   = index;
        for (auto size = int64_t{1}; size < number_samples; ++size) {
            while (time + static_cast<double>(weights_info[index]) <= time_step) {
                time += static_cast<double>(weights_info[index]);
                ++index;
                TCM_CHECK(index < weights_info.size(), std::runtime_error, "time step is too big");
            }
            time -= time_step;
            indices_info[size] = index;
        }
    });
    return indices.to(indices.options().device(device),
                      /*non_blocking=*/pinned_memory);
}

TCM_NAMESPACE_END
