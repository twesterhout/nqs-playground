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
#include "errors.hpp"
#include <omp.h>

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

TCM_EXPORT ZanellaGenerator::ZanellaGenerator(ls_spin_basis const& basis,
                                              std::vector<std::pair<unsigned, unsigned>> edges)
    : _basis{ls_copy_spin_basis(&check_basis(basis))}
    , _edges{std::move(edges)}
{
    if (omp_get_max_active_levels() < 2) { omp_set_max_active_levels(2); }
    TCM_CHECK(!_edges.empty(), std::invalid_argument, "'edges' list must not be empty");
    for (auto const [i, j] : _edges) {
        TCM_CHECK(i != j && i < ls_get_number_spins(_basis) && j < ls_get_number_spins(_basis),
                  std::invalid_argument, fmt::format("'edges' list contains an invalid edge: ({}, {})", i, j));
    }
}

TCM_EXPORT ZanellaGenerator::~ZanellaGenerator()
{
    ls_destroy_spin_basis(const_cast<ls_spin_basis*>(static_cast<ls_spin_basis const*>(_basis)));
}

#if 0
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
    else {
        y = y.contiguous();
    }
    return {std::move(y), std::move(counts)};
}
#else
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

    auto const outer_num_threads = std::min<int>(x_info.size(), omp_get_max_threads());
    auto const inner_num_threads = omp_get_max_threads() / outer_num_threads;
#pragma omp parallel for num_threads(outer_num_threads)
    for (auto i = int64_t{0}; i < x_info.size(); ++i) {
        auto const initial_count = generate_general(x_info[i], row(y_info, i));
        counts_info[i] = project_states(row(y_info, i), initial_count, inner_num_threads);
    }
    auto const size = *std::max_element(counts_info.data, counts_info.data + batch_size);

    y = torch::narrow(y, /*dim=*/1, /*start=*/0, /*length=*/size);
    if (device.type() != torch::DeviceType::CPU) {
        y      = y.to(y.options().device(device), /*non_blocking=*/pin_memory);
        counts = counts.to(counts.options().device(device), /*non_blocking=*/pin_memory);
    }
    else {
        y = y.contiguous();
    }
    return {std::move(y), std::move(counts)};
}
#endif

TCM_NOINLINE auto ZanellaGenerator::generate_general(ls_bits512 const& spin, TensorInfo<ls_bits512> out) const -> int64_t
{
    auto const number_spins = ls_get_number_spins(_basis);
    auto       count = int64_t{0};
    // for (auto i = 0U; i < number_spins - 1; ++i) {
    //     for (auto j = i + 1U; j < number_spins; ++j) {
    for (auto const [i, j] : _edges) {
        if (test_bit(spin, i) != test_bit(spin, j)) {
            auto possible = spin;
            toggle_bit(possible, i);
            toggle_bit(possible, j);
            out[count++] = possible;
        }
    }
    //}
    TCM_CHECK(count > 0, std::runtime_error,
              "ZanellaGenerator got stuck: all potential states lie in a "
              "different magnetization sector");
    return count;
}

TCM_NOINLINE auto ZanellaGenerator::project_states(TensorInfo<ls_bits512> spins, int64_t count, int const num_threads) const -> int64_t
{
    TCM_ASSERT(count <= spins.size(), "buffer overflow");
    TCM_CHECK(spins.stride() == 1, std::runtime_error, "expected 'spins' to be contiguous");
    std::mutex zero_norm_mutex;
    std::vector<int64_t> zero_norm;

#pragma omp parallel for if(num_threads > 0) num_threads(num_threads)
    for (auto i = int64_t{0}; i < count; ++i) {
        std::complex<double> dummy;
        double norm;
        auto& spin = spins[i];
        auto repr = spin;
        ls_get_state_info(_basis, &spin, &repr, &dummy, &norm);
        if (norm > 0) { spin = repr; }
        else {
            std::unique_lock<std::mutex> lock{zero_norm_mutex};
            zero_norm.push_back(i);
        }
    }

    TCM_CHECK(static_cast<int64_t>(zero_norm.size()) < count, std::runtime_error,
              "ZanellaGenerator got stuck: all potential states have norm 0");
    if (!zero_norm.empty()) {
        std::sort(std::begin(zero_norm), std::end(zero_norm));
        auto const& fill_value = [&]() -> ls_bits512 const& {
            auto i = int64_t{0};
            for (; i < static_cast<int64_t>(zero_norm.size())
                        && zero_norm[static_cast<size_t>(i)] == i;
                    ++i) {}
            return spins[i];
        }();
        for (auto const i : zero_norm) {
            spins[i] = fill_value;
        }
    }

    std::sort(spins.data, spins.data + count);
    count = std::unique(spins.data, spins.data + count) - spins.data;
    return count;
}

#if 0
TCM_EXPORT auto ZanellaGenerator::generate(ls_bits512 const&         spin,
                                           TensorInfo<ls_bits512, 1> out) const -> unsigned
{
    auto const number_spins = ls_get_number_spins(_basis);
    ls_bits512 repr;
    set_zero(repr);
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
#endif

TCM_EXPORT auto zanella_choose_samples(torch::Tensor weights, int64_t const number_samples,
                                       double const time_step, c10::Device const device)
    -> torch::Tensor
{
    TCM_CHECK(weights.dim() == 1, std::invalid_argument,
              fmt::format("'weights' has wrong shape: [{}]; expected it to be a vector",
                          fmt::join(weights.sizes(), ", ")));
    TCM_CHECK(weights.device().type() == c10::DeviceType::CPU, std::invalid_argument,
              "'weights' must reside on the CPU");

    auto const pinned_memory = device != c10::DeviceType::CPU;
    auto       indices =
        torch::empty(std::initializer_list<int64_t>{number_samples},
                     torch::TensorOptions{}.dtype(torch::kInt64).pinned_memory(pinned_memory));
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
