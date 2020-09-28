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

#include "tabu.hpp"
#include "cpu/kernels.hpp"
#include "spin_basis.hpp"

#include <iostream>

TCM_NAMESPACE_BEGIN

class TabuGenerator {
  private:
    std::shared_ptr<BasisBase const> _basis;
    gsl::not_null<RandomGenerator*>  _generator;

  public:
    TabuGenerator(std::shared_ptr<BasisBase const> basis,
                  RandomGenerator& generator = global_random_generator());

    auto operator()(torch::Tensor x, c10::Device device) const
        -> std::tuple<torch::Tensor, std::vector<std::array<int, 3>>,
                      std::vector<int64_t>>;

    auto basis() const noexcept -> BasisBase const& { return *_basis; }

  private:
    auto generate(bits512 const& spin, gsl::span<bits512> out,
                  gsl::span<std::array<int, 3>> indices) const -> unsigned;
};

namespace {
template <class T>
TCM_NOINLINE auto compress(gsl::span<T> xs, gsl::span<int64_t const> counts,
                           unsigned const block_size) -> int64_t
{
    if (counts.empty()) { return 0; }
    auto dest = xs.data() + counts[0];
    auto src  = xs.data() + block_size;
    for (auto i = 1U; i < counts.size(); ++i) {
        auto const n = static_cast<uint64_t>(counts[i]);
        std::memmove(dest, src, sizeof(T) * n);
        dest += n;
        src += block_size;
    }
    return dest - xs.data();
}
} // namespace

TCM_EXPORT TabuGenerator::TabuGenerator(std::shared_ptr<BasisBase const> basis,
                                        RandomGenerator& generator)
    : _basis{std::move(basis)}, _generator{std::addressof(generator)}
{
    TCM_CHECK(_basis != nullptr, std::invalid_argument,
              "basis must not be nullptr");
}

TCM_EXPORT auto TabuGenerator::operator()(torch::Tensor x,
                                          c10::Device   device) const
    -> std::tuple<torch::Tensor, std::vector<std::array<int, 3>>,
                  std::vector<int64_t>>
{
    auto const pin_memory = device.type() != torch::DeviceType::CPU;
    auto const batch_size = x.size(0);
    TCM_CHECK(_basis->hamming_weight().has_value(), std::runtime_error,
              "TabuGenerator currently only supports bases with fixed "
              "magnetisation");
    auto const max_states = [n = _basis->number_spins()]() {
        return n * (n - 1) / 2;
    }();
    auto y = torch::empty(
        std::initializer_list<int64_t>{batch_size * max_states, 8L},
        torch::TensorOptions{}.dtype(torch::kInt64).pinned_memory(pin_memory));
    auto indices = std::vector<std::array<int, 3>>(
        static_cast<uint64_t>(batch_size * max_states));
    auto counts = std::vector<int64_t>(static_cast<size_t>(batch_size));

    auto const x_info = obtain_tensor_info<bits512 const>(x, "x");
    auto const y_info = obtain_tensor_info<bits512>(y);

    // #pragma omp parallel for
    for (auto i = 0L; i < x_info.size(); ++i) {
        auto const out = gsl::span<bits512>{y_info.data + i * max_states,
                                            static_cast<size_t>(max_states)};
        auto const is  = gsl::span<std::array<int, 3>>{
            indices.data() + i * max_states, static_cast<size_t>(max_states)};
        counts[static_cast<size_t>(i)] = generate(x_info[i], out, is);
    }
    auto const size = compress(
        gsl::span<bits512>{y_info.data, static_cast<size_t>(y_info.size())},
        counts, max_states);
    compress(gsl::span<std::array<int, 3>>{indices}, counts, max_states);

    y = torch::narrow(y, /*dim=*/0, /*start=*/0, /*length=*/size);
    indices.resize(static_cast<uint64_t>(size));
    if (device.type() != torch::DeviceType::CPU) {
        y = y.to(y.options().device(device), /*non_blocking=*/pin_memory);
    }
#if 1

#endif
    return {std::move(y), std::move(indices), std::move(counts)};
}

TCM_EXPORT auto
TabuGenerator::generate(bits512 const& spin, gsl::span<bits512> out,
                        gsl::span<std::array<int, 3>> indices) const -> unsigned
{
    // std::cout << "TabuGenerator::generate(" << spin.words[0] << ")\n";
    auto count = 0U;
    for (auto i = 0U; i < _basis->number_spins() - 1U; ++i) {
        for (auto j = i + 1U; j < _basis->number_spins(); ++j) {
            if (are_not_aligned(spin, i, j)) {
                auto index = 0U;
                auto const [repr, _, norm] =
                    _basis->full_info(flipped(spin, i, j), &index);
                if (norm > 0 && repr != spin) {
                    // std::cout << "Storing " << repr.words[0] << ", (" << i
                    //           << ", " << j << ", " << index << ")\n";
                    out[count]     = repr;
                    indices[count] = {static_cast<int>(i), static_cast<int>(j),
                                      static_cast<int>(index)};
                    ++count;
                }
            }
        }
    }
    return count;
}

// Given two site indices (i, j) compute the corresponding index in the alphas array.
//
// alphas is a 1d representation of the following matrix of dimension n x n:
//
//     _ X X X X X
//     _ _ X X X X
//     _ _ _ X X X
//     _ _ _ _ X X
//     _ _ _ _ _ X
//     _ _ _ _ _ _
constexpr auto compressed_index(unsigned const i, unsigned const j,
                                unsigned const n) noexcept -> unsigned
{
    // Normal row-major ordering minus the correction due to upper triangular structure.
    return i * n + j - (i + 1) * (i + 2) / 2;
}

auto tabu_compute_flags(gsl::span<std::array<int, 3> const> indices,
                        gsl::span<int8_t const> alphas, unsigned number_spins)
    -> std::vector<int8_t>
{
    auto out = std::vector<int8_t>(indices.size());
    std::transform(std::begin(indices), std::end(indices), std::begin(out),
                   [&alphas, number_spins](auto const& t) {
                       auto const [i, j, _] = t;
                       return alphas[compressed_index(i, j, number_spins)];
                   });
    return out;
}

// Kind of like sampling from a multinomial distribution given by `weights` except that we only take
// into account weights for which the corresponding flag is set.
template <class scalar_t>
auto choose_direction(gsl::span<scalar_t const> weights,
                      gsl::span<int8_t const> flags, double sum) -> int64_t
{
    TCM_CHECK(
        sum >= 0, std::invalid_argument,
        fmt::format("invalid sum: {}; expected a non-negative number", sum));
    std::cout << "weights: [";
    for (auto const w : weights) {
        std::cout << w << ", ";
    }
    std::cout << "]\n";
    std::cout << "flags: [";
    for (auto const f : flags) {
        std::cout << static_cast<int>(f) << ", ";
    }
    std::cout << "]\n";
    for (;;) {
        std::uniform_real_distribution<double> dist{0.0, sum};
        auto const u = dist(global_random_generator());
        auto       s = 0.0;
        for (auto i = uint64_t{0}; i < weights.size(); ++i) {
            if (flags[i]) {
                auto const w = weights[i];
                TCM_CHECK(w >= scalar_t{0}, std::invalid_argument,
                          fmt::format("encountered a negative weight: {}", w));
                s += static_cast<double>(w);
                if (s >= u) { return static_cast<int64_t>(i); }
            }
        }
        // Check that this is indeed an extremely unlikely numerical instability
        // rather than a bug
        TCM_CHECK(std::abs(s - sum) < 1e-7L * std::max(s, sum),
                  std::runtime_error,
                  fmt::format(
                      "provided sum does not match the computed one: {} != {}",
                      sum, s));
        sum = s;
    }
}

auto tabu_compute_flags(gsl::span<std::array<int, 3> const> indices,
                        gsl::span<int8_t const> alphas, unsigned number_spins)
    -> std::vector<int8_t>;

template <class scalar_t>
auto tabu_sum_rates(gsl::span<scalar_t const> jump_rates,
                    gsl::span<int8_t const> flags) -> std::tuple<double, double>
{
    TCM_CHECK(
        jump_rates.size() == flags.size(), std::invalid_argument,
        fmt::format("jump_rates and flags have different lengths: {} != {}",
                    jump_rates.size(), flags.size()));
    auto sum_plus = 0.0;
    auto sum_min  = 0.0;
    for (auto i = 0U; i < jump_rates.size(); ++i) {
        auto const rate = static_cast<double>(jump_rates[i]);
        if (flags[i]) { sum_plus += rate; }
        else {
            sum_min += rate;
        }
    }
    return {sum_plus, sum_min};
}

auto transform_pair(std::array<int, 3> const& indices, BasisBase const& basis)
    -> std::array<int, 2>
{
    bits512 spin;
    std::memset(spin.words, 0, sizeof(spin.words));
    set_bit_to(spin, std::get<0>(indices), true);
    set_bit_to(spin, std::get<1>(indices), true);
    spin = basis.apply_nth(spin, static_cast<unsigned>(std::get<2>(indices)));
    std::array<int, 2> out;
    auto               size = 0U;
    for (auto i = 0; i < static_cast<int>(std::size(spin.words)); ++i) {
        if (spin.words[i] == 0) { continue; }
        auto const first = __builtin_ctzl(spin.words[i]);
        out[size++]      = 64 * i + first;
        if (size == 2) { break; }
        auto const last = 63 - __builtin_clzl(spin.words[i]);
        if (last != first) {
            out[size++] = 64 * i + last;
            break;
        }
    }
    TCM_CHECK(size == 2U, std::runtime_error, "Bug! Bug! Bug!");
    return out;
}

auto transform_alphas(gsl::span<int8_t>         alphas,
                      std::array<int, 3> const& indices, BasisBase const& basis)
{
    std::vector<int8_t> old_alphas(std::begin(alphas), std::end(alphas));
    auto const          number_spins = basis.number_spins();
    old_alphas[compressed_index(std::get<0>(indices), std::get<1>(indices),
                                number_spins)] *= -1;
    auto       offset         = 0U;
    auto const symmetry_index = std::get<2>(indices);
    for (auto i = 0; i < number_spins - 1; ++i) {
        for (auto j = i + 1; j < number_spins; ++j, ++offset) {
            auto const [new_i, new_j] =
                transform_pair({i, j, symmetry_index}, basis);
            alphas[compressed_index(new_i, new_j, number_spins)] =
                old_alphas[offset];
        }
    }
}

template <class scalar_t> struct TabuWorker {
    gsl::span<bits512 const>            possible_states;
    gsl::span<scalar_t const>           possible_log_probs;
    gsl::span<std::array<int, 3> const> indices;
    gsl::span<int8_t>                   alphas;
    scalar_t&                           current_weight;
    bits512&                            current_state;
    scalar_t&                           current_log_prob;
    BasisBase const&                    basis;

    auto run() -> void
    {
        auto const jump_rates =
            tabu_jump_rates(possible_log_probs, current_log_prob);
        auto flags = tabu_compute_flags(indices, alphas, basis.number_spins());
        auto [rates_sum_plus, rates_sum_minus] =
            tabu_sum_rates(gsl::span{jump_rates}, flags);
        auto const rates_sum_max = std::max(rates_sum_plus, rates_sum_minus);
        auto&      generator     = global_random_generator();

        for (;;) {
            if (std::uniform_real_distribution<double>{}(
                    generator)*rates_sum_max
                <= rates_sum_plus) {
                auto const i = static_cast<uint64_t>(choose_direction(
                    gsl::span{jump_rates}, flags, rates_sum_plus));
                current_weight =
                    -std::log1p(
                        -std::uniform_real_distribution<double>{}(generator))
                    / rates_sum_max;
                current_state    = possible_states[i];
                current_log_prob = possible_log_probs[i];
                transform_alphas(alphas, indices[i], basis);
                break;
            }
            else {
                // flipping tau
                std::cout << "flipping tau..." << '\n';
                std::for_each(std::begin(alphas), std::end(alphas),
                              [](auto& a) { a *= -1; });
                std::for_each(std::begin(flags), std::end(flags),
                              [](auto& a) { a *= -1; });
                std::swap(rates_sum_plus, rates_sum_minus);
            }
        }
    }
};

class TabuProcess {
    // Output tensors
    torch::Tensor states;
    torch::Tensor log_probs;
    torch::Tensor weights;
    // Bookkeeping for Tabu
    torch::Tensor alphas;
    // Misc
    TabuGenerator generate_fn;
    v2::ForwardT  log_prob_fn;
    unsigned      number_samples;
    unsigned      number_discarded;
    c10::Device   device;

    /// Allocate and initialize `states`, `log_probs`, and `weights`. This also initializes
    /// `current_*` members.
    auto init_output(torch::Tensor init_state) -> void;
    auto init_alphas() -> void;

    auto forward(torch::Tensor state) -> torch::Tensor;

  public:
    TabuProcess(torch::Tensor _init_state, v2::ForwardT _log_prob_fn,
                std::shared_ptr<BasisBase const> _basis,
                unsigned _number_samples, unsigned _number_discarded);

    auto run() -> void;

    auto result() -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;
};

TCM_EXPORT
TabuProcess::TabuProcess(torch::Tensor _init_state, v2::ForwardT _log_prob_fn,
                         std::shared_ptr<BasisBase const> _basis,
                         unsigned _number_samples, unsigned _number_discarded)
    : states{}
    , log_probs{}
    , weights{}
    , alphas{}
    , generate_fn{std::move(_basis)}
    , log_prob_fn{std::move(_log_prob_fn)}
    , number_samples{_number_samples}
    , number_discarded{_number_discarded}
    , device{_init_state.device()}
{
    init_output(std::move(_init_state));
    init_alphas();
}

namespace {
auto _prepend_dim(torch::Tensor init, unsigned extra_dim) -> torch::Tensor
{
    auto init_shape = init.sizes();
    auto shape      = std::vector<int64_t>(init_shape.size() + 1U);
    shape[0]        = extra_dim;
    std::copy(std::begin(init_shape), std::end(init_shape),
              std::next(std::begin(shape)));
    return init.new_empty(shape);
}
} // namespace

auto TabuProcess::init_output(torch::Tensor init_state) -> void
{
    auto init_log_prob = log_prob_fn(init_state);
    if (init_log_prob.dim() > 1) { init_log_prob.squeeze_(/*dim=*/1); }
    states    = _prepend_dim(init_state, number_samples);
    log_probs = _prepend_dim(init_log_prob, number_samples);
    weights =
        torch::empty({static_cast<int64_t>(number_samples), init_state.size(0)},
                     torch::TensorOptions{}
                         .dtype(torch::kFloat32)
                         .device(c10::DeviceType::CPU));
    states[0].copy_(init_state);
    log_probs[0].copy_(init_log_prob);
}

auto TabuProcess::init_alphas() -> void
{
    auto const n             = generate_fn.basis().number_spins();
    auto const number_pairs  = n * (n - 1) / 2;
    auto const number_chains = states.size(1);
    auto const options =
        torch::TensorOptions{}.dtype(torch::kInt8).device(c10::DeviceType::CPU);
    alphas = torch::ones({number_chains, static_cast<int64_t>(number_pairs)},
                         options);
}

auto TabuProcess::forward(torch::Tensor state) -> torch::Tensor
{
    if (state.device() != device) {
        state = state.to(state.options().device(device));
    }
    auto log_prob = log_prob_fn(state);
    if (log_prob.dim() > 1) { log_prob.squeeze_(/*dim=*/1); }
    return log_prob.to(log_prob.options().device(c10::DeviceType::CPU));
}

template <class T> auto cum_sum(gsl::span<T const> xs) -> std::vector<T>
{
    auto sum = std::vector<int64_t>(xs.size());
    if (!xs.empty()) {
        sum[0] = T{0};
        for (auto i = 1UL; i < xs.size(); ++i) {
            sum[i] = sum[i - 1] + xs[i - 1];
        }
    }
    return sum;
}

auto TabuProcess::run() -> void
{
    auto const number_chains = static_cast<unsigned>(states.size(1));
    auto       iteration     = 0L;
    auto       discarded     = 0L;
    for (;;) {
        std::cout << "iteration=" << iteration << ", discarded=" << discarded
                  << '\n';
        auto const [possible_state, indices, counts] =
            generate_fn(states[iteration], device);
        auto const possible_log_prob = forward(possible_state);
        auto const offsets           = cum_sum(gsl::span{counts});

        auto weights_accessor   = weights.accessor<float, 2>();
        auto states_accessor    = states.accessor<int64_t, 3>();
        auto log_probs_accessor = log_probs.accessor<float, 2>();

        for (auto i = 0U; i < number_chains; ++i) {
            auto const size             = static_cast<uint64_t>(counts[i]);
            auto const _possible_states = gsl::span<bits512 const>{
                reinterpret_cast<bits512 const*>(possible_state.data_ptr())
                    + offsets[i],
                size};
            auto const _possible_log_probs = gsl::span<float const>{
                possible_log_prob.data_ptr<float>() + offsets[i], size};
            auto const _indices = gsl::span<std::array<int, 3> const>{
                indices.data() + offsets[i], size};
            auto const _alphas =
                gsl::span<int8_t>{alphas[i].data_ptr<int8_t>(),
                                  static_cast<uint64_t>(alphas.size(1))};
            auto& _current_weight = weights_accessor[iteration][i];
            auto& _current_state  = *reinterpret_cast<bits512*>(
                states_accessor[iteration][i].data());
            auto& _current_log_prob = log_probs_accessor[iteration][i];
            std::cout << _current_weight << ", " << _current_state.words[0]
                      << ", " << _current_log_prob << '\n';

            TabuWorker<float> worker{_possible_states,  _possible_log_probs,
                                     _indices,          _alphas,
                                     _current_weight,   _current_state,
                                     _current_log_prob, generate_fn.basis()};
            worker.run();
        }

        if (discarded < number_discarded) { ++discarded; }
        else {
            ++iteration;
            if (iteration == number_samples) { break; }
            for (auto i = 0U; i < number_chains; ++i) {
                *reinterpret_cast<bits512*>(
                    states_accessor[iteration][i].data()) =
                    *reinterpret_cast<bits512*>(
                        states_accessor[iteration - 1][i].data());
                log_probs_accessor[iteration][i] =
                    log_probs_accessor[iteration - 1][i];
            }
        }
    }
}

auto TabuProcess::result()
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
{
    return {states, log_probs, weights};
}

TCM_EXPORT auto tabu_process(torch::Tensor                    current_state,
                             v2::ForwardT                     log_prob_fn,
                             std::shared_ptr<BasisBase const> basis,
                             unsigned number_samples, unsigned number_discarded)
    -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
{
    auto process =
        TabuProcess{current_state, std::move(log_prob_fn), std::move(basis),
                    number_samples, number_discarded};
    process.run();
    return process.result();
}

TCM_NAMESPACE_END
