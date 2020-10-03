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

#include "metropolis.hpp"
#include "errors.hpp"
#include "random.hpp"
#include "spin_basis.hpp"

#include <boost/align/is_aligned.hpp>
#include <random> // for uniform_real_distribution

TCM_NAMESPACE_BEGIN

namespace detail {

constexpr auto find_nth_one_simple(uint64_t const v, unsigned r) noexcept
    -> unsigned
{
    ++r; // counting from 1
    auto i = 0U;
    for (; i < 64U; ++i) {
        if ((v >> i) & 0x01) {
            --r;
            if (r == 0) { break; }
        }
    }
    return i;
}

constexpr auto find_nth_one(uint64_t const v, unsigned r) noexcept -> unsigned
{
    // Do a normal parallel bit count for a 64-bit integer,
    // but store all intermediate steps.
    // a = (v & 0x5555...) + ((v >> 1) & 0x5555...);
    auto const a = v - ((v >> 1) & ~0UL / 3);
    // b = (a & 0x3333...) + ((a >> 2) & 0x3333...);
    auto const b = (a & ~0UL / 5) + ((a >> 2) & ~0UL / 5);
    // c = (b & 0x0F0F...) + ((b >> 4) & 0x0F0F...);
    auto const c = (b + (b >> 4)) & ~0UL / 0x11;
    // d = (c & 0x00FF...) + ((c >> 8) & 0x00FF...);
    auto const d = (c + (c >> 8)) & ~0UL / 0x101;
    auto       s = 0U; // Output: Resulting position of bit with rank r [0-63]
    // Now do branchless select!
    auto t =
        static_cast<unsigned>((d + (d >> 16)) & 0xFF); // Bit count temporary.
    ++r;
    // if (r > t) {s += 32; r -= t;}
    s += ((t - r) & 0x100) >> 3;
    r -= (t & ((t - r) >> 8));
    t = (d >> s) & 0xFF;
    // if (r > t) {s += 16; r -= t;}
    s += ((t - r) & 0x100) >> 4;
    r -= (t & ((t - r) >> 8));
    t = (c >> s) & 0xF;
    // if (r > t) {s += 8; r -= t;}
    s += ((t - r) & 0x100) >> 5;
    r -= (t & ((t - r) >> 8));
    t = (b >> s) & 0x7;
    // if (r > t) {s += 4; r -= t;}
    s += ((t - r) & 0x100) >> 6;
    r -= (t & ((t - r) >> 8));
    t = (a >> s) & 0x3;
    // if (r > t) {s += 2; r -= t;}
    s += ((t - r) & 0x100) >> 7;
    r -= (t & ((t - r) >> 8));
    t = (v >> s) & 0x1;
    // if (r > t) ++s;
    s += ((t - r) & 0x100) >> 8;
    return s;
}

inline auto find_nth_one(bits512 const& x, unsigned n) noexcept -> unsigned
{
    static_assert(std::is_same_v<unsigned long, uint64_t>,
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    for (auto i = 0U;; ++i) {
        if (i == std::size(x.words)) { return 512U; }
        auto const count =
            static_cast<unsigned>(__builtin_popcountl(x.words[i]));
        if (n <= count) { return 64U * i + find_nth_one(x.words[i], n); }
        n -= count;
    }
}

inline auto find_nth_zero(bits512 const& x, unsigned const n) noexcept
    -> unsigned
{
    using std::begin, std::end;
    bits512 y;
    std::transform(begin(x.words), end(x.words), begin(y.words),
                   [](auto const w) { return ~w; });
    return find_nth_one(y, n);
}

} // namespace detail

TCM_EXPORT
MetropolisKernel::MetropolisKernel(std::shared_ptr<BasisBase const> basis,
                                   RandomGenerator&                 generator)
    : _basis{std::move(basis)}, _generator{std::addressof(generator)}
{
    TCM_CHECK(_basis != nullptr, std::invalid_argument,
              "basis must not be nullptr");
}

TCM_EXPORT auto MetropolisKernel::operator()(torch::Tensor x) const
    -> std::tuple<torch::Tensor, torch::Tensor>
{
    auto       pin_memory = false;
    auto const device     = x.device();
    if (device.type() != torch::DeviceType::CPU) {
        pin_memory = true;
        x          = x.cpu();
    }
    auto x_info = obtain_tensor_info<bits512 const>(x, "x");
    auto y      = torch::empty(
        std::initializer_list<int64_t>{x_info.size(), 8L},
        torch::TensorOptions{}.dtype(torch::kInt64).pinned_memory(pin_memory));
    auto norm = torch::empty(std::initializer_list<int64_t>{x_info.size()},
                             torch::TensorOptions{}
                                 .dtype(torch::kFloat32)
                                 .pinned_memory(pin_memory));
    kernel_cpu(x_info, obtain_tensor_info<bits512>(y),
               obtain_tensor_info<float>(norm));
    if (device.type() != torch::DeviceType::CPU) {
        y = y.to(y.options().device(device), /*non_blocking=*/pin_memory);
        norm =
            norm.to(norm.options().device(device), /*non_blocking=*/pin_memory);
    }
    return {std::move(y), std::move(norm)};
}

#if 0
TCM_EXPORT auto MetropolisKernel::operator()(torch::Tensor x) const
    -> std::tuple<torch::Tensor, torch::Tensor>
{
    TCM_CHECK(x.dim() == 1, std::invalid_argument,
              fmt::format("x has wrong number of dimensions: {}; expected "
                          "a one-dimensional tensor",
                          x.dim()));
    TCM_CHECK_TYPE("x", x, torch::kInt64);
    TCM_CHECK_CONTIGUOUS("x", x);
    auto       pin_memory = false;
    auto const device     = x.device();
    auto const n          = x.numel();
    if (device.type() != torch::DeviceType::CPU) {
        pin_memory = true;
        x          = x.cpu();
    }
    auto y = torch::empty(
        {n},
        torch::TensorOptions{}.dtype(torch::kInt64).pinned_memory(pin_memory));
    auto norm = torch::empty({n}, torch::TensorOptions{}
                                      .dtype(torch::kFloat32)
                                      .pinned_memory(pin_memory));
    kernel_cpu(
        static_cast<size_t>(n), reinterpret_cast<uint64_t const*>(x.data_ptr()),
        reinterpret_cast<uint64_t*>(y.data_ptr()), norm.data_ptr<float>());
    if (device.type() != torch::DeviceType::CPU) {
        y = y.to(y.options().device(device), /*non_blocking=*/pin_memory);
        norm =
            norm.to(norm.options().device(device), /*non_blocking=*/pin_memory);
    }
    return {std::move(y), std::move(norm)};
}
#endif

auto MetropolisKernel::kernel_cpu(TensorInfo<bits512 const> const& src_info,
                                  TensorInfo<bits512> const&       dst_info,
                                  TensorInfo<float> const& norm_info) const
    -> void
{
    auto const loop = [&](auto&& task) {
        auto const* src  = src_info.data;
        auto*       dst  = dst_info.data;
        auto*       norm = norm_info.data;
        for (auto i = int64_t{0}; i < src_info.size(); ++i,
                  src += src_info.stride(), dst += dst_info.stride(),
                  norm += norm_info.stride()) {
            std::tie(*dst, *norm) = task(*src);
        }
    };
    if (_basis->hamming_weight().has_value()) {
        loop([this](auto const& x) -> std::tuple<bits512, float> {
            for (;;) {
                auto const i = detail::find_nth_one(
                    x, random_bounded(*_basis->hamming_weight(), *_generator));
                auto const j = detail::find_nth_zero(
                    x, random_bounded(_basis->number_spins()
                                          - *_basis->hamming_weight(),
                                      *_generator));
                auto const [y, _, normalization] =
                    _basis->full_info(flipped(x, i, j));
                if (normalization > 0.0 && y != x) {
                    return {y, static_cast<float>(normalization)};
                }
            }
        });
    }
    else {
        loop([this](auto const& x) -> std::tuple<bits512, float> {
            auto const i = random_bounded(_basis->number_spins(), *_generator);
            auto const [y, _, normalization] = _basis->full_info(flipped(x, i));
            return {y, static_cast<float>(normalization)};
        });
    }
}

#if 0
auto MetropolisKernel::kernel_cpu(size_t const                          n,
                                  SpinBasis::StateT const* TCM_RESTRICT src,
                                  SpinBasis::StateT* TCM_RESTRICT       dst,
                                  float* TCM_RESTRICT norm) const -> void
{
    auto const loop = [n, src, dst, norm](auto&& task) {
        for (auto i = size_t{0}; i < n; ++i) {
            auto const [dst_i, norm_i] = task(src[i]);
            dst[i]                     = dst_i;
            norm[i]                    = static_cast<float>(norm_i);
        }
    };
    if (_basis->hamming_weight().has_value()) {
        loop([this](auto const x) -> std::tuple<uint64_t, double> {
            auto i = detail::find_nth_set(
                x, random_bounded(*_basis->hamming_weight(), *_generator));
            auto j = detail::find_nth_set(
                ~x, random_bounded(_basis->number_spins()
                                       - *_basis->hamming_weight(),
                                   *_generator));
            auto const [y, _, normalization] =
                _basis->full_info(detail::flipped(x, i, j));
            return {y, normalization};
        });
    }
    else {
        loop([this](auto const x) -> std::tuple<uint64_t, double> {
            auto i = random_bounded(_basis->number_spins(), *_generator);
            auto const [y, _, normalization] =
                _basis->full_info(detail::flipped(x, i));
            return {y, normalization};
        });
    }
}
#endif

TCM_EXPORT
ProposalGenerator::ProposalGenerator(std::shared_ptr<BasisBase const> basis,
                                     RandomGenerator&                 generator)
    : _basis{std::move(basis)}, _generator{std::addressof(generator)}
{
    TCM_CHECK(_basis != nullptr, std::invalid_argument,
              "basis must not be nullptr");
}

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

#if 0
TCM_EXPORT auto ProposalGenerator::operator()(torch::Tensor x) const
    -> std::tuple<torch::Tensor, std::vector<int64_t>>
{
    auto       pin_memory = false;
    auto const device     = x.device();
    auto const batch_size = x.size(0);
    if (device.type() != torch::DeviceType::CPU) {
        pin_memory = true;
        x          = x.to(x.options().device(torch::DeviceType::CPU),
                 /*non_blocking=*/true);
    }
    TCM_CHECK(_basis->hamming_weight().has_value(), std::runtime_error,
              "ProposalGenerator currently only supports bases with fixed "
              "magnetisation");
    auto const max_states =
        *_basis->hamming_weight()
        * (_basis->number_spins() - *_basis->hamming_weight());
    auto y = torch::empty(
        std::initializer_list<int64_t>{batch_size * max_states, 8L},
        torch::TensorOptions{}.dtype(torch::kInt64).pinned_memory(pin_memory));
    std::vector<int64_t> counts(static_cast<size_t>(batch_size));

    auto                 x_info = obtain_tensor_info<bits512 const>(x, "x");
    auto                 y_info = obtain_tensor_info<bits512, false>(y);
    auto                 size   = 0L;
    std::vector<bits512> workspace;
    for (auto i = 0L; i < x_info.size(); ++i) {
        generate(x_info.data[i * x_info.stride()], workspace);
        // Copy
        for (auto const& spin : workspace) {
            TCM_ASSERT(size < y_info.size(), "output buffer is full");
            y_info.data[(size++) * y_info.stride()] = spin;
        }
        counts[static_cast<size_t>(i)] = static_cast<int64_t>(workspace.size());
    }

    y = torch::narrow(y, /*dim=*/0, /*start=*/0, /*length=*/size);
    if (device.type() != torch::DeviceType::CPU) {
        y = y.to(y.options().device(device), /*non_blocking=*/pin_memory);
    }
    return {std::move(y), std::move(counts)};
}
#else
TCM_EXPORT auto ProposalGenerator::operator()(torch::Tensor x) const
    -> std::tuple<torch::Tensor, std::vector<int64_t>>
{
    auto       pin_memory = false;
    auto const device     = x.device();
    auto const batch_size = x.size(0);
    if (device.type() != torch::DeviceType::CPU) {
        pin_memory = true;
        x          = x.to(x.options().device(torch::DeviceType::CPU),
                 /*non_blocking=*/true);
    }
    TCM_CHECK(_basis->hamming_weight().has_value(), std::runtime_error,
              "ProposalGenerator currently only supports bases with fixed "
              "magnetisation");
    auto const max_states =
        *_basis->hamming_weight()
        * (_basis->number_spins() - *_basis->hamming_weight());
    auto y = torch::empty(
        std::initializer_list<int64_t>{batch_size * max_states, 8L},
        torch::TensorOptions{}.dtype(torch::kInt64).pinned_memory(pin_memory));
    std::vector<int64_t> counts(static_cast<size_t>(batch_size));

    auto const x_info = obtain_tensor_info<bits512 const>(x, "x");
    auto const y_info = obtain_tensor_info<bits512>(y);

#    pragma omp parallel for
    for (auto i = 0L; i < x_info.size(); ++i) {
        counts[static_cast<size_t>(i)] =
            generate(x_info[i], gsl::span<bits512>{y_info.data + i * max_states,
                                                   max_states});
    }
    auto const size = compress(
        gsl::span<bits512>{y_info.data, static_cast<size_t>(y_info.size())},
        counts, max_states);

    y = torch::narrow(y, /*dim=*/0, /*start=*/0, /*length=*/size);
    if (device.type() != torch::DeviceType::CPU) {
        y = y.to(y.options().device(device), /*non_blocking=*/pin_memory);
    }
    return {std::move(y), std::move(counts)};
}
#endif

TCM_EXPORT auto ProposalGenerator::generate(bits512 const&     spin,
                                            gsl::span<bits512> out) const
    -> unsigned
{
    TCM_ASSERT(_basis->number_spins() > 0, "");
    auto count = 0U;

    for (auto i = 0U; i < _basis->number_spins() - 1; ++i) {
        for (auto j = i + 1U; j < _basis->number_spins(); ++j) {
            if (are_not_aligned(spin, i, j)) {
                auto const [repr, _, norm] =
                    _basis->full_info(flipped(spin, i, j));
                if (norm > 0 && repr != spin) { out[count++] = repr; }
            }
        }
    }

    std::sort(out.data(), out.data() + count);
    count = std::unique(out.data(), out.data() + count) - out.data();
    return count;
}

TCM_EXPORT auto ProposalGenerator::generate(bits512 const&        spin,
                                            std::vector<bits512>& ys) const
    -> void
{
    TCM_ASSERT(_basis->number_spins() > 0, "");
    auto const max_states =
        *_basis->hamming_weight()
        * (_basis->number_spins() - *_basis->hamming_weight());
    ys.clear();
    ys.reserve(max_states);

    for (auto i = 0U; i < _basis->number_spins() - 1; ++i) {
        for (auto j = i + 1U; j < _basis->number_spins(); ++j) {
            if (are_not_aligned(spin, i, j)) {
                auto const [repr, _, norm] =
                    _basis->full_info(flipped(spin, i, j));
                if (norm > 0 && repr != spin) { ys.push_back(repr); }
            }
        }
    }

    using std::begin, std::end;
    std::sort(begin(ys), end(ys));
    ys.resize(static_cast<size_t>(
        std::distance(begin(ys), std::unique(begin(ys), end(ys)))));
}

#if 0
TCM_EXPORT auto _add_waiting_time_(torch::Tensor time, torch::Tensor rates)
    -> void
{
    torch::NoGradGuard no_grad;
    TCM_CHECK(time.device().type() == torch::DeviceType::CPU,
              std::invalid_argument, "t must reside on the CPU");
    TCM_CHECK(rates.device().type() == torch::DeviceType::CPU,
              std::invalid_argument, "t must reside on the CPU");
    auto time_info  = obtain_tensor_info<float>(time, "t");
    auto rates_info = obtain_tensor_info<float>(rates, "rates");
    TCM_CHECK(time_info.size() == rates_info.size(), std::invalid_argument,
              fmt::format(
                  "t and rates tensors have incompatible shapes: [{}] != [{}]",
                  fmt::join(time.sizes(), ", "),
                  fmt::join(rates.sizes(), ", ")));
    auto& generator = global_random_generator();
    for (auto i = 0L; i < time_info.size(); ++i) {
        auto const waiting =
            -std::log1p(-std::uniform_real_distribution<double>{}(generator))
            / rates_info.data[i * rates_info.stride()];
        time_info.data[i * time_info.stride()] += static_cast<float>(waiting);
    }
}
#endif

TCM_EXPORT auto zanella_waiting_time(torch::Tensor                rates,
                                     c10::optional<torch::Tensor> out)
    -> torch::Tensor
{
    torch::NoGradGuard no_grad;
    // We assume that rates lives on the CPU since it'll usually be small and it
    // makes no sense to lauch GPU kernels for it.
    TCM_CHECK(rates.device().type() == torch::DeviceType::CPU,
              std::invalid_argument, "rates must reside on the CPU");
    auto rates_info = obtain_tensor_info<float>(rates, "rates");

    TensorInfo<float> out_info;
    if (!out.has_value()) {
        out.emplace(torch::empty({static_cast<int64_t>(rates_info.size())},
                                 torch::kFloat32));
        out_info = obtain_tensor_info<float>(*out, "out");
    }
    else {
        // If out is present, make sure it lives on CPU as well, has the right
        // dtype and shape
        TCM_CHECK(out->device().type() == torch::DeviceType::CPU,
                  std::invalid_argument, "out must reside on the CPU");
        out_info = obtain_tensor_info<float>(*out, "out");
        TCM_CHECK(out_info.size() == rates_info.size(), std::invalid_argument,
                  fmt::format("out has wrong size: {}; expected {}",
                              out_info.size(), rates_info.size()));
    }

    // The kernel
    auto& generator = global_random_generator();
    for (auto i = 0L; i < rates_info.size(); ++i) {
        auto const waiting =
            -std::log1p(-std::uniform_real_distribution<double>{}(generator))
            / static_cast<double>(rates_info[i]);
        out_info[i] = static_cast<float>(waiting);
    }

    return *out;
}

#if 0
TCM_EXPORT auto
_store_ready_samples_(torch::Tensor states, torch::Tensor log_probs,
                      torch::Tensor sizes, torch::Tensor current_state,
                      torch::Tensor current_log_prob, torch::Tensor times,
                      float const thin_rate) -> bool
{
    TCM_CHECK(thin_rate > 0.0f, std::invalid_argument,
              fmt::format("invalid thin_rate: {}; expected a positive number",
                          thin_rate));
    TCM_CHECK(times.device().type() == torch::DeviceType::CPU,
              std::invalid_argument, "times must reside on the CPU");
    TCM_CHECK(sizes.device().type() == torch::DeviceType::CPU,
              std::invalid_argument, "sizes must reside on the CPU");
    auto times_info = obtain_tensor_info<float>(times, "times");
    auto sizes_info = obtain_tensor_info<int64_t>(sizes, "sizes");
    TCM_CHECK(
        times_info.size() == sizes_info.size(), std::invalid_argument,
        fmt::format(
            "times and sizes tensors have incompatible shapes: [{}] != [{}]",
            fmt::join(times.sizes(), ", "), fmt::join(sizes.sizes(), ", ")));
    auto const max_size = states.size(0);

    auto const get_dst = [max_size](auto& tensor, int64_t const j,
                                    int64_t const offset, int64_t const count) {
        return torch::narrow(
            torch::narrow(tensor, /*dim=*/1, /*start=*/j, /*length=*/1),
            /*dim=*/0,
            /*start=*/offset,
            /*length=*/std::min(count, max_size - offset));
    };

    for (auto i = 0L; i < times_info.size(); ++i) {
        auto& n = sizes_info.data[i * sizes_info.stride()];
        auto& t = times_info.data[i * times_info.stride()];
        TCM_ASSERT(n <= max_size, "");
        // We know that times_info is bounded by ≈40 (because
        // -log1p(-(1 - ε)) ≈ 36 for float64), so the following is safe.
        int quotient;
        t = std::remquo(t, thin_rate, &quotient);
        if (quotient > 0 && n < max_size) {
            get_dst(states, i, n, quotient)
                .copy_(current_state[i], /*non_blocking=*/true);
            get_dst(log_probs, i, n, quotient)
                .copy_(current_log_prob[i], /*non_blocking=*/true);
            n = std::min(n + quotient, max_size);
        }
    }

    auto should_stop = true;
    for (auto i = 0L; i < times_info.size(); ++i) {
        auto const n = sizes_info.data[i * sizes_info.stride()];
        TCM_ASSERT(n <= max_size, "");
        if (n != max_size) {
            should_stop = false;
            break;
        }
    }
    return should_stop;
}
#endif

TCM_EXPORT auto
zanella_choose_samples(torch::Tensor weights, int64_t const number_samples,
                       double const time_step, c10::Device const device)
    -> torch::Tensor
{
    TCM_CHECK(
        weights.dim() == 1, std::invalid_argument,
        fmt::format("weights has wrong shape: [{}]; expected it to be a vector",
                    fmt::join(weights.sizes(), ", ")));
    TCM_CHECK(weights.device().type() == c10::DeviceType::CPU,
              std::invalid_argument, "weights must reside on the CPU");

    auto const pinned_memory = device != c10::DeviceType::CPU;
    auto       indices =
        torch::empty({number_samples}, torch::TensorOptions{}
                                           .dtype(torch::kInt64)
                                           .pinned_memory(pinned_memory));
    if (number_samples == 0) { return indices.to(device); }

    AT_DISPATCH_FLOATING_TYPES(
        weights.scalar_type(), "zanella_choose_samples", [&] {
            auto weights_info = obtain_tensor_info<scalar_t const>(weights);
            auto indices_info = obtain_tensor_info<int64_t>(indices);
            auto time         = 0.0;
            auto index        = int64_t{0};
            indices_info[0]   = index;
            for (auto size = int64_t{1}; size < number_samples; ++size) {
                while (time + static_cast<double>(weights_info[index])
                       <= time_step) {
                    time += static_cast<double>(weights_info[index]);
                    ++index;
                    TCM_CHECK(index < weights_info.size(), std::runtime_error,
                              "time step is too big");
                }
                time -= time_step;
                indices_info[size] = index;
            }
        });
    return indices.to(indices.options().device(device),
                      /*non_blocking=*/pinned_memory);
}

template <class scalar_t>
auto sample_one_from_multinomial(TensorInfo<scalar_t const> weights,
                                 scalar_t                   sum) -> int64_t
{
    TCM_CHECK(
        sum >= scalar_t{0}, std::invalid_argument,
        fmt::format("invalid sum: {}; expected a non-negative number", sum));
    for (;;) {
        std::uniform_real_distribution<long double> dist{0.0L, sum};
        auto const u = dist(global_random_generator());
        auto       s = 0.0L;
        for (auto i = int64_t{0}; i < weights.size(); ++i) {
            auto const w = weights[i];
            TCM_CHECK(w >= scalar_t{0}, std::invalid_argument,
                      fmt::format("encountered a negative weight: {}", w));
            s += static_cast<long double>(w);
            if (s >= u) { return i; }
        }
        // Check that this is indeed an extremely unlikely numerical instability
        // rather than a bug
        TCM_CHECK(std::abs(s - static_cast<long double>(sum))
                      < 1e-5L * std::max<long double>(s, sum),
                  std::runtime_error,
                  fmt::format(
                      "provided sum does not match the computed one: {} != {}",
                      sum, s));
        sum = static_cast<scalar_t>(s);
    }
}

TCM_EXPORT auto zanella_next_state_index(torch::Tensor jump_rates,
                                         torch::Tensor jump_rates_sum,
                                         std::vector<int64_t> const& counts,
                                         c10::Device const           device)
    -> torch::Tensor
{
    using scalar_t = float;
    TCM_CHECK(jump_rates.device().is_cpu(), std::invalid_argument,
              "jump_rates must reside on the CPU");
    TCM_CHECK(jump_rates_sum.device().is_cpu(), std::invalid_argument,
              "jump_rates_sum must reside on the CPU");
    auto out = torch::empty({static_cast<int64_t>(counts.size())},
                            torch::TensorOptions{}
                                .dtype(torch::kInt64)
                                .pinned_memory(!device.is_cpu()));

    auto const out_info = obtain_tensor_info<int64_t>(out);
    auto const sum_info = obtain_tensor_info<scalar_t const>(jump_rates_sum);
    auto const weights_info = obtain_tensor_info<scalar_t const>(jump_rates);
    auto       offset       = int64_t{0};
    for (auto i = int64_t{0}; i < out_info.size(); ++i) {
        auto const n = counts[static_cast<size_t>(i)];
        TCM_CHECK(n >= 0, std::runtime_error, "negative count");
        TCM_CHECK(offset + n <= weights_info.size(), std::runtime_error,
                  "sum of counts exceeds the size of jump_rates");
        out_info[i] = offset
                      + sample_one_from_multinomial(
                          slice(weights_info, offset, offset + n), sum_info[i]);
        offset += n;
    }
    TCM_CHECK(offset == weights_info.size(), std::runtime_error,
              "sum of counts is smaller than the size of jump_rates");

    if (!device.is_cpu()) {
        return out.to(out.options().device(device), /*non_blocking=*/true);
    }
    return out;
}

#if 0
TCM_EXPORT auto zanella_next_state_index(torch::Tensor               jump_rates,
                                         std::vector<int64_t> const& counts,
                                         c10::optional<torch::Tensor> out)
    -> torch::Tensor
{
    if (!out.has_value()) {
        out.emplace(torch::empty(
            std::initializer_list<int64_t>{static_cast<int64_t>(counts.size())},
            torch::TensorOptions{}
                .dtype(torch::kInt64)
                .device(jump_rates.device())));
    }
    auto& indices = *out;
    auto  offset  = 0L;
    for (auto i = 0L; i < static_cast<int64_t>(counts.size()); ++i) {
        auto const n = static_cast<int64_t>(counts[static_cast<size_t>(i)]);
        indices[i] =
            offset
            + torch::multinomial(torch::narrow(jump_rates, /*dim=*/0,
                                               /*start=*/offset, /*length=*/n),
                                 1)
                  .item<int64_t>();
        offset += n;
    }
    return indices;
}
#endif

#if 0
TCM_EXPORT auto zanella_jump_rates(torch::Tensor current_log_prob,
                                   torch::Tensor proposed_log_prob,
                                   std::vector<int64_t> const& counts,
                                   torch::Device const         target_device)
    -> std::tuple<torch::Tensor, torch::Tensor>
{
    torch::NoGradGuard no_grad;

    auto rates = torch::empty_like(proposed_log_prob);
    rates.copy_(proposed_log_prob);
    auto parts = torch::split_with_sizes(rates, counts);
    for (auto i = size_t{0}; i < counts.size(); ++i) {
        parts[i].sub_(current_log_prob[static_cast<int64_t>(i)]);
    }
    rates.exp_();
    rates.clamp_max_(1.0f);
    auto rates_target = rates.device() != target_device
                            ? rates.to(rates.options().device(target_device),
                                       /*non_blocking=*/true, /*copy=*/false)
                            : rates;

    auto rates_sum = torch::empty_like(current_log_prob);
    for (auto i = size_t{0}; i < counts.size(); ++i) {
        rates_sum[static_cast<int64_t>(i)] = parts[i].sum();
    }
    auto rates_sum_target =
        rates_sum.device() != target_device
            ? rates_sum.to(rates_sum.options().device(target_device),
                           /*non_blocking=*/true, /*copy=*/false)
            : rates_sum;

    return std::make_tuple(std::move(rates_target),
                           std::move(rates_sum_target));
}
#endif

TCM_NAMESPACE_END
