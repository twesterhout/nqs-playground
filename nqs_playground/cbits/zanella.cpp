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
#include "hedley.h"
#include <lattice_symmetries/lattice_symmetries.h>
#include <omp.h>
#include <sstream>

auto operator==(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool
{
    for (auto i = 0; i < static_cast<int>(std::size(x.words)); ++i) {
        if (x.words[i] != y.words[i]) { return false; }
    }
    return true;
}

auto operator!=(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool { return !(x == y); }

auto operator<(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool
{
    for (auto i = 0; i < static_cast<int>(std::size(x.words)); ++i) {
        if (x.words[i] < y.words[i]) { return true; }
        if (x.words[i] > y.words[i]) { return false; }
    }
    return false;
}

auto operator>(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool { return y < x; }

auto operator<=(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool { return !(x > y); }

auto operator>=(ls_bits512 const& x, ls_bits512 const& y) noexcept -> bool { return !(x < y); }

namespace tcm {

constexpr auto toggle_bit(uint64_t& bits, unsigned const i) noexcept -> void
{
    // TCM_ASSERT(i < 64U, "index out of bounds");
    bits ^= uint64_t{1} << uint64_t{i};
}
constexpr auto toggle_bit(ls_bits512& bits, unsigned const i) noexcept -> void
{
    // TCM_ASSERT(i < 512U, "index out of bounds");
    return toggle_bit(bits.words[i / 64U], i % 64U);
}
constexpr auto test_bit(uint64_t const bits, unsigned const i) noexcept -> bool
{
    // TCM_ASSERT(i < 64U, "index out of bounds");
    return static_cast<bool>((bits >> i) & 1U);
}
constexpr auto test_bit(ls_bits512 const& bits, unsigned const i) noexcept -> bool
{
    // TCM_ASSERT(i < 512U, "index out of bounds");
    return test_bit(bits.words[i / 64U], i % 64U);
}
constexpr auto set_zero(uint64_t& bits) noexcept -> void { bits = 0UL; }
constexpr auto set_zero(ls_bits512& bits) noexcept -> void
{
    for (auto& w : bits.words) {
        set_zero(w);
    }
}

HEDLEY_PUBLIC ZanellaGenerator::ZanellaGenerator(ls_spin_basis const&                       basis,
                                                 std::vector<std::pair<unsigned, unsigned>> edges)
    : _basis{ls_copy_spin_basis(&basis)}, _edges{std::move(edges)}
{
    if (_edges.empty()) { throw std::invalid_argument{"'edges' list must not be empty"}; }
    auto const n = ls_get_number_spins(_basis);
    for (auto const [i, j] : _edges) {
        if (i == j || i >= n || j >= n) {
            std::ostringstream msg;
            msg << "'edges' list constains an invalid edge: (" << i << ", " << j << ")";
            throw std::invalid_argument{msg.str()};
        }
    }
}

HEDLEY_PUBLIC ZanellaGenerator::~ZanellaGenerator()
{
    ls_destroy_spin_basis(const_cast<ls_spin_basis*>(static_cast<ls_spin_basis const*>(_basis)));
}

HEDLEY_PUBLIC auto ZanellaGenerator::max_states() const noexcept -> uint64_t
{
    auto const number_spins   = ls_get_number_spins(_basis);
    auto const hamming_weight = ls_get_hamming_weight(_basis);
    if (hamming_weight != -1) {
        auto const m = static_cast<unsigned>(hamming_weight);
        return m * (number_spins - m);
    }
    // By construction, number_spins > 1
    return number_spins * (number_spins - 1);
}

HEDLEY_PUBLIC auto ZanellaGenerator::operator()(torch::Tensor x) const
    -> std::tuple<torch::Tensor, torch::Tensor>
{
    auto       pin_memory = false;
    auto const device     = x.device();
    auto const batch_size = x.size(0);
    if (batch_size == 0) { throw std::invalid_argument{"expected 'x' to contain a least one row"}; }
    if (device.type() != torch::DeviceType::CPU) {
        pin_memory = true;
        x          = x.to(x.options().device(torch::DeviceType::CPU),
                 /*non_blocking=*/true);
    }
    // TCM_CHECK(ls_get_hamming_weight(_basis) != -1, std::runtime_error,
    //           "ZanellaGenerator currently only supports bases with fixed magnetisation");
    auto const max_possible_states = static_cast<int64_t>(max_states());
    auto const options = torch::TensorOptions{}.dtype(torch::kInt64).pinned_memory(pin_memory);
    auto       y =
        torch::zeros(std::initializer_list<int64_t>{batch_size, max_possible_states, 8L}, options);
    auto counts = torch::empty(std::initializer_list<int64_t>{batch_size}, options);

    // auto const x_info      = tensor_info<ls_bits512 const>(x, "x");
    // auto const y_info      = tensor_info<ls_bits512, 2>(y);
    // auto const counts_info = tensor_info<int64_t>(counts);
    // auto const outer_num_threads = std::min<int>(x_info.size(), omp_get_max_threads());
    // auto const inner_num_threads = omp_get_max_threads() / outer_num_threads;
    auto const* x_ptr      = reinterpret_cast<ls_bits512 const*>(x.data_ptr<int64_t>());
    auto*       y_ptr      = reinterpret_cast<ls_bits512*>(y.data_ptr<int64_t>());
    auto*       counts_ptr = counts.data_ptr<int64_t>();
#pragma omp parallel for
    for (auto i = int64_t{0}; i < batch_size; ++i) {
        auto* dst     = y_ptr + max_possible_states * i;
        counts_ptr[i] = project(dst, generate(x_ptr[i], dst));
    }
    auto const size = *std::max_element(counts_ptr, counts_ptr + batch_size);

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

auto ZanellaGenerator::generate(ls_bits512 const& spin, ls_bits512* out) const -> int64_t
{
    auto const number_spins = ls_get_number_spins(_basis);
    auto       count        = int64_t{0};
    for (auto const [i, j] : _edges) {
        if (test_bit(spin, i) != test_bit(spin, j)) {
            auto possible = spin;
            toggle_bit(possible, i);
            toggle_bit(possible, j);
            out[count++] = possible;
        }
    }
    if (count == 0) {
        throw std::runtime_error{"ZanellaGenerator got stuck: all potential states lie in a "
                                 "different magnetization sector"};
    }
    return count;
}

auto ZanellaGenerator::project(ls_bits512* spins, int64_t const count) const -> int64_t
{
    ls_bits512 repr;
    set_zero(repr);
    auto offset = int64_t{0};
    for (auto i = int64_t{0}; i < count; ++i) {
        std::complex<double> _dummy;
        double               norm;
        ls_get_state_info(_basis, spins + i, &repr, &_dummy, &norm);
        if (norm > 0) {
            spins[offset] = repr;
            ++offset;
        }
    }
    if (offset == 0) {
        throw std::runtime_error{"ZanellaGenerator got stuck: all potential states have norm 0"};
    }
    std::sort(spins, spins + offset);
    return std::unique(spins, spins + offset) - spins;
}

} // namespace tcm
