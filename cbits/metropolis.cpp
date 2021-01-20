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

TCM_NAMESPACE_BEGIN

namespace {
constexpr auto find_nth_one_simple(uint64_t const v, unsigned r) noexcept -> unsigned
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
    auto t = static_cast<unsigned>((d + (d >> 16)) & 0xFF); // Bit count temporary.
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

auto find_nth_one(ls_bits512 const& x, unsigned n) noexcept -> unsigned
{
    static_assert(std::is_same_v<unsigned long, uint64_t>, TCM_STATIC_ASSERT_BUG_MESSAGE);
    for (auto i = 0U;; ++i) {
        if (i == std::size(x.words)) { return 512U; }
        auto const count = static_cast<unsigned>(__builtin_popcountl(x.words[i]));
        if (n <= count) { return 64U * i + find_nth_one(x.words[i], n); }
        n -= count;
    }
}

auto find_nth_zero(ls_bits512 const& x, unsigned const n) noexcept -> unsigned
{
    using std::begin, std::end;
    ls_bits512 y;
    std::transform(begin(x.words), end(x.words), begin(y.words), [](auto const w) { return ~w; });
    return find_nth_one(y, n);
}

auto check_basis(ls_spin_basis const& basis) -> ls_spin_basis const&
{
    TCM_CHECK(ls_get_number_spins(&basis) > 1, std::invalid_argument,
              "'ZanellaGenerator' uses binary flips, system size must thus be at least 2");
    return basis;
}
} // namespace

TCM_EXPORT MetropolisGenerator::MetropolisGenerator(ls_spin_basis const& basis,
                                                    RandomGenerator&     generator)
    : _basis{ls_copy_spin_basis(&check_basis(basis))}, _generator{&generator}
{}

TCM_EXPORT MetropolisGenerator::~MetropolisGenerator()
{
    ls_destroy_spin_basis(const_cast<ls_spin_basis*>(static_cast<ls_spin_basis const*>(_basis)));
}

TCM_EXPORT auto MetropolisGenerator::operator()(torch::Tensor x, c10::ScalarType dtype) const
    -> std::tuple<torch::Tensor, torch::Tensor>
{
    auto       pin_memory = false;
    auto const device     = x.device();
    if (device.type() != torch::DeviceType::CPU) {
        pin_memory = true;
        x          = x.cpu();
    }
    auto x_info = tensor_info<ls_bits512 const>(x, "x");
    auto y      = torch::empty(std::initializer_list<int64_t>{x_info.size(), 8L},
                          torch::TensorOptions{}.dtype(torch::kInt64).pinned_memory(pin_memory));
    auto norm   = torch::empty(std::initializer_list<int64_t>{x_info.size()},
                             torch::TensorOptions{}.dtype(dtype).pinned_memory(pin_memory));
    AT_DISPATCH_FLOATING_TYPES(dtype, "MetropolisGenerator::operator()", [&] {
        generate(x_info, tensor_info<ls_bits512>(y), tensor_info<scalar_t>(norm));
    });
    if (device.type() != torch::DeviceType::CPU) {
        y    = y.to(y.options().device(device), /*non_blocking=*/pin_memory);
        norm = norm.to(norm.options().device(device), /*non_blocking=*/pin_memory);
    }
    return {std::move(y), std::move(norm)};
}

template <class scalar_t>
auto MetropolisGenerator::generate(TensorInfo<ls_bits512 const> src, TensorInfo<ls_bits512> dst,
                                   TensorInfo<scalar_t> norm) const -> void
{
    auto const loop = [&](auto&& task) {
        for (auto i = int64_t{0}; i < src.size(); ++i) {
            std::tie(dst[i], norm[i]) = task(src[i]);
        }
    };
    auto const number_spins   = ls_get_number_spins(_basis);
    auto const hamming_weight = ls_get_hamming_weight(_basis);
    if (hamming_weight != -1) {
        loop([this, number_spins,
              hamming_weight](ls_bits512 const& x) -> std::tuple<ls_bits512, scalar_t> {
            for (;;) {
                auto const i = find_nth_one(
                    x, random_bounded(static_cast<unsigned>(hamming_weight), *_generator));
                auto const j = find_nth_zero(
                    x, random_bounded(number_spins - static_cast<unsigned>(hamming_weight),
                                      *_generator));
                ls_bits512 y = x;
                toggle_bit(y, i);
                toggle_bit(y, j);
                ls_bits512 repr;
                set_zero(repr);
                std::complex<double> character;
                double               _norm;
                ls_get_state_info(_basis, &y, &repr, &character, &_norm);
                if (_norm > 0.0 && repr != x) { return {repr, static_cast<scalar_t>(_norm)}; }
            }
        });
    }
    else {
        loop([this, number_spins](ls_bits512 const& x) -> std::tuple<ls_bits512, scalar_t> {
            for (;;) {
                auto const i = random_bounded(number_spins, *_generator);
                ls_bits512 y = x;
                toggle_bit(y, i);
                ls_bits512 repr;
                set_zero(repr);
                std::complex<double> character;
                double               _norm;
                ls_get_state_info(_basis, &y, &repr, &character, &_norm);
                if (_norm > 0.0) { return {repr, static_cast<scalar_t>(_norm)}; }
            }
        });
    }
}

TCM_NAMESPACE_END
