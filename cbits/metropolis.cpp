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
#include "random.hpp"
#include "spin_basis.hpp"

TCM_NAMESPACE_BEGIN

namespace detail {

template <class... Ints>
constexpr auto flipped(uint64_t const spin, Ints const... is) noexcept
    -> uint64_t
{
    return spin ^ (... | (uint64_t{1} << is));
}

constexpr auto find_nth_set(uint64_t const v, unsigned r) noexcept -> unsigned
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

} // namespace detail

TCM_EXPORT
MetropolisKernel::MetropolisKernel(std::shared_ptr<SpinBasis const> basis,
                                   RandomGenerator& generator) noexcept
    : _basis{std::move(basis)}, _generator{std::addressof(generator)}
{}

TCM_EXPORT auto MetropolisKernel::operator()(torch::Tensor const& x,
                                             bool pin_memory) const
    -> std::tuple<torch::Tensor, torch::Tensor>
{
    TCM_CHECK(x.dim(), std::invalid_argument,
              fmt::format("x has wrong number of dimensions: {}; expected "
                          "a one-dimensional tensor",
                          x.dim()));
    TCM_CHECK_CONTIGUOUS("x", x);
    TCM_CHECK(x.device().type() == torch::DeviceType::CPU,
              std::invalid_argument, fmt::format("x must reside on the CPU"));
    auto const n = x.numel();
    auto       y = torch::empty(
        {n},
        torch::TensorOptions{}.dtype(torch::kInt64).pinned_memory(pin_memory));
    auto norm = torch::empty({n}, torch::TensorOptions{}
                                      .dtype(torch::kFloat32)
                                      .pinned_memory(pin_memory));
    kernel_cpu(
        static_cast<size_t>(n), reinterpret_cast<uint64_t const*>(x.data_ptr()),
        reinterpret_cast<uint64_t*>(y.data_ptr()), norm.data_ptr<float>());
    return {std::move(y), std::move(norm)};
}

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

TCM_NAMESPACE_END
