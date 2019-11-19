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

#pragma once

#include "config.hpp"
#include "errors.hpp"
#include "quantum_state.hpp"
#include "pool.hpp"

#include <boost/align/aligned_allocator.hpp>
#include <gsl/gsl-lite.hpp>
#include <vectorclass/version2/vectorclass.h>
#include <flat_hash_map/bytell_hash_map.hpp>
#include <taskflow/taskflow.hpp>

#include <array>
#include <chrono>
#include <type_traits>
#include <vector>

TCM_NAMESPACE_BEGIN

// Permutations {{{
namespace detail {

/// Rounds an integer up to the next power of 2. If \p is already a power of 2,
/// then \p x itself is returned.
///
/// https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
constexpr auto round_up_pow_2(unsigned x) noexcept -> unsigned
{
    x--;
    x |= x >> 1U;
    x |= x >> 2U;
    x |= x >> 4U;
    x |= x >> 8U;
    x |= x >> 16U;
    x++;
    return x;
}

static_assert(round_up_pow_2(1) == 1, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(round_up_pow_2(2) == 2, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(round_up_pow_2(3) == 4, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(round_up_pow_2(4) == 4, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(round_up_pow_2(5) == 8, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(round_up_pow_2(123412398) == 134217728,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(round_up_pow_2(32028753) == 33554432,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(round_up_pow_2(0xFFFFFFF) == 0x10000000,
              TCM_STATIC_ASSERT_BUG_MESSAGE);

/// Returns base-2 logarithm of \p x rounded down to the nearest integer.
///
/// https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
constexpr auto log2(unsigned x) noexcept -> unsigned
{
    // clang-format off
    unsigned r      = (x > 0xFFFF) << 4U; x >>= r;
    unsigned shift  = (x > 0xFF  ) << 3U; x >>= shift; r |= shift;
             shift  = (x > 0xF   ) << 2U; x >>= shift; r |= shift;
             shift  = (x > 0x3   ) << 1U; x >>= shift; r |= shift;
                                                       r |= (x >> 1U);
    // clang-format on
    return r;
}

static_assert(log2(1) == 0, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(log2(2) == 1, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(log2(3) == 1, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(log2(4) == 2, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(log2(5) == 2, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(log2(123498711) == 26, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(log2(419224229) == 28, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(log2(0xFFFFFFFF) == 31, TCM_STATIC_ASSERT_BUG_MESSAGE);

/// Performs one step of the Butterfly network. It exchanges bits with distance
/// \p d between them if the corresponding bits in the mask \p m are set.
template <class UInt, class = std::enable_if_t<std::is_integral<UInt>::value
                                               && !std::is_signed<UInt>::value>>
constexpr auto bit_permute_step(UInt const x, UInt const m, unsigned d) noexcept
    -> UInt
{
    auto const y = (x ^ (x >> d)) & m;
    return x ^ y ^ (y << d);
}

static_assert(bit_permute_step(5930UL, 272UL, 2U) == 5930UL,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(bit_permute_step(5930UL, 65UL, 1U) == 5929UL,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(bit_permute_step(56166UL, 2820UL, 4U) == 63846UL,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(bit_permute_step(13658UL, 242UL, 8U) == 22328UL,
              TCM_STATIC_ASSERT_BUG_MESSAGE);

/// Forward propagates `x` through a butterfly network specified by `masks`.
template <class UInt, size_t N>
__attribute__((optimize("unroll-loops"))) constexpr auto
bfly(UInt x, std::array<UInt, N> const& masks) noexcept -> UInt
{
    static_assert(N <= log2(std::numeric_limits<UInt>::digits),
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    for (auto i = 0U; i < N; ++i) {
        x = bit_permute_step(x, masks[i], 1U << i);
    }
    return x;
}

/// Backward propagates `x` through a butterfly network specified by `masks`.
template <class UInt, size_t N>
__attribute__((optimize("unroll-loops"))) constexpr auto
ibfly(UInt x, std::array<UInt, N> const& masks) noexcept -> UInt
{
    static_assert(N <= log2(std::numeric_limits<UInt>::digits),
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    for (auto i = static_cast<unsigned>(N);
         i-- > 0U;) { // Iterates backwards from N - 1 to 0
        x = bit_permute_step(x, masks[i], 1U << i);
    }
    return x;
}

template <class UInt> struct BenesNetwork {
    static_assert(std::is_integral<UInt>::value && !std::is_signed<UInt>::value,
                  "UInt must be an unsigned integral type.");

    using MasksT = std::array<UInt, log2(std::numeric_limits<UInt>::digits)>;
    MasksT fwd;
    MasksT bwd;

    constexpr auto operator()(UInt const x) const noexcept -> UInt
    {
        return ibfly(bfly(x, fwd), bwd);
    }
};

constexpr auto test_benes_network() noexcept -> int
{
    constexpr BenesNetwork<uint8_t> p1{{0, 0, 1}, {5, 1, 0}};
    static_assert(p1(0) == 0, TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(p1(182) == 171, TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(p1(255) == 255, TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(p1(254) == 239, TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(p1(101) == 114, TCM_STATIC_ASSERT_BUG_MESSAGE);

    constexpr BenesNetwork<uint32_t> p2{
        {1162937344U, 304095283U, 67502857U, 786593U, 17233U},
        {16793941U, 18882595U, 263168U, 18U, 0U}};
    static_assert(p2(1706703868U) == 2923286909U,
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(p2(384262095U) == 1188297710U, TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(p2(991361073U) == 1634567835U, TCM_STATIC_ASSERT_BUG_MESSAGE);

    return 0;
}

} // namespace detail

// }}}

// Symmetry {{{
struct Symmetry {
  public:
    using UInt = uint64_t;

  private:
    detail::BenesNetwork<UInt> _permute;
    unsigned                   _sector;
    unsigned                   _periodicity;

  public:
    constexpr Symmetry(detail::BenesNetwork<UInt> permute,
                       unsigned const sector, unsigned const periodicity)
        : _permute{permute}, _sector{sector}, _periodicity{periodicity}
    {
        TCM_CHECK(
            periodicity > 0, std::invalid_argument,
            fmt::format("invalid periodicity: {}; expected a positive integer",
                        periodicity));
        TCM_CHECK(
            sector < periodicity, std::invalid_argument,
            fmt::format("invalid sector: {}; expected an integer in [0, {})",
                        sector, periodicity));
    }

    constexpr Symmetry(Symmetry const&) noexcept = default;
    constexpr Symmetry(Symmetry&&) noexcept      = default;
    constexpr auto operator=(Symmetry const&) noexcept -> Symmetry& = default;
    constexpr auto operator=(Symmetry&&) noexcept -> Symmetry& = default;

    constexpr auto operator()(UInt const x) const noexcept -> UInt
    {
        return _permute(x);
    }
    constexpr auto sector() const noexcept { return _sector; }
    constexpr auto periodicity() const noexcept { return _periodicity; }
    constexpr auto phase() const noexcept -> double
    {
        return static_cast<double>(_sector) / static_cast<double>(_periodicity);
    }
    auto eigenvalue() const noexcept -> std::complex<double>
    {
        auto const arg = -2.0 * M_PI * phase();
        return std::complex<double>{std::cos(arg), std::sin(arg)};
    }

    constexpr auto _state_as_tuple() const noexcept
        -> std::tuple<typename detail::BenesNetwork<UInt>::MasksT,
                      typename detail::BenesNetwork<UInt>::MasksT, unsigned,
                      unsigned>
    {
        return {_permute.fwd, _permute.bwd, _sector, _periodicity};
    }
}; // }}}

namespace detail {

template <class Iterator, class T>
constexpr auto is_iterator_for() noexcept -> bool
{
    return std::is_same<
        std::remove_const_t<std::remove_reference_t<
            typename std::iterator_traits<Iterator>::reference>>,
        T>::value;
}

// find_representative {{{
/// Given a symmetry group, returns the representative state of orbit of \p x.
///
/// Representative state is defined as the smallest integer in the orbit.
template <class Iterator, class Sentinel,
          class = std::enable_if_t<is_iterator_for<Iterator, Symmetry>()
                                   && is_iterator_for<Sentinel, Symmetry>()>>
constexpr auto find_representative(Iterator first, Sentinel last,
                                   Symmetry::UInt const x) noexcept
    -> Symmetry::UInt
{
    auto repr = x;
    for (; first != last; ++first) {
        auto const y = (*first)(x);
        if (y < repr) { repr = y; }
    }
    return repr;
}

template auto find_representative(Symmetry const* first, Symmetry const* last,
                                  Symmetry::UInt x) noexcept -> Symmetry::UInt;
template auto find_representative(Symmetry* first, Symmetry* last,
                                  Symmetry::UInt x) noexcept -> Symmetry::UInt;
// }}}

// find_normalisation {{{
/// Given a symmetry group and a representative state \p x, finds the norm of
/// the basis element corresponding to \p x. If the group is empty, then 1 is
/// returned because there's effectively no basis transformation.
template <class Iterator, class Sentinel,
          class = std::enable_if_t<is_iterator_for<Iterator, Symmetry>()
                                   && is_iterator_for<Sentinel, Symmetry>()>>
constexpr auto find_normalisation(Iterator first, Sentinel last,
                                  Symmetry::UInt const x,
                                  std::true_type /*is representative?*/)
    -> double
{
    static_assert(
        std::is_same<typename std::iterator_traits<Iterator>::iterator_category,
                     std::random_access_iterator_tag>::value,
        TCM_STATIC_ASSERT_BUG_MESSAGE);
    if (first == last) { return 1.0; }
    auto const count = static_cast<unsigned>(std::distance(first, last));
    auto       norm  = 0.0;
    for (auto i = 0U; i < count; ++i, ++first) {
        auto const y = (*first)(x);
        TCM_ASSERT(y >= x, "x must be a representative state");
        if (y == x) {
            // We're actually interested in
            // std::conj(first->eigenvalue()).real(), but Re[z*] == Re[z].
            norm += first->eigenvalue().real();
        }
    }

    // We need to detect the case when norm is not zero, but only because of
    // inaccurate arithmetics. epsilon here is chosen somewhat arbitrarily...
    constexpr auto epsilon = 1.0e-5;
    if (std::abs(norm) <= epsilon) { norm = 0.0; }
    TCM_CHECK(
        norm >= 0.0, std::runtime_error,
        fmt::format("state {} appears to have negative squared norm {} :/", x,
                    norm));
    return std::sqrt(norm / static_cast<double>(count));
}
// }}}

// get_info {{{
template <class Iterator, class Sentinel,
          class = std::enable_if_t<is_iterator_for<Iterator, Symmetry>()
                                   && is_iterator_for<Sentinel, Symmetry>()>>
constexpr auto get_info(Iterator first, Sentinel last, Symmetry::UInt const x)
    -> std::tuple</*representative=*/Symmetry::UInt,
                  /*eigenvalue=*/std::complex<double>, /*norm=*/double>
{
    static_assert(
        std::is_same<typename std::iterator_traits<Iterator>::iterator_category,
                     std::random_access_iterator_tag>::value,
        TCM_STATIC_ASSERT_BUG_MESSAGE);
    if (first == last) {
        return std::make_tuple(x, std::complex<double>{1.0, 0.0}, 1.0);
    }
    constexpr auto almost_equal = [](double const a, double const b) {
        using std::max, std::abs;
        constexpr auto const epsilon = 1.0e-7;
        return abs(b - a) <= max(abs(a), abs(b)) * epsilon;
    };
    auto const count = static_cast<unsigned>(std::distance(first, last));
    auto       repr  = x;
    auto       phase = 0.0;
    auto       norm  = 0.0;
    for (auto i = 0U; i < count; ++i, ++first) {
        auto const y = (*first)(x);
        if (y == x) {
            // We're actually interested in
            // std::conj(first->eigenvalue()).real(), but Re[z*] == Re[z].
            norm += first->eigenvalue().real();
        }
        if (y < repr) {
            repr  = y;
            phase = first->phase();
        }
    }

    // We need to detect the case when norm is not zero, but only because of
    // inaccurate arithmetics
    constexpr auto epsilon = 1.0e-5;
    if (std::abs(norm) <= epsilon) { norm = 0.0; }
    TCM_CHECK(
        norm >= 0.0, std::runtime_error,
        fmt::format("state {} appears to have negative squared norm {} :/", x,
                    norm));
    norm = std::sqrt(norm / static_cast<double>(count));

#if defined(TCM_DEBUG) // This is a sanity check
    if (norm > 0.0) {
        for (first = last - static_cast<ptrdiff_t>(count); first != last;
             ++first) {
            auto const y = (*first)(x);
            if (y == repr) {
                TCM_CHECK(
                    almost_equal(first->phase(), phase), std::logic_error,
                    fmt::format("The result of a long discussion that gσ "
                                "= hσ => λ(g) = λ(h) is wrong: {} != {}, σ={}",
                                first->phase(), phase, y));
            }
        }
    }
#endif
    auto const arg = 2.0 * M_PI * phase;
    return std::make_tuple(
        repr, std::complex<double>{std::cos(arg), std::sin(arg)}, norm);
}
// }}}

// generate_states {{{
#if 0
template <class Iterator, class Sentinel,
          class = std::enable_if_t<is_iterator_for<Iterator, Symmetry>()
                                   && is_iterator_for<Sentinel, Symmetry>()>>
auto generate_states(Iterator first, Sentinel last, unsigned number_spins,
                     std::optional<unsigned> hamming_weight)
    -> std::vector<Symmetry::UInt>
{
    TCM_CHECK(0 < number_spins && number_spins <= 64, std::invalid_argument,
              fmt::format("invalid number of spins: {}; expected a "
                          "positive integer not greater than 64.",
                          number_spins));
    std::vector<Symmetry::UInt> states;
    auto const                  handle = [&states, first, last](auto const x) {
        auto const [repr, _, norm] = get_info(first, last, x);
        if (repr == x && norm > 0.0) { states.push_back(x); }
    };
    if (hamming_weight.has_value()) {
        TCM_CHECK(*hamming_weight <= number_spins, std::invalid_argument,
                  fmt::format("invalid hamming weight: {}; expected a "
                              "non-negative integer not greater than {}.",
                              *hamming_weight, number_spins));

        if (*hamming_weight == 0U) { return {uint64_t{0}}; }
        if (*hamming_weight == 64U) { return {~uint64_t{0}}; }

        auto       current = (~uint64_t{0}) >> (64U - *hamming_weight);
        auto const upper_bound =
            number_spins > *hamming_weight
                ? (current << (number_spins - *hamming_weight))
                : current;
        auto const next = [](uint64_t const v) {
            auto const t =
                v | (v - 1U); // t gets v's least significant 0 bits set to 1
            // Next set to 1 the most significant bit to change,
            // set to 0 the least significant ones, and add the necessary 1 bits.
            return (t + 1U)
                   | (((~t & -~t) - 1U)
                      >> (static_cast<unsigned>(__builtin_ctzl(v)) + 1U));
        };
        // TODO(twesterhout): This loop can be parallelised
        for (; current < upper_bound; current = next(current)) {
            handle(current);
        }
        TCM_ASSERT(current == upper_bound, "");
        handle(current);
        return states;
    }
    else {
        auto current     = uint64_t{0};
        auto upper_bound = number_spins == 64U
                               ? (~uint64_t{0})
                               : ((~uint64_t{0}) >> (64U - number_spins));
        // TODO(twesterhout): This loop can be parallelised
        for (; current < upper_bound; ++current) {
            handle(current);
        }
        TCM_ASSERT(current == upper_bound, "");
        handle(current);
        return states;
    }
}
#endif

auto global_executor() noexcept -> tf::Executor&;

// }}}


// BasisCache {{{
struct BasisCache {
    // TODO: write a proper allocator which will determine the page size at
    // runtime...
    template <class T>
    using BufferT = std::vector<T, boost::alignment::aligned_allocator<
                                       T, std::max<size_t>(4096, alignof(T))>>;

  public:
    using StatesT = BufferT<Symmetry::UInt>;
    using RangesT = BufferT<std::pair<uint64_t, uint64_t>>;

  private:
    static constexpr auto bits = 16U;

    StatesT _states;
    RangesT _ranges;

  public:
    inline BasisCache(gsl::span<Symmetry const> symmetries,
                      unsigned                  number_spins,
                      std::optional<unsigned>   hamming_weight);

    inline BasisCache(StatesT&& states, RangesT&& ranges);

    BasisCache(BasisCache const&)     = default;
    BasisCache(BasisCache&&) noexcept = default;
    BasisCache& operator=(BasisCache const&) = default;
    BasisCache& operator=(BasisCache&&) noexcept = default;

    inline auto states() const noexcept -> gsl::span<Symmetry::UInt const>;
    inline auto number_states() const noexcept -> uint64_t;
    inline auto index(Symmetry::UInt x, unsigned number_spins) const
        -> uint64_t;

    constexpr auto _get_state() const noexcept
        -> std::tuple<StatesT const&, RangesT const&>;
};
// }}}

// generate_states {{{
auto generate_states_parallel(gsl::span<Symmetry const>     symmetries,
                              unsigned const                number_spins,
                              std::optional<unsigned> const hamming_weight)
    -> BasisCache::StatesT;
// }}}

// generate_ranges {{{
template <
    unsigned Bits, class Iterator, class Sentinel,
    class = std::enable_if_t<is_iterator_for<Iterator, Symmetry::UInt>()
                             && is_iterator_for<Sentinel, Symmetry::UInt>()>>
auto generate_ranges(Iterator first, Sentinel last, unsigned number_spins)
    -> BasisCache::RangesT
{
    static_assert(0 < Bits && Bits <= 16U, TCM_STATIC_ASSERT_BUG_MESSAGE);
    constexpr auto      size  = 1U << Bits;
    constexpr auto      empty = std::make_pair(~uint64_t{0}, uint64_t{0});
    auto const          shift = number_spins > Bits ? number_spins - Bits : 0U;
    BasisCache::RangesT ranges;
    ranges.reserve(size);

    auto const begin = first;
    for (auto i = 0U; i < size; ++i) {
        auto element = empty;
        if (first != last && ((*first) >> shift) == i) {
            element.first = static_cast<uint64_t>((first++) - begin);
            ++element.second;
            while (((*first) >> shift) == i && first != last) {
                ++element.second;
                ++first;
            }
        }
        ranges.push_back(element);
    }

    return ranges;
}
// }}}

// BasisCache IMPLEMENTATION {{{
BasisCache::BasisCache(gsl::span<Symmetry const> symmetries,
                       unsigned const            number_spins,
                       std::optional<unsigned>   hamming_weight)
    : _states{generate_states_parallel(symmetries, number_spins,
                                       std::move(hamming_weight))}
    , _ranges{
          generate_ranges<bits>(_states.cbegin(), _states.cend(), number_spins)}
{}

BasisCache::BasisCache(StatesT&& states, RangesT&& ranges)
    : _states{std::move(states)}, _ranges{std::move(ranges)}
{}

auto BasisCache::states() const noexcept -> gsl::span<Symmetry::UInt const>
{
    return _states;
}

auto BasisCache::number_states() const noexcept -> uint64_t
{
    return _states.size();
}

auto BasisCache::index(Symmetry::UInt const x,
                       unsigned const       number_spins) const -> uint64_t
{
    TCM_ASSERT(number_spins <= 64U, "");
    using std::begin, std::end;
    auto const  shift = number_spins > bits ? number_spins - bits : 0U;
    auto const& range = _ranges[(x >> shift) & ((1U << bits) - 1U)];
    auto const  first = begin(_states) + static_cast<ptrdiff_t>(range.first);
    auto const  last  = first + static_cast<ptrdiff_t>(range.second);
    auto        i     = std::lower_bound(first, last, x);
    TCM_CHECK(
        i != last && *i == x, std::runtime_error,
        fmt::format("invalid state: {}; expected a basis representative", x));
    return static_cast<uint64_t>(i - begin(_states));
}

constexpr auto BasisCache::_get_state() const noexcept
    -> std::tuple<StatesT const&, RangesT const&>
{
    return {_states, _ranges};
}
// }}}

} // namespace detail

// SpinBasis {{{
class SpinBasis : public std::enable_shared_from_this<SpinBasis> {
  public:
    using UInt   = Symmetry::UInt;
    using StateT = UInt;

  private:
    std::vector<Symmetry>             _symmetries;
    unsigned                          _number_spins;
    std::optional<unsigned>           _hamming_weight;
    std::optional<detail::BasisCache> _cache;

  public:
    inline SpinBasis(std::vector<Symmetry> symmetries, unsigned number_spins,
                     std::optional<unsigned>           hamming_weight,
                     std::optional<detail::BasisCache> cache = std::nullopt);

    inline auto representative(StateT x) const noexcept -> StateT;
    inline auto normalisation(StateT x) const -> StateT;
    inline auto full_info(StateT x) const
        -> std::tuple<StateT, std::complex<double>, double>;
    inline auto    index(StateT const x) const -> uint64_t;
    constexpr auto number_spins() const noexcept -> unsigned;
    inline auto    number_states() const -> uint64_t;
    inline auto    states() const -> gsl::span<StateT const>;
    inline auto    build() -> void;

    constexpr auto _get_state() const noexcept
        -> std::tuple<std::vector<Symmetry> const&, unsigned,
                      std::optional<unsigned>,
                      std::optional<detail::BasisCache> const&>;
}; // }}}

// SpinBasis IMPLEMENTATION {{{
SpinBasis::SpinBasis(std::vector<Symmetry> symmetries, unsigned number_spins,
                     std::optional<unsigned>           hamming_weight,
                     std::optional<detail::BasisCache> cache)
    : _symmetries{std::move(symmetries)}
    , _number_spins{number_spins}
    , _hamming_weight{std::move(hamming_weight)}
    , _cache{std::move(cache)}
{
    TCM_CHECK(0 < _number_spins && _number_spins <= 64, std::invalid_argument,
              fmt::format("invalid number_spins: {}; expected a "
                          "positive integer not greater than 64.",
                          _number_spins));
    if (_hamming_weight.has_value()) {
        TCM_CHECK(*_hamming_weight <= _number_spins, std::invalid_argument,
                  fmt::format("invalid hamming_weight: {}; expected a "
                              "non-negative integer not greater than {}.",
                              *_hamming_weight, _number_spins));
    }
}

auto SpinBasis::normalisation(StateT x) const -> StateT
{
    using std::begin, std::end;
    TCM_CHECK(representative(x) == x, std::runtime_error,
              fmt::format("invalid state {}; expected a representative", x));
    return detail::find_normalisation(begin(_symmetries), end(_symmetries), x,
                                      std::true_type{});
}

auto SpinBasis::representative(StateT const x) const noexcept -> StateT
{
    using std::begin, std::end;
    return detail::find_representative(begin(_symmetries), end(_symmetries), x);
}

auto SpinBasis::full_info(StateT const x) const
    -> std::tuple<StateT, std::complex<double>, double>
{
    using std::begin, std::end;
    return detail::get_info(begin(_symmetries), end(_symmetries), x);
}

auto SpinBasis::index(StateT const x) const -> uint64_t
{
    TCM_CHECK(_cache.has_value(), std::runtime_error,
              "cache must be initialised before calling `index()`; use "
              "`build()` member function to initialise the cache.");
    return _cache->index(x, number_spins());
}

auto SpinBasis::build() -> void
{
    if (!_cache.has_value()) {
        _cache.emplace(gsl::span<Symmetry const>{_symmetries}, _number_spins,
                       _hamming_weight);
    }
}

constexpr auto SpinBasis::number_spins() const noexcept -> unsigned
{
    return _number_spins;
}

auto SpinBasis::number_states() const -> uint64_t
{
    TCM_CHECK(_cache.has_value(), std::runtime_error,
              "cache must be initialised before calling `number_states()`; "
              "use `build()` member function to initialise the cache.");
    return _cache->number_states();
}

auto SpinBasis::states() const -> gsl::span<StateT const>
{
    TCM_CHECK(_cache.has_value(), std::runtime_error,
              "cache must be initialised before calling `states()`; use "
              "`build()` member function to initialise the cache.");
    return _cache->states();
}


constexpr auto SpinBasis::_get_state() const noexcept
    -> std::tuple<std::vector<Symmetry> const&, unsigned,
                  std::optional<unsigned>,
                  std::optional<detail::BasisCache> const&>
{
    return {_symmetries, _number_spins, _hamming_weight, _cache};
}
// }}}

namespace v2 {

// unpack {{{
namespace detail {
    struct _IdentityProjection {
        template <class T> constexpr decltype(auto) operator()(T&& x) const noexcept
        {
            return std::forward<T>(x);
        }
    };

    TCM_FORCEINLINE auto _unpack(uint8_t const bits) noexcept -> vcl::Vec8f
    {
        auto const one = vcl::Vec8f{1.0f}; // 1.0f == 0x3f800000
        auto const two = vcl::Vec8f{2.0f};
        // Adding 0x3f800000 to select ensures that we're working with valid
        // floats rather than denormals
        auto const select = vcl::Vec8f{vcl::reinterpret_f(vcl::Vec8i{
            0x3f800000 + (1 << 0), 0x3f800000 + (1 << 1), 0x3f800000 + (1 << 2),
            0x3f800000 + (1 << 3), 0x3f800000 + (1 << 4), 0x3f800000 + (1 << 5),
            0x3f800000 + (1 << 6), 0x3f800000 + (1 << 7)})};
        auto       broadcasted =
            vcl::Vec8f{vcl::reinterpret_f(vcl::Vec8i{static_cast<int>(bits)})};
        broadcasted |= one;
        broadcasted &= select;
        broadcasted = broadcasted == select;
        broadcasted &= two;
        broadcasted -= one;
        return broadcasted;
    }

    template <bool Unsafe>
    TCM_FORCEINLINE auto _unpack(SpinBasis::StateT x,
                                 unsigned const    number_spins,
                                 float*            out) TCM_NOEXCEPT -> float*
    {
        auto const chunks = number_spins / 8U;
        auto const rest   = number_spins % 8U;
        auto const y      = x; // Only for testing
        for (auto i = 0U; i < chunks; ++i, out += 8, x >>= 8U) {
            _unpack(static_cast<uint8_t>(x & 0xFF)).store(out);
        }
        if (rest != 0) {
            auto const t = _unpack(static_cast<uint8_t>(x & 0xFF));
            if constexpr (Unsafe) { t.store(out); }
            else {
                t.store_partial(static_cast<int>(rest), out);
            }
            out += rest;
        }

        TCM_ASSERT(
            ([y, number_spins, out]() {
                auto* p = out - static_cast<ptrdiff_t>(number_spins);
                for (auto i = 0U; i < number_spins; ++i) {
                    if (!((p[i] == 1.0f && ((y >> i) & 0x01) == 0x01)
                          || (p[i] == -1.0f && ((y >> i) & 0x01) == 0x00))) {
                        return false;
                    }
                }
                return true;
            }()),
            noexcept_format(
                "{} vs [{}]", y,
                fmt::join(out - static_cast<ptrdiff_t>(number_spins), out,
                          ", ")));
        return out;
    }
} // namespace detail

template <class Iterator, class Sentinel,
          class Projection = detail::_IdentityProjection>
TCM_NOINLINE auto unpack(Iterator first, Sentinel last,
                         unsigned const number_spins, torch::Tensor dst,
                         Projection proj = Projection{}) -> void
{
    if (first == last) { return; }
    auto const size = static_cast<size_t>(std::distance(first, last));
    TCM_ASSERT(dst.dim() == 2,
               noexcept_format("invalid dimension {}", dst.dim()));
    TCM_ASSERT(size == static_cast<size_t>(dst.size(0)),
               noexcept_format("sizes don't match: size={}, dst.size(0)={}",
                               size, dst.size(0)));
    TCM_ASSERT(
        number_spins == static_cast<size_t>(dst.size(1)),
        noexcept_format("sizes don't match: number_spins={}, dst.size(1)={}",
                        number_spins, dst.size(1)));
    TCM_ASSERT(dst.is_contiguous(), "Output tensor must be contiguous");

    auto const rest = number_spins % 8;
    auto const tail = std::min<size_t>(
        ((8U - rest) + number_spins - 1U) / number_spins, size);
    auto* data = dst.data_ptr<float>();
    for (auto i = size_t{0}; i < size - tail; ++i, ++first) {
        data =
            detail::_unpack</*Unsafe=*/true>(proj(*first), number_spins, data);
    }
    for (auto i = size - tail; i < size; ++i, ++first) {
        data =
            detail::_unpack</*Unsafe=*/false>(proj(*first), number_spins, data);
    }
}
// }}}

template <class T, class = void> struct is_complex : std::false_type {};
template <class T>
struct is_complex<std::complex<T>,
                  std::enable_if_t<std::is_floating_point<T>::value>>
    : std::true_type {};

template <class T> inline constexpr bool is_complex_v = is_complex<T>::value;

#if 1
class Heisenberg : public std::enable_shared_from_this<Heisenberg> {
  public:
    using real_type    = double;
    using complex_type = std::complex<real_type>;
    using edge_type    = std::tuple<complex_type, uint16_t, uint16_t>;
    using spec_type =
        std::vector<edge_type,
                    boost::alignment::aligned_allocator<edge_type, 64>>;

    struct pool_deleter_type {
        Pool* pool;

        auto operator()(void* p) const noexcept { pool->free(p); }
    };

    using buffer_type =
        std::unique_ptr<std::pair<SpinBasis::StateT, complex_type>[],
                        pool_deleter_type>;

  private:
    spec_type                        _edges; ///< Graph edges
    std::shared_ptr<SpinBasis const> _basis;
    unsigned _max_index; ///< The greatest site index present in `_edges`.
                         ///< It is used to detect errors when one tries to
                         ///< apply the hamiltonian to a spin configuration
                         ///< which is too short.
    bool         _is_real;
    mutable Pool _pool;

  public:
    /// Constructs a hamiltonian given graph edges and couplings.
    Heisenberg(spec_type edges, std::shared_ptr<SpinBasis const> basis);

    /// Copy and Move constructors/assignments
    Heisenberg(Heisenberg const&)     = default;
    Heisenberg(Heisenberg&&) noexcept = default;
    Heisenberg& operator=(Heisenberg const&) = default;
    Heisenberg& operator=(Heisenberg&&) noexcept = default;

    /// Returns the number of edges in the graph
    /*constexpr*/ auto size() const noexcept -> size_t { return _edges.size(); }

    /// Returns the greatest index encountered in `_edges`.
    ///
    /// \precondition `size() != 0`
    /*constexpr*/ auto max_index() const noexcept -> size_t
    {
        TCM_ASSERT(!_edges.empty(), "_max_index is not defined");
        return _max_index;
    }

    constexpr auto is_real() const noexcept -> bool { return _is_real; }

    /// Returns a *reference* to graph edges.
    /*constexpr*/ auto edges() const noexcept -> gsl::span<edge_type const>
    {
        return _edges;
    }

    auto basis() const noexcept -> std::shared_ptr<SpinBasis const>
    {
        return _basis;
    }

    auto operator()(SpinBasis::StateT spin) const
        -> std::pair<buffer_type, size_t>;

    template <class Callback>
    auto operator()(SpinBasis::StateT const spin, Callback&& callback) const
        -> void
    {
        TCM_ASSERT(_edges.empty() || max_index() < _basis->number_spins(),
                   fmt::format("`spin` is too short {}; expected >{}",
                               _basis->number_spins(), max_index()));
        auto const norm = std::get<2>(_basis->full_info(spin));
        TCM_CHECK(norm > 0.0, std::runtime_error,
                  fmt::format("state {} does not belong to the basis", spin));
        auto coeff = complex_type{0, 0};
        for (auto const [coupling, first, second] : edges()) {
            // Heisenberg hamiltonian works more or less like this:
            //
            //     K|↑↑⟩ = J|↑↑⟩
            //     K|↓↓⟩ = J|↓↓⟩
            //     K|↑↓⟩ = -J|↑↓⟩ + 2J|↓↑⟩
            //     K|↓↑⟩ = -J|↓↑⟩ + 2J|↑↓⟩
            //
            // where K is the "kernel". We want to perform
            // |ψ⟩ += c * K|σᵢσⱼ⟩ for each edge (i, j).
            //
            auto const not_aligned =
                ((spin >> first) ^ (spin >> second)) & 0x01;
            if (not_aligned) {
                coeff -= coupling;
                auto [branch_spin, branch_eigenvalue, branch_norm] =
                    _basis->full_info(
                        spin
                        ^ ((uint64_t{1} << first) | (uint64_t{1} << second)));
                if (branch_norm > 0.0) {
                    callback(branch_spin, 2.0 * coupling * branch_norm / norm
                                              * branch_eigenvalue);
                }
            }
            else {
                coeff += coupling;
            }
        }
        callback(spin, coeff);
    }

    template <class T, class = std::enable_if_t<
                           std::is_floating_point_v<T> || is_complex_v<T>>>
    auto operator()(gsl::span<T const> x, gsl::span<T> y) const -> void
    {
        TCM_CHECK(x.size() == y.size() && y.size() == _basis->number_states(),
                  std::invalid_argument,
                  fmt::format(
                      "vectors have invalid sizes: {0}, {1}; expected {2}, {2}",
                      x.size(), y.size(), _basis->number_spins()));
        if constexpr (!is_complex<T>::value) {
            TCM_CHECK(_is_real, std::runtime_error,
                      "cannot apply a complex-valued Hamiltonian to a "
                      "real-valued vector");
        }
        auto&      executor = ::TCM_NAMESPACE::detail::global_executor();
        auto const states   = _basis->states();
        auto const chunk_size =
            std::max(500UL, states.size() / (20UL * executor.num_workers()));

        struct alignas(64) Task {
            Heisenberg const&     self;
            T const*              x_p;
            T*                    y_p;
            Symmetry::UInt const* states_p;

            auto operator()(uint64_t const j) const -> void
            {
                auto acc = T{0};
                self(states_p[j],
                     [&acc, this](auto const spin, auto const coeff) {
                         if constexpr (is_complex<T>::value) {
                             acc += static_cast<T>(std::conj(coeff))
                                    * x_p[self._basis->index(spin)];
                         }
                         else {
                             acc += static_cast<T>(coeff.real())
                                    * x_p[self._basis->index(spin)];
                         }
                     });
                y_p[j] = acc;
            }
        } task{*this, x.data(), y.data(), states.data()};

        tf::Taskflow taskflow;
        taskflow.parallel_for(uint64_t{0}, y.size(), uint64_t{1},
                              std::cref(task), chunk_size);
        executor.run(taskflow).wait();
    }

  private:
    inline auto get_buffer() const -> buffer_type;
};
#endif

using QuantumState    = detail::QuantumState<SpinBasis::StateT>;
using Polynomial      = detail::Polynomial<SpinBasis::StateT, Heisenberg>;
using PolynomialState = detail::PolynomialState<SpinBasis::StateT, Heisenberg>;

} // namespace v2

TCM_NAMESPACE_END

