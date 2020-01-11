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
#include "symmetry.hpp"

#include <boost/align/aligned_allocator.hpp>
#include <gsl/gsl-lite.hpp>

#include <array>
#include <type_traits>
#include <vector>

TCM_NAMESPACE_BEGIN

namespace detail {

/// Checks whether Iterator is an iterator over objects of type T.
template <class Iterator, class T>
constexpr auto is_iterator_for() noexcept -> bool
{
    return std::is_same<
        std::remove_const_t<std::remove_reference_t<
            typename std::iterator_traits<Iterator>::reference>>,
        T>::value;
}

#if 0
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
#endif

#if 0
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
#endif

} // namespace detail

namespace detail {
struct BasisCache;
} // namespace detail

// SpinBasis {{{
class TCM_IMPORT SpinBasis : public std::enable_shared_from_this<SpinBasis> {
  public:
    using UInt   = Symmetry::UInt;
    using StateT = UInt;

  private:
    std::vector<Symmetry>               _symmetries;
    unsigned                            _number_spins;
    std::optional<unsigned>             _hamming_weight;
    std::unique_ptr<detail::BasisCache> _cache;

  public:
    SpinBasis(std::vector<Symmetry> symmetries, unsigned number_spins,
              std::optional<unsigned> hamming_weight);

    SpinBasis(SpinBasis const&)     = delete;
    SpinBasis(SpinBasis&&) noexcept = delete;
    auto operator=(SpinBasis const&) -> SpinBasis& = delete;
    auto operator=(SpinBasis &&) -> SpinBasis& = delete;

    // We actually want the desctructor to be implicitly defined, but then
    // the definition of BasisCache should be available. So we defer this step.
    ~SpinBasis();

    // inline auto representative(StateT x) const noexcept -> StateT;
    // inline auto normalisation(StateT x) const -> StateT;
    auto full_info(StateT x) const
        -> std::tuple<StateT, std::complex<double>, double>;

    constexpr auto number_spins() const noexcept -> unsigned;
    constexpr auto hamming_weight() const noexcept -> std::optional<unsigned>;

    auto is_real() const noexcept -> bool;
    auto build() -> void;
    auto states() const -> gsl::span<StateT const>;
    auto number_states() const -> uint64_t;
    auto index(StateT const x) const -> uint64_t;

#if 0
    constexpr auto _get_state() const noexcept
        -> std::tuple<std::vector<Symmetry> const&, unsigned,
                      std::optional<unsigned>,
                      std::optional<detail::BasisCache> const&>;
#endif
}; // }}}

// SpinBasis IMPLEMENTATION {{{

#if 0
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
#endif

constexpr auto SpinBasis::number_spins() const noexcept -> unsigned
{
    return _number_spins;
}

constexpr auto SpinBasis::hamming_weight() const noexcept
    -> std::optional<unsigned>
{
    return _hamming_weight;
}

#if 0
constexpr auto SpinBasis::_get_state() const noexcept
    -> std::tuple<std::vector<Symmetry> const&, unsigned,
                  std::optional<unsigned>,
                  std::optional<detail::BasisCache> const&>
{
    return {_symmetries, _number_spins, _hamming_weight, _cache};
}
#endif
// }}}

TCM_IMPORT auto expand_states(SpinBasis const& basis, torch::Tensor src,
                              torch::Tensor states) -> torch::Tensor;

TCM_NAMESPACE_END
