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

#include "symmetry.hpp"

#include <optional>
#include <type_traits>
#include <vector>

TCM_NAMESPACE_BEGIN

// BasisBase {{{
class TCM_EXPORT BasisBase : public std::enable_shared_from_this<BasisBase> {
  protected:
    unsigned                _number_spins;
    std::optional<unsigned> _hamming_weight;
    bool                    _has_symmetries;

  public:
    BasisBase(unsigned number_spins, std::optional<unsigned> hamming_weight,
              bool has_symmetries);
    BasisBase(BasisBase const&)     = delete;
    BasisBase(BasisBase&&) noexcept = delete;
    auto operator=(BasisBase const&) -> BasisBase& = delete;
    auto operator=(BasisBase &&) -> BasisBase& = delete;

    constexpr auto number_spins() const noexcept -> unsigned;
    constexpr auto hamming_weight() const noexcept -> std::optional<unsigned>;
    constexpr auto has_symmetries() const noexcept -> bool;

    virtual auto full_info(bits512 const& x) const
        -> std::tuple<bits512, std::complex<double>, double> = 0;
    virtual auto is_real() const noexcept -> bool            = 0;

    virtual ~BasisBase() = default;
};

constexpr auto BasisBase::number_spins() const noexcept -> unsigned
{
    return _number_spins;
}

constexpr auto BasisBase::hamming_weight() const noexcept
    -> std::optional<unsigned>
{
    return _hamming_weight;
}

constexpr auto BasisBase::has_symmetries() const noexcept -> bool
{
    return _has_symmetries;
}
// }}}

// SmallSpinBasis {{{

namespace detail {
struct BasisCache;
} // namespace detail

class TCM_EXPORT SmallSpinBasis : public BasisBase {
  public:
    using UInt      = uint64_t;
    using StateT    = UInt;
    using SymmetryT = v2::Symmetry<64>;
    using _PickleStateT =
        std::tuple<unsigned, std::optional<unsigned>, std::vector<SymmetryT>,
                   std::vector<uint64_t>>;

    struct Alternative {
        std::vector<Symmetry8x64>     _chunks;
        std::vector<v2::Symmetry<64>> _rest;
    };

  private:
    std::vector<SymmetryT>              _symmetries;
    std::unique_ptr<detail::BasisCache> _cache;
    Alternative                         _alternative;

  public:
    SmallSpinBasis(std::vector<SymmetryT> symmetries, unsigned number_spins,
                   std::optional<unsigned> hamming_weight);

    SmallSpinBasis(std::vector<SymmetryT> symmetries, unsigned number_spins,
                   std::optional<unsigned>             hamming_weight,
                   std::unique_ptr<detail::BasisCache> _unsafe_cache);

    SmallSpinBasis(SmallSpinBasis const&)     = delete;
    SmallSpinBasis(SmallSpinBasis&&) noexcept = delete;
    auto operator=(SmallSpinBasis const&) -> SmallSpinBasis& = delete;
    auto operator=(SmallSpinBasis &&) -> SmallSpinBasis& = delete;

    // We actually want the desctructor to be implicitly defined, but then
    // the definition of BasisCache should be available. So we defer this step.
    ~SmallSpinBasis() override;

    auto full_info(uint64_t x) const
        -> std::tuple<uint64_t, std::complex<double>, double>;

    auto full_info(bits512 const& x) const
        -> std::tuple<bits512, std::complex<double>, double> override;

    auto is_real() const noexcept -> bool override;

    auto build() -> void;
    auto states() const -> gsl::span<StateT const>;
    auto number_states() const -> uint64_t;
    auto index(StateT const x) const -> uint64_t;

    auto        _internal_state() const -> _PickleStateT;
    static auto _from_internal_state(_PickleStateT const&)
        -> std::shared_ptr<SmallSpinBasis>;
}; // }}}

// BigSpinBasis {{{
class TCM_EXPORT BigSpinBasis : public BasisBase {
  public:
    using StateT    = bits512;
    using SymmetryT = v2::Symmetry<512>;

  private:
    std::vector<SymmetryT> _symmetries;

  public:
    BigSpinBasis(std::vector<SymmetryT> symmetries, unsigned number_spins,
                 std::optional<unsigned> hamming_weight);

    BigSpinBasis(BigSpinBasis const&) = delete;
    BigSpinBasis(BigSpinBasis&&)      = delete;
    auto operator=(BigSpinBasis const&) -> BigSpinBasis& = delete;
    auto operator=(BigSpinBasis &&) -> BigSpinBasis& = delete;

    ~BigSpinBasis() override;

    auto full_info(bits512 const& x) const
        -> std::tuple<bits512, std::complex<double>, double> override;

    auto is_real() const noexcept -> bool override;
}; // }}}

TCM_NAMESPACE_END
