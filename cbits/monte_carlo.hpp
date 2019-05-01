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

#include "nn.hpp"
#include "polynomial.hpp"
#include "random.hpp"
#include "spin.hpp"

#include <ska_sort/ska_sort.hpp>
#include <omp.h>

TCM_NAMESPACE_BEGIN

// ---------------------------- [RandomFlipper] ---------------------------- {{{
/// \brief A random walker (which preserves magnetisation) in the space of spin
/// configurations.
///
/// At each step one obtains the proposed spin flips using #read() function. To
/// move forward one then calls #next(). This process can be repeated
/// indefinitely (if your random number generator is good).
class RandomFlipper {
  public:
    using index_type = unsigned;

    static constexpr index_type number_flips = 2;

    using value_type = std::array<index_type, number_flips>;

  private:
    std::vector<index_type>         _storage;
    gsl::span<index_type>           _ups;
    gsl::span<index_type>           _downs;
    gsl::not_null<RandomGenerator*> _generator;
    index_type                      _i;

  public:
    explicit RandomFlipper(
        SpinVector       initial_spin,
        RandomGenerator& generator = global_random_generator());

    RandomFlipper(RandomFlipper const&) = delete;
    // TODO(twesterhout): Force this to be noexcept even before C++17?
    RandomFlipper(RandomFlipper&&)      = default;
    RandomFlipper& operator=(RandomFlipper const&) = delete;
    // TODO(twesterhout): Force this to be noexcept even before C++17?
    RandomFlipper& operator=(RandomFlipper&&) = default;

    inline auto read() const noexcept -> value_type;
    inline auto next(bool accepted) -> void;

  private:
    auto        shuffle() -> void;
    inline auto swap_accepted() noexcept -> void;
};

auto RandomFlipper::read() const noexcept -> value_type
{
    using std::begin;
    using std::end;
    constexpr auto n = number_flips / 2;
    TCM_ASSERT(_i + n <= _ups.size() && _i + n <= _downs.size(),
               "Index out of bounds");
    value_type proposed;
    std::copy(begin(_ups) + _i, begin(_ups) + _i + n, begin(proposed));
    std::copy(begin(_downs) + _i, begin(_downs) + _i + n, begin(proposed) + n);
    return proposed;
}

auto RandomFlipper::next(bool const accepted) -> void
{
    constexpr auto n = number_flips / 2;
    TCM_ASSERT(_i + n <= _ups.size() && _i + n <= _downs.size(),
               "Index out of bounds");
    if (accepted) { swap_accepted(); }
    _i += n;
    if (_i + n > _ups.size() || _i + n > _downs.size()) {
        shuffle();
        _i = 0;
    }
}

auto RandomFlipper::swap_accepted() noexcept -> void
{
    constexpr auto n = number_flips / 2;
    TCM_ASSERT(_i + n <= _ups.size() && _i + n <= _downs.size(),
               "Index out of bounds");
    for (auto i = _i; i < _i + n; ++i) {
        std::swap(_ups[i], _downs[i]);
    }
}
// ---------------------------- [RandomFlipper] ---------------------------- }}}

// ----------------------------- [ChainState] ------------------------------ {{{
struct alignas(32) ChainState {
    SpinVector spin;  ///< Spin configuration σ
    real_type  value; ///< Wave function ψ(σ)
    size_t     count; ///< Number of times this state has been visited

    static_assert(std::is_trivially_copyable<SpinVector>::value,
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(std::is_trivially_copyable<real_type>::value,
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(std::is_trivially_copyable<size_t>::value,
                  TCM_STATIC_ASSERT_BUG_MESSAGE);

    // Simple constructor to make `emplace` happy.
    constexpr ChainState(SpinVector s, real_type v, size_t n) noexcept;
    constexpr ChainState(ChainState const&) noexcept = default;
    constexpr ChainState(ChainState&&) noexcept      = default;
    constexpr ChainState& operator=(ChainState const&) noexcept = default;
    constexpr ChainState& operator=(ChainState&) noexcept = default;

    static constexpr auto magic_count() noexcept -> size_t;
    static constexpr auto magic() noexcept -> ChainState;
    static constexpr auto struct_format() noexcept -> char const*;

    /// Merges `other` into this. This amounts to just adding together the
    /// `count`s. In DEBUG mode, however, we also make sure that only states
    /// with the same `spin` and `value` attributes can be merged.
    inline auto merge(ChainState const& other) TCM_NOEXCEPT -> void;
};

static_assert(sizeof(ChainState) == 32, TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(std::is_trivially_copy_constructible<ChainState>::value,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(std::is_trivially_copy_assignable<ChainState>::value,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(std::is_trivially_move_constructible<ChainState>::value,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(std::is_trivially_move_assignable<ChainState>::value,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
static_assert(std::is_trivially_destructible<ChainState>::value,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
// TODO(twesterhout): This fails on Clang... why??
#if !defined(TCM_CLANG)
static_assert(std::is_trivially_copyable<ChainState>::value,
              TCM_STATIC_ASSERT_BUG_MESSAGE);
#endif

constexpr ChainState::ChainState(SpinVector s, real_type v, size_t n) noexcept
    : spin{s}, value{v}, count{n}
{}

constexpr auto ChainState::magic_count() noexcept -> size_t
{
    // We choose a random value between (2^64 - 1)/32 and (2^64 - 1). It
    // cannot arise under normal circumstances (i.e. from Monte Carlo
    // sampling) due to the lower bound.
    constexpr auto value = 0xdcf0e37cf3630f77;
    static_assert(value
                      > std::numeric_limits<size_t>::max() / sizeof(ChainState),
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(value < std::numeric_limits<size_t>::max(),
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    return value;
}

constexpr auto ChainState::magic() noexcept -> ChainState
{
    return ChainState{SpinVector{}, real_type{0}, magic_count()};
}

constexpr auto ChainState::struct_format() noexcept -> char const*
{
    static_assert(sizeof(SpinVector) == 8 * sizeof(short),
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(sizeof(size_t) == 8, TCM_STATIC_ASSERT_BUG_MESSAGE);
    return "8HdQ";
}

/// Merges `other` into this. This amounts to just adding together the
/// `count`s. In DEBUG mode, however, we also make sure that only states
/// with the same `spin` and `value` attributes can be merged.
auto ChainState::merge(ChainState const& other) TCM_NOEXCEPT -> void
{
    auto const isclose = [](auto const a, auto const b) noexcept->bool
    {
        using std::abs;
        using std::max;
        constexpr auto atol = 1.0E-5;
        constexpr auto rtol = 1.0E-3;
        return abs(a - b) <= atol + rtol * max(abs(a), abs(b));
    };
    TCM_ASSERT(spin == other.spin,
               "only states with the same spin can be merged");
    TCM_ASSERT(
        isclose(value, other.value),
        fmt::format("Different forward passes with the same input should "
                    "produce the same results: {} != {}",
                    value, other.value));
    count += other.count;
}

#define TCM_MAKE_OPERATOR_USING_KEY(op)                                        \
    inline constexpr auto operator op(ChainState const& x,                     \
                                      ChainState const& y) TCM_NOEXCEPT->bool  \
    {                                                                          \
        TCM_ASSERT(x.spin.size() == y.spin.size(),                             \
                   "States corresponding to different system sizes can't be "  \
                   "compared");                                                \
        TCM_ASSERT(x.spin.size() <= 64,                                        \
                   "Longer spin chains are not (yet) supported");              \
        return x.spin.key(unsafe_tag) op y.spin.key(unsafe_tag);               \
    }

TCM_MAKE_OPERATOR_USING_KEY(==)
TCM_MAKE_OPERATOR_USING_KEY(!=)
TCM_MAKE_OPERATOR_USING_KEY(<)
TCM_MAKE_OPERATOR_USING_KEY(>)
TCM_MAKE_OPERATOR_USING_KEY(<=)
TCM_MAKE_OPERATOR_USING_KEY(>=)

#undef TCM_MAKE_OPERATOR_USING_KEY
// ----------------------------- [ChainState] ------------------------------ }}}

// ----------------------------- [ChainResult] ----------------------------- {{{
struct ChainResult : public std::enable_shared_from_this<ChainResult> {
    using SamplesT =
        std::vector<ChainState, boost::alignment::aligned_allocator<
                                    ChainState, alignof(ChainState)> /**/>;

  private:
    SamplesT _samples;

  public:
    /// \precondition `samples` must be sorted!
    inline ChainResult(SamplesT samples) noexcept;
    ChainResult() noexcept              = default;
    ChainResult(ChainResult const&)     = delete;
    ChainResult(ChainResult&&) noexcept = default;
    ChainResult& operator=(ChainResult const&) = delete;
    ChainResult& operator=(ChainResult&&) noexcept = default;

    inline auto number_spins() const noexcept -> size_t;
    inline auto size() const noexcept -> size_t;
    inline auto empty() const noexcept -> bool;

    constexpr auto samples() const & noexcept -> SamplesT const&;
    inline auto    samples() && noexcept -> SamplesT;

    auto vectors() const -> torch::Tensor;
    auto values() -> torch::Tensor;
    auto counts() const -> torch::Tensor;
    auto values(torch::Tensor const&) -> void;

    auto buffer_info() -> pybind11::buffer_info;
    auto auto_shrink() -> void;
};

ChainResult::ChainResult(SamplesT samples) noexcept
    : _samples{std::move(samples)}
{}

auto ChainResult::number_spins() const noexcept -> size_t
{
    return _samples.empty() ? 0 : _samples[0].spin.size();
}

auto ChainResult::size() const noexcept -> size_t { return _samples.size(); }
auto ChainResult::empty() const noexcept -> bool { return _samples.empty(); }

constexpr auto ChainResult::samples() const & noexcept -> SamplesT const&
{
    return _samples;
}

auto ChainResult::samples() && noexcept -> SamplesT
{
    return std::move(_samples);
}

auto merge(ChainResult const& x, ChainResult const& y) -> ChainResult;
auto merge(std::vector<ChainResult>&& results) -> ChainResult;
// ----------------------------- [ChainResult] ----------------------------- }}}

// ----------------------------- [MarkovChain] ----------------------------- {{{
template <class ForwardFn, class ProbabilityFn> class MarkovChain {
    using SamplesT = ChainResult::SamplesT;

  private:
    SamplesT                        _samples;
    RandomFlipper                   _flipper;
    ForwardFn                       _forward;
    ProbabilityFn                   _prob;
    gsl::not_null<RandomGenerator*> _generator;
    size_t                          _accepted;
    size_t                          _count;

  public:
    MarkovChain(ForwardFn forward, ProbabilityFn prob, SpinVector spin,
                RandomGenerator& generator, bool record_first = false);

    MarkovChain(MarkovChain const&) = default;
    MarkovChain(MarkovChain&&)      = default;
    MarkovChain& operator=(MarkovChain const&) = default;
    MarkovChain& operator=(MarkovChain&&) = default;

    inline auto next() -> void;
    inline auto skip() -> void;
    inline auto release(bool sorted = true) && -> SamplesT;

    constexpr auto steps() const noexcept -> size_t;

  private:
    auto current() const noexcept -> ChainState const&
    {
        TCM_ASSERT(!_samples.empty(), "Use after `release`");
        return _samples.back();
    }

    auto current() noexcept -> ChainState&
    {
        TCM_ASSERT(!_samples.empty(), "Use after `release`");
        return _samples.back();
    }

    auto next_impl(bool record) -> void;
    auto sort() -> void;
    auto compress() -> void;
};

template <class ForwardFn, class ProbabilityFn>
MarkovChain<ForwardFn, ProbabilityFn>::MarkovChain(ForwardFn        forward,
                                                   ProbabilityFn    prob,
                                                   SpinVector const spin,
                                                   RandomGenerator& generator,
                                                   bool record_first)
    : _samples{}
    , _flipper{spin, generator}
    , _forward{std::move(forward)}
    , _prob{std::move(prob)}
    , _generator{std::addressof(generator)}
    , _accepted{0}
    , _count{0}
{
    TCM_CHECK(spin.size() <= 64, std::runtime_error,
              fmt::format("Sorry, but such long spin chains ({}) are not (yet) "
                          "supported. The greatest supported length is {}",
                          spin.size(), 64));
    _samples.emplace_back(spin, _forward(spin), record_first);
}

template <class ForwardFn, class ProbabilityFn>
auto MarkovChain<ForwardFn, ProbabilityFn>::next() -> void
{
    next_impl(true);
}

template <class ForwardFn, class ProbabilityFn>
auto MarkovChain<ForwardFn, ProbabilityFn>::skip() -> void
{
    next_impl(false);
}

template <class ForwardFn, class ProbabilityFn>
constexpr auto MarkovChain<ForwardFn, ProbabilityFn>::steps() const noexcept
    -> size_t
{
    return _count;
}

template <class ForwardFn, class ProbabilityFn>
TCM_NOINLINE auto MarkovChain<ForwardFn, ProbabilityFn>::next_impl(bool record)
    -> void
{
    TCM_CHECK(!_samples.empty(), std::logic_error, "Use after `release()`");
    auto const u = std::generate_canonical<
        real_type, std::numeric_limits<real_type>::digits>(*_generator);
    auto spin = current().spin.flipped(_flipper.read());
    // _forward returns a float and we increase the precision which is safe
    auto const value       = static_cast<real_type>(_forward(spin));
    auto const probability = _prob(current().value, value);

    _count += record;
    if (u <= probability) {
        if (current().count > 0) { _samples.emplace_back(spin, value, record); }
        else {
            current() = ChainState{spin, value, record};
        }
        _accepted += record;
        _flipper.next(true);
    }
    else {
        current().count += record;
        _flipper.next(false);
    }
}

template <class ForwardFn, class ProbabilityFn>
TCM_NOINLINE auto MarkovChain<ForwardFn, ProbabilityFn>::sort() -> void
{
    TCM_ASSERT(!_samples.empty(), "Use after `release()`");
    auto const number_spins = current().spin.size();
    TCM_ASSERT(number_spins <= 64, "Spin chain too long");
    ska_sort(std::begin(_samples), std::end(_samples), [](auto const& x) {
        return x.spin.key(
            unsafe_tag /*Yes, we have checked that size() <= 64*/);
    });
}

template <class ForwardFn, class ProbabilityFn>
auto MarkovChain<ForwardFn, ProbabilityFn>::compress() -> void
{
    TCM_ASSERT(!_samples.empty(), "Use after `release()`");
    auto const pred = [](auto const& x, auto const& y) {
        return x.spin == y.spin;
    };
    auto const merge = [](auto& acc, auto&& x) { acc.merge(std::move(x)); };
    auto const first = ::TCM_NAMESPACE::compress(
        std::begin(_samples), std::end(_samples), pred, merge);
    _samples.erase(first, std::end(_samples));
}

template <class ForwardFn, class ProbabilityFn>
auto MarkovChain<ForwardFn, ProbabilityFn>::release(bool sorted) && -> SamplesT
{
    TCM_ASSERT(!_samples.empty(), "use after `release`");
    if (current().count == 0) {
        _samples.pop_back();
        if (_samples.empty()) { return {}; }
    }
    TCM_ASSERT(std::all_of(std::begin(_samples), std::end(_samples),
                           [](auto const& x) { return x.count > 0; }),
               "");
    if (sorted) {
        sort();
        compress();
    }
    return std::move(_samples);
}
// ----------------------------- [MarkovChain] ----------------------------- }}}

struct Options {
    unsigned number_spins;
    int      magnetisation;
    unsigned batch_size;
    /// [number_chains, begin, end, step]
    std::array<unsigned, 4> steps;
};

struct DefaultProbFn {
    using RealT = real_type;

    auto operator()(RealT current, RealT suggested) const noexcept -> RealT
    {
        current   = current * current;
        suggested = suggested * suggested;
        if (current <= suggested) return real_type{1};
        return suggested / current;
    }
};

// ----------------------------- [sample_some] ----------------------------- {{{
namespace detail {
template <class Function, class = void> struct FunctionWrapperHelper;

template <class Function>
struct FunctionWrapperHelper<
    Function,
    std::enable_if_t<std::is_lvalue_reference<Function>::value> /**/> {
    using type         = std::remove_reference_t<Function>;
    using wrapper_type = std::reference_wrapper<type>;

    TCM_FORCEINLINE constexpr auto operator()(type& x) noexcept -> wrapper_type
    {
        return std::ref(x);
    }
};

template <class Function>
struct FunctionWrapperHelper<
    Function,
    std::enable_if_t<std::is_rvalue_reference<Function>::value> /**/> {
    using type         = std::remove_const_t<std::remove_reference_t<Function>>;
    using wrapper_type = type;

    TCM_FORCEINLINE constexpr auto operator()(type&& x) noexcept -> wrapper_type
    {
        static_assert(std::is_nothrow_move_constructible<wrapper_type>::value,
                      TCM_STATIC_ASSERT_BUG_MESSAGE);
        return static_cast<type&&>(x);
    }
};
} // namespace detail

template <class ForwardFn>
auto sample_some(ForwardFn&& psi, Options const& options,
                 optional<SpinVector> initial_spin = nullopt,
                 RandomGenerator*     gen          = nullptr) -> ChainResult
{
    auto const begin     = options.steps[1];
    auto const end       = options.steps[2];
    auto const step      = options.steps[3];
    auto&      generator = (gen != nullptr) ? *gen : global_random_generator();
    auto const spin =
        initial_spin.has_value()
            ? *initial_spin
            : SpinVector::random(options.number_spins, options.magnetisation,
                                 generator);
    if (begin == end) { return {}; }

    using Wrapper = detail::FunctionWrapperHelper<ForwardFn&&>;
    MarkovChain<typename Wrapper::wrapper_type, DefaultProbFn> chain{
        /*forward=*/Wrapper{}(std::forward<ForwardFn>(psi)),
        /*probability=*/DefaultProbFn{},
        /*spin=*/spin, /*generator=*/generator, /*record_first=*/begin == 0};
    // If begin == 0, we start recording immediately, so the initial state of
    // the chain is the first sample. Hence in that case we initialise i to 1.
    auto i = static_cast<unsigned>(begin == 0u);
    if (begin != 0u && begin + step < end) {
        for (; i < begin; ++i) {
            chain.skip();
        }
        chain.next();
    }
    for (; i + step < end; i += step) {
        for (auto j_skip = 0u; j_skip < step - 1; ++j_skip) {
            chain.skip();
        }
        chain.next();
    }
    return ChainResult{std::move(chain).release(/*sorted=*/true)};
}

auto sample_some(std::string const& filename, Polynomial const& polynomial,
                 Options const& options, int num_threads = -1) -> ChainResult;

auto sample_some(CombiningState const& psi, Polynomial const& polynomial,
                 Options const& options, int num_threads = -1) -> ChainResult;

auto sample_difference(std::string const& new_state_filename,
                       std::string const& old_state_filename,
                       Polynomial const& polynomial, Options const& options,
                       int num_threads = -1) -> ChainResult;

auto sample_difference(CombiningState const& new_psi,
                       CombiningState const& old_psi,
                       Polynomial const& polynomial, Options const& options,
                       int num_threads = -1) -> ChainResult;

auto sample_amplitude_difference(std::string const& new_state_filename,
                                 std::string const& old_state_filename,
                                 Polynomial const&  polynomial,
                                 Options const& options, int num_threads = -1)
    -> ChainResult;

auto sample_amplitude_difference(AmplitudeNet const&   new_psi,
                                 CombiningState const& old_psi,
                                 Polynomial const&     polynomial,
                                 Options const& options, int num_threads = -1)
    -> ChainResult;
// ----------------------------- [sample_some] ----------------------------- }}}

auto bind_options(pybind11::module) -> void;
auto bind_chain_result(pybind11::module) -> void;
auto bind_sampling(pybind11::module) -> void;

TCM_NAMESPACE_END

namespace fmt {
template <> struct formatter<::TCM_NAMESPACE::Options> {
    template <typename ParseContext> constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(::TCM_NAMESPACE::Options const& options, FormatContext& ctx)
    {
        return format_to(
            ctx.out(),
            "Options(number_spins={}, magnetisation={}, steps=({}, {}, {}))",
            options.number_spins, options.magnetisation,
            std::get<0>(options.steps), std::get<1>(options.steps),
            std::get<2>(options.steps));
    }
};
} // namespace fmt
