#include "spin_basis.hpp"
#include "common.hpp"
#include "parallel.hpp"

#include <boost/align/aligned_allocator.hpp>
#include <forward_list>

TCM_NAMESPACE_BEGIN

namespace detail {

// BasisCache {{{
struct BasisCache {
    // TODO: write a proper allocator which will determine the page size at
    // runtime...
    template <class T>
    using BufferT = std::vector<T, boost::alignment::aligned_allocator<
                                       T, std::max<size_t>(4096, alignof(T))>>;

  public:
    using StatesT = BufferT<uint64_t>;
    using RangesT = BufferT<std::pair<uint64_t, uint64_t>>;

  private:
    static constexpr auto bits = 16U;

    StatesT _states;
    RangesT _ranges;

  public:
    inline BasisCache(gsl::span<v2::Symmetry<64> const> symmetries,
                      unsigned                          number_spins,
                      std::optional<unsigned>           hamming_weight,
                      std::vector<uint64_t>             _unsafe_states = {});

#if 0
    inline BasisCache(StatesT&& states, RangesT&& ranges);
#endif

    BasisCache(BasisCache const&)     = default;
    BasisCache(BasisCache&&) noexcept = default;
    BasisCache& operator=(BasisCache const&) = default;
    BasisCache& operator=(BasisCache&&) noexcept = default;

    inline auto states() const noexcept -> gsl::span<uint64_t const>;
    inline auto number_states() const noexcept -> uint64_t;
    inline auto index(uint64_t x, unsigned number_spins) const -> uint64_t;
};
// }}}

// generate_states {{{
/// Returns the closest integer to x with the specified hamming weight
inline auto closest_hamming(uint64_t x, int const hamming_weight) noexcept
    -> uint64_t
{
    TCM_ASSERT(0 <= hamming_weight && hamming_weight <= 64,
               "invalid hamming weight");
    auto const weight = __builtin_popcountl(x);
    if (weight > hamming_weight) {
        auto mask = ~uint64_t{0};
        do {
            mask <<= 1;
            x &= mask;
        } while (__builtin_popcountl(x) > hamming_weight);
    }
    else if (weight < hamming_weight) {
        auto mask = uint64_t{1};
        x |= mask;
        while (__builtin_popcountl(x) < hamming_weight) {
            mask <<= 1;
            x |= mask;
        }
    }
    return x;
}

template <bool FixedHammingWeight> struct GenerateStatesTask {
    uint64_t                          current;
    uint64_t                          upper_bound;
    gsl::span<v2::Symmetry<64> const> symmetries;
    std::vector<uint64_t>*            states;

    GenerateStatesTask(GenerateStatesTask const&) noexcept = default;
    GenerateStatesTask(GenerateStatesTask&&) noexcept      = default;
    GenerateStatesTask& operator=(GenerateStatesTask const&) noexcept = default;
    GenerateStatesTask& operator=(GenerateStatesTask&&) noexcept = default;

    auto operator()() -> void
    {
        if constexpr (FixedHammingWeight) {
            TCM_ASSERT(
                __builtin_popcountl(current)
                    == __builtin_popcountl(upper_bound),
                "current and upper_bound must have the same Hamming weight");
        }
        for (; current < upper_bound; current = next(current)) {
            handle(current);
        }
        TCM_ASSERT(current == upper_bound, "");
        handle(current);
    }

    static auto next(uint64_t const v) noexcept -> uint64_t
    {
        if constexpr (FixedHammingWeight) {
            auto const t =
                v | (v - 1U); // t gets v's least significant 0 bits set to 1
            // Next set to 1 the most significant bit to change,
            // set to 0 the least significant ones, and add the necessary 1 bits.
            return (t + 1U)
                   | (((~t & -~t) - 1U)
                      >> (static_cast<unsigned>(__builtin_ctzl(v)) + 1U));
        }
        else {
            return v + 1;
        }
    }

  private:
    auto handle(uint64_t const x) -> void
    {
        using std::begin, std::end;
        auto const [repr, _, norm] = full_info(symmetries, x);
        if (repr == x && norm > 0.0) { states->push_back(x); }
    }
};

template <bool FixedHammingWeight, class Callback>
auto split_into_tasks(
    uint64_t current, uint64_t const bound, unsigned const number_chunks,
    Callback&& callback) noexcept(noexcept(std::
                                               declval<Callback&>()(
                                                   std::declval<uint64_t>(),
                                                   std::declval<uint64_t>())))
    -> void
{
    TCM_ASSERT(current <= bound, "invalid range");
    TCM_ASSERT(0 < number_chunks, "invalid number of chunks");
    if constexpr (FixedHammingWeight) {
        TCM_ASSERT(__builtin_popcountl(current) == __builtin_popcountl(bound),
                   "current and bound must have the same Hamming weight");
    }
    auto const hamming_weight = __builtin_popcount(current);
    auto const chunk_size =
        std::max(uint64_t{1}, (bound - current) / number_chunks);
    for (;;) {
        if (bound - current <= chunk_size) {
            callback(current, bound);
            break;
        }
        auto const next =
            FixedHammingWeight
                ? closest_hamming(current + chunk_size, hamming_weight)
                : current + chunk_size;
        TCM_ASSERT(next >= current, "");
        if (next >= bound) {
            callback(current, bound);
            break;
        }
        callback(current, next);
        current = GenerateStatesTask<FixedHammingWeight>::next(next);
    }
}

inline auto generate_states(gsl::span<v2::Symmetry<64> const> symmetries,
                            unsigned const                    number_spins,
                            std::optional<unsigned> const     hamming_weight)
    -> BasisCache::StatesT
{
    TCM_CHECK(0 < number_spins && number_spins <= 64, std::invalid_argument,
              fmt::format("invalid number of spins: {}; expected a "
                          "positive integer not greater than 64.",
                          number_spins));

    auto const number_chunks =
        20U * static_cast<unsigned>(omp_get_max_threads());

    omp_task_handler task_handler;
    // We want to make sure that references to states remain valid, so we can't
    // use vector here.
    std::forward_list<std::vector<uint64_t>> states;
    if (hamming_weight.has_value()) {
        TCM_CHECK(*hamming_weight <= number_spins, std::invalid_argument,
                  fmt::format("invalid hamming weight: {}; expected a "
                              "non-negative integer not greater than {}.",
                              *hamming_weight, number_spins));
        if (*hamming_weight == 0U) { return {uint64_t{0}}; }
        if (*hamming_weight == 64U) { return {~uint64_t{0}}; }
        auto const current = (~uint64_t{0}) >> (64U - *hamming_weight);
        auto const bound   = number_spins > *hamming_weight
                                 ? (current << (number_spins - *hamming_weight))
                                 : current;
#pragma omp parallel default(none) firstprivate(current, bound, number_chunks) \
    shared(symmetries, states, task_handler)
        {
#pragma omp single nowait
            {
                split_into_tasks<true>(
                    current, bound, number_chunks,
                    [&symmetries, &states, &task_handler](auto const first,
                                                          auto const last) {
                        auto& chunk = states.emplace_front();
                        chunk.reserve(1048576UL / sizeof(uint64_t));
                        task_handler.submit(GenerateStatesTask<true>{
                            first, last, symmetries, std::addressof(chunk)});
                    });
            }
        }
    }
    else {
        auto current = uint64_t{0};
        auto bound   = number_spins == 64U
                           ? (~uint64_t{0})
                           : ((~uint64_t{0}) >> (64U - number_spins));
#pragma omp parallel default(none) firstprivate(current, bound, number_chunks) \
    shared(symmetries, states, task_handler)
        {
#pragma omp single nowait
            {
                split_into_tasks<false>(
                    current, bound, number_chunks,
                    [&symmetries, &states, &task_handler](auto const first,
                                                          auto const last) {
                        auto& chunk = states.emplace_front();
                        chunk.reserve(1048576UL / sizeof(uint64_t));
                        task_handler.submit(GenerateStatesTask<false>{
                            first, last, symmetries, std::addressof(chunk)});
                    });
            }
        }
    }

    task_handler.check_errors();
    states.reverse();
    auto r = BasisCache::StatesT{};
    r.reserve(std::accumulate(
        std::begin(states), std::end(states), size_t{0},
        [](auto acc, auto const& x) { return acc + x.size(); }));
    std::for_each(std::begin(states), std::end(states), [&r](auto& x) {
        r.insert(std::end(r), std::begin(x), std::end(x));
    });
    return r;
}
// }}}

// generate_ranges {{{
template <unsigned Bits, class Iterator, class Sentinel>
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
BasisCache::BasisCache(gsl::span<v2::Symmetry<64> const> symmetries,
                       unsigned const                    number_spins,
                       std::optional<unsigned>           hamming_weight,
                       std::vector<uint64_t>             _unsafe_states)
    : _states{_unsafe_states.empty() ? generate_states(
                  symmetries, number_spins, std::move(hamming_weight))
                                     : StatesT{std::begin(_unsafe_states),
                                               std::end(_unsafe_states)}}
    , _ranges{
          generate_ranges<bits>(_states.cbegin(), _states.cend(), number_spins)}
{}

#if 0
BasisCache::BasisCache(StatesT&& states, RangesT&& ranges)
    : _states{std::move(states)}, _ranges{std::move(ranges)}
{}
#endif

auto BasisCache::states() const noexcept -> gsl::span<uint64_t const>
{
    return _states;
}

auto BasisCache::number_states() const noexcept -> uint64_t
{
    return _states.size();
}

auto BasisCache::index(uint64_t const x, unsigned const number_spins) const
    -> uint64_t
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
// }}}

} // namespace detail

// BasisBase {{{
TCM_EXPORT
BasisBase::BasisBase(unsigned const                number_spins,
                     std::optional<unsigned> const hamming_weight,
                     bool const                    has_symmetries)
    : _number_spins{number_spins}
    , _hamming_weight{hamming_weight}
    , _has_symmetries{has_symmetries}
{
    TCM_CHECK(
        0 < _number_spins, std::invalid_argument,
        fmt::format("invalid number_spins: {}; expected a positive integer",
                    _number_spins));
    if (_hamming_weight.has_value()) {
        TCM_CHECK(*_hamming_weight <= _number_spins, std::invalid_argument,
                  fmt::format("invalid hamming_weight: {}; expected a "
                              "non-negative integer not greater than {}.",
                              *_hamming_weight, _number_spins));
    }
}
// }}}

// SmallSpinBasis {{{
TCM_EXPORT
SmallSpinBasis::SmallSpinBasis(std::vector<v2::Symmetry<64>> symmetries,
                               unsigned                      number_spins,
                               std::optional<unsigned>       hamming_weight)
    : SmallSpinBasis{std::move(symmetries), number_spins,
                     std::move(hamming_weight), nullptr}
{}

TCM_EXPORT
SmallSpinBasis::SmallSpinBasis(
    std::vector<v2::Symmetry<64>> symmetries, unsigned number_spins,
    std::optional<unsigned>             hamming_weight,
    std::unique_ptr<detail::BasisCache> _unsafe_cache)
    : BasisBase{number_spins, hamming_weight, !symmetries.empty()}
    , _symmetries{std::move(symmetries)}
    , _cache{std::move(_unsafe_cache)}
{
    TCM_CHECK(_number_spins <= 64, std::invalid_argument,
              fmt::format("invalid number_spins: {}; use SpinBasis for systems "
                          "larger than 64 spins",
                          _number_spins));
    auto const number_chunks = _symmetries.size() / 8U;
    auto const remaining     = _symmetries.size() % 8U;
    auto       i             = size_t{0};
    for (auto j = uint64_t{0}; j < number_chunks; ++j, i += 8) {
        _alternative._chunks.emplace_back(
            gsl::span<v2::Symmetry<64> const>{_symmetries.data() + i, 8U});
    }
    for (auto j = uint64_t{0}; j < remaining; ++j, ++i) {
        _alternative._rest.push_back(_symmetries[i]);
    }
}

TCM_EXPORT SmallSpinBasis::~SmallSpinBasis() = default;

TCM_EXPORT auto SmallSpinBasis::full_info(uint64_t const x,
                                          unsigned*      symmetry_index) const
    -> std::tuple<StateT, std::complex<double>, double>
{
#if 0
    return ::TCM_NAMESPACE::full_info(_symmetries, x);
#else
    if (has_symmetries()) {
        auto r2 = representative(_alternative._chunks, _alternative._rest, x);
        // TCM_CHECK(std::get<0>(r1) == std::get<0>(r2), std::runtime_error, "");
        // TCM_CHECK(std::get<2>(r1) == std::get<1>(r2), std::runtime_error,
        //           fmt::format("{} != {}", std::get<2>(r1), std::get<1>(r2)));
        auto const eigenvalue =
            _symmetries[static_cast<uint64_t>(std::get<2>(r2))].eigenvalue();
        // TCM_CHECK(std::get<1>(r1) == eigenvalue, std::runtime_error, "");
        if (symmetry_index != nullptr) {
            *symmetry_index = static_cast<unsigned>(std::get<2>(r2));
        }
        return {std::get<0>(r2), eigenvalue, std::get<1>(r2)};
    }
    return {x, 1.0, 1.0};
#endif
}

TCM_EXPORT auto SmallSpinBasis::full_info(bits512 const& x,
                                          unsigned*      symmetry_index) const
    -> std::tuple<bits512, std::complex<double>, double>
{
    auto const [_r, eigenvalue, norm] = full_info(x.words[0], symmetry_index);
    bits512 representative;
    representative.words[0] = _r;
    std::fill(representative.words + 1, representative.words + 8, 0UL);
    return std::make_tuple(representative, eigenvalue, norm);
}

TCM_EXPORT auto SmallSpinBasis::index(uint64_t const x) const -> uint64_t
{
    TCM_CHECK(_cache != nullptr, std::runtime_error,
              "cache must be initialised before calling `index()`; use "
              "`build()` member function to initialise the cache.");
    return _cache->index(x, number_spins());
}

TCM_EXPORT auto SmallSpinBasis::is_real() const noexcept -> bool
{
    using std::begin, std::end;
    return std::all_of(begin(_symmetries), end(_symmetries), [](auto const& s) {
        return s.eigenvalue().imag() == 0;
    });
}

TCM_EXPORT auto SmallSpinBasis::build() -> void
{
    if (_cache == nullptr) {
        _cache = std::make_unique<detail::BasisCache>(
            gsl::span<v2::Symmetry<64> const>{_symmetries}, _number_spins,
            _hamming_weight);
    }
}

TCM_EXPORT auto SmallSpinBasis::number_states() const -> uint64_t
{
    TCM_CHECK(_cache != nullptr, std::runtime_error,
              "cache must be initialised before calling `number_states()`; "
              "use `build()` member function to initialise the cache.");
    return _cache->number_states();
}

TCM_EXPORT auto SmallSpinBasis::states() const -> gsl::span<StateT const>
{
    TCM_CHECK(_cache != nullptr, std::runtime_error,
              "cache must be initialised before calling `states()`; use "
              "`build()` member function to initialise the cache.");
    return _cache->states();
}

TCM_EXPORT auto SmallSpinBasis::_internal_state() const -> _PickleStateT
{
    auto representatives = [this]() {
        std::vector<uint64_t> out;
        if (_cache != nullptr) {
            out.reserve(_cache->number_states());
            auto const in = _cache->states();
            out.insert(std::end(out), std::begin(in), std::end(in));
        }
        return out;
    }();
    return std::make_tuple(number_spins(), hamming_weight(), _symmetries,
                           std::move(representatives));
}

TCM_EXPORT auto SmallSpinBasis::_from_internal_state(_PickleStateT const& state)
    -> std::shared_ptr<SmallSpinBasis>
{
    auto cache = std::make_unique<detail::BasisCache>(
        gsl::span{std::get<2>(state)}, std::get<0>(state), std::get<1>(state),
        std::get<3>(state));
    return std::make_shared<SmallSpinBasis>(
        std::get<2>(state), std::get<0>(state), std::get<1>(state),
        std::move(cache));
}
// }}}

// BigSpinBasis {{{
TCM_EXPORT BigSpinBasis::BigSpinBasis(std::vector<v2::Symmetry<512>> symmetries,
                                      unsigned                number_spins,
                                      std::optional<unsigned> hamming_weight)
    : BasisBase{number_spins, hamming_weight, !symmetries.empty()}
    , _symmetries{std::move(symmetries)}
{
    TCM_CHECK(_number_spins <= 512, std::invalid_argument,
              fmt::format("invalid number_spins: {}; systems larger than 512 "
                          "spins are not supported",
                          _number_spins));
}

TCM_EXPORT BigSpinBasis::~BigSpinBasis() = default;

TCM_EXPORT auto BigSpinBasis::full_info(bits512 const& x,
                                        unsigned*      symmetry_index) const
    -> std::tuple<bits512, std::complex<double>, double>
{
    return ::TCM_NAMESPACE::full_info(_symmetries, x);
}

TCM_EXPORT auto BigSpinBasis::is_real() const noexcept -> bool
{
    using std::begin, std::end;
    return std::all_of(begin(_symmetries), end(_symmetries), [](auto const& s) {
        return s.eigenvalue().imag() == 0;
    });
}
// }}}

TCM_NAMESPACE_END
