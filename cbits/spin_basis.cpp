#include "spin_basis.hpp"
#include "common.hpp"
#include "parallel.hpp"
// #include <boost/pool/pool_alloc.hpp>

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

    auto _state_as_tuple() const
        -> std::tuple<std::vector<Symmetry::UInt>,
                      std::vector<std::pair<uint64_t, uint64_t>>>;
};
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
    using UInt = Symmetry::UInt;

    UInt                      current;
    UInt                      upper_bound;
    gsl::span<Symmetry const> symmetries;
    std::vector<UInt>*        states;

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
        static_assert(std::is_same<Symmetry::UInt, uint64_t>::value,
                      TCM_STATIC_ASSERT_BUG_MESSAGE);
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
    auto handle(Symmetry::UInt const x) -> void
    {
        using std::begin, std::end;
        auto const [repr, _, norm] =
            get_info(begin(symmetries), end(symmetries), x);
        if (repr == x && norm > 0.0) { states->push_back(x); }
    }
};

template <bool FixedHammingWeight, class Callback>
auto split_into_tasks(
    Symmetry::UInt current, Symmetry::UInt const bound,
    unsigned const number_chunks,
    Callback&&
        callback) noexcept(noexcept(std::
                                        declval<Callback&>()(
                                            std::declval<Symmetry::UInt>(),
                                            std::declval<Symmetry::UInt>())))
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

#if 0
inline auto generate_states_parallel(
    gsl::span<Symmetry const> symmetries, unsigned const number_spins,
    std::optional<unsigned> const hamming_weight) -> BasisCache::StatesT
{
    TCM_CHECK(0 < number_spins && number_spins <= 64, std::invalid_argument,
              fmt::format("invalid number of spins: {}; expected a "
                          "positive integer not greater than 64.",
                          number_spins));

    auto&        executor = global_executor();
    tf::Taskflow taskflow;
    auto const   number_chunks = 20U * executor.num_workers();

    // We want to make sure that references to states remain valid, so we can't
    // use vector here.
    std::forward_list<std::vector<Symmetry::UInt>,
                      boost::fast_pool_allocator<std::vector<Symmetry::UInt>>>
        states;
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
        split_into_tasks<true>(
            current, bound, number_chunks,
            [&symmetries, &states, &taskflow](auto const first,
                                              auto const last) {
                auto& chunk = states.emplace_front();
                taskflow.emplace(GenerateStatesTask<true>{
                    first, last, symmetries, std::addressof(chunk)});
            });
    }
    else {
        auto current = uint64_t{0};
        auto bound   = number_spins == 64U
                         ? (~uint64_t{0})
                         : ((~uint64_t{0}) >> (64U - number_spins));
        split_into_tasks<false>(
            current, bound, number_chunks,
            [&symmetries, &states, &taskflow](auto const first,
                                              auto const last) {
                auto& chunk = states.emplace_front();
                taskflow.emplace(GenerateStatesTask<false>{
                    first, last, symmetries, std::addressof(chunk)});
            });
    }

    states.reverse();
    executor.run(taskflow).wait();

    auto r = BasisCache::StatesT{};
    r.reserve(std::accumulate(
        std::begin(states), std::end(states), size_t{0},
        [](auto acc, auto const& x) { return acc + x.size(); }));
    std::for_each(std::begin(states), std::end(states), [&r](auto& x) {
        r.insert(std::end(r), std::begin(x), std::end(x));
    });
    return r;
}
#endif

inline auto generate_states(gsl::span<Symmetry const>     symmetries,
                            unsigned const                number_spins,
                            std::optional<unsigned> const hamming_weight)
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
    std::forward_list<std::vector<Symmetry::UInt>> states;
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
                        chunk.reserve(1048576UL / sizeof(Symmetry::UInt));
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
                        chunk.reserve(1048576UL / sizeof(Symmetry::UInt));
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
    : _states{generate_states(symmetries, number_spins,
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

auto BasisCache::_state_as_tuple() const
    -> std::tuple<std::vector<Symmetry::UInt>,
                  std::vector<std::pair<uint64_t, uint64_t>>>
{
    return {{_states.begin(), _states.end()}, {_ranges.begin(), _ranges.end()}};
}
// }}}

} // namespace detail

TCM_EXPORT SpinBasis::SpinBasis(std::vector<Symmetry>   symmetries,
                                unsigned                number_spins,
                                std::optional<unsigned> hamming_weight)
    : _symmetries{std::move(symmetries)}
    , _number_spins{number_spins}
    , _hamming_weight{std::move(hamming_weight)}
    , _cache{nullptr}
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

TCM_EXPORT SpinBasis::~SpinBasis() = default;

TCM_EXPORT auto SpinBasis::full_info(StateT const x) const
    -> std::tuple<StateT, std::complex<double>, double>
{
    using std::begin, std::end;
    return detail::get_info(begin(_symmetries), end(_symmetries), x);
}

TCM_EXPORT auto SpinBasis::index(StateT const x) const -> uint64_t
{
    TCM_CHECK(_cache != nullptr, std::runtime_error,
              "cache must be initialised before calling `index()`; use "
              "`build()` member function to initialise the cache.");
    return _cache->index(x, number_spins());
}

TCM_EXPORT auto SpinBasis::is_real() const noexcept -> bool
{
    using std::begin, std::end;
    return std::all_of(begin(_symmetries), end(_symmetries), [](auto const& s) {
        return s.sector() == 0 || s.periodicity() == 2 * s.sector();
    });
}

TCM_EXPORT auto SpinBasis::build() -> void
{
    if (_cache == nullptr) {
        _cache = std::make_unique<detail::BasisCache>(
            gsl::span<Symmetry const>{_symmetries}, _number_spins,
            _hamming_weight);
    }
}

TCM_EXPORT auto SpinBasis::number_states() const -> uint64_t
{
    TCM_CHECK(_cache != nullptr, std::runtime_error,
              "cache must be initialised before calling `number_states()`; "
              "use `build()` member function to initialise the cache.");
    return _cache->number_states();
}

TCM_EXPORT auto SpinBasis::states() const -> gsl::span<StateT const>
{
    TCM_CHECK(_cache != nullptr, std::runtime_error,
              "cache must be initialised before calling `states()`; use "
              "`build()` member function to initialise the cache.");
    return _cache->states();
}

TCM_EXPORT auto SpinBasis::_state_as_tuple() const -> PickleStateT
{
    return {_symmetries, _number_spins, _hamming_weight,
            _cache == nullptr ? std::nullopt
                              : std::optional{_cache->_state_as_tuple()}};
}

TCM_EXPORT auto SpinBasis::_from_tuple_state(PickleStateT const& state)
    -> std::shared_ptr<SpinBasis>
{
    auto const& [symmetries, number_spins, hamming_weight, cache] = state;
    auto basis =
        std::make_shared<SpinBasis>(symmetries, number_spins, hamming_weight);
    if (cache.has_value()) {
        auto const& [states, ranges] = *cache;
        basis->_cache                = std::make_unique<detail::BasisCache>(
            detail::BasisCache::StatesT{states.begin(), states.end()},
            detail::BasisCache::RangesT{ranges.begin(), ranges.end()});
    }
    return basis;
}

// expand_basis {{{
template <class T>
auto expand_states(SpinBasis const& basis, gsl::span<T const> src,
                   gsl::span<SpinBasis::StateT const> states, gsl::span<T> dst)
    -> void
{
    TCM_CHECK(
        basis.number_states() == src.size(), std::invalid_argument,
        fmt::format("src has wrong size: {}; expected a 1D array of length {}",
                    src.size(), basis.number_states()));
    TCM_CHECK(states.size() == dst.size(), std::invalid_argument,
              fmt::format("states and dst have different sizes: {} != {}",
                          states.size(), dst.size()));
    TCM_CHECK(basis.is_real(), std::runtime_error,
              fmt::format("cannot expand the state into a real vector, because "
                          "some symmetries have complex eigenvalues"));

    auto&      executor = global_executor();
    auto const chunk_size =
        std::max(2000UL, states.size() / (20UL * executor.num_workers()));
    tf::Taskflow taskflow;
    taskflow.parallel_for(
        size_t{0}, dst.size(), size_t{1},
        [&basis, src_p = src.data(), states_p = states.data(),
         dst_p = dst.data()](auto const i) {
            auto const [spin, eigenvalue, norm] = basis.full_info(states_p[i]);
            if constexpr (is_complex_v<T>) {
                dst_p[i] = static_cast<T>(eigenvalue / norm)
                           * src_p[basis.index(spin)];
            }
            else {
                dst_p[i] = static_cast<T>(norm / eigenvalue.real())
                           * src_p[basis.index(spin)];
            }
        },
        chunk_size);
    executor.run(taskflow).wait();
}

TCM_EXPORT auto expand_states(SpinBasis const& basis, torch::Tensor src,
                              torch::Tensor states) -> torch::Tensor
{
    TCM_CHECK(basis.is_real(), std::runtime_error,
              fmt::format("cannot expand the state into a real vector, because "
                          "some symmetries have complex eigenvalues"));
    TCM_CHECK_TYPE("states", states, torch::kInt64);
    TCM_CHECK_SHAPE("states", states, {-1});
    TCM_CHECK_CONTIGUOUS("states", states);
    TCM_CHECK(states.device().type() == c10::DeviceType::CPU, std::domain_error,
              fmt::format("states must reside on the CPU"));

    TCM_CHECK_SHAPE("src", src, {static_cast<int64_t>(basis.number_states())});
    TCM_CHECK_CONTIGUOUS("src", src);
    TCM_CHECK(src.device().type() == c10::DeviceType::CPU, std::domain_error,
              fmt::format("src must reside on the CPU"));
    auto const dtype = src.scalar_type();
    TCM_CHECK(
        dtype == torch::kFloat32 || dtype == torch::kFloat64, std::domain_error,
        fmt::format("src has wrong data type: {}; expected either {} or {}",
                    src.scalar_type(), torch::kFloat32, torch::kFloat64));

    auto out = torch::empty(std::initializer_list<int64_t>{states.numel()},
                            torch::TensorOptions{}.dtype(dtype));
    if (dtype == torch::kFloat32) {
        expand_states<float>(
            basis,
            gsl::span<float const>{static_cast<float const*>(src.data_ptr()),
                                   basis.number_states()},
            gsl::span<uint64_t const>{
                static_cast<uint64_t const*>(states.data_ptr()),
                static_cast<size_t>(states.numel())},
            gsl::span<float>{static_cast<float*>(out.data_ptr()),
                             static_cast<size_t>(out.numel())});
    }
    else {
        TCM_ASSERT(dtype == torch::kFloat64, "");
        expand_states<double>(
            basis,
            gsl::span<double const>{static_cast<double const*>(src.data_ptr()),
                                    basis.number_states()},
            gsl::span<uint64_t const>{
                static_cast<uint64_t const*>(states.data_ptr()),
                static_cast<size_t>(states.numel())},
            gsl::span<double>{static_cast<double*>(out.data_ptr()),
                              static_cast<size_t>(out.numel())});
    }
    return out;
}
// }}}

TCM_NAMESPACE_END
