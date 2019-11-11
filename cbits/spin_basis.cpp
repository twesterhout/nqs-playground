#include "spin_basis.hpp"
#include <boost/pool/pool_alloc.hpp>

TCM_NAMESPACE_BEGIN

namespace detail {

auto global_executor() noexcept -> tf::Executor&
{
    static tf::Executor executor;
    return executor;
}

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

    GenerateStatesTask(GenerateStatesTask const&) = default;
    GenerateStatesTask(GenerateStatesTask&&)      = default;
    GenerateStatesTask& operator=(GenerateStatesTask const&) = default;
    GenerateStatesTask& operator=(GenerateStatesTask&&) = default;

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
auto split_into_tasks(Symmetry::UInt current, Symmetry::UInt const bound,
                      unsigned const number_chunks, Callback&& callback) -> void
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

auto generate_states_parallel(gsl::span<Symmetry const>     symmetries,
                              unsigned const                number_spins,
                              std::optional<unsigned> const hamming_weight)
    -> std::vector<Symmetry::UInt>
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

    auto r = std::move(states.front());
    states.pop_front();
    r.reserve(r.size()
              + std::accumulate(
                  std::begin(states), std::end(states), size_t{0},
                  [](auto acc, auto const& x) { return acc + x.size(); }));
    std::for_each(std::begin(states), std::end(states), [&r](auto& x) {
        r.insert(std::end(r), std::begin(x), std::end(x));
    });
    return r;
}

} // namespace detail

namespace v2 {

namespace {
/// Finds the largest index used in `_edges`.
///
/// \precondition Range must not be empty.
template <class Iter, class = std::enable_if_t<std::is_same<
                          typename std::iterator_traits<Iter>::value_type,
                          Heisenberg::edge_type>::value> /**/>
static auto find_max_index(Iter begin, Iter end) -> unsigned
{
    TCM_ASSERT(begin != end, "Range is empty");
    // This implementation is quite inefficient, but it's not on the hot
    // bath, so who cares ;)
    auto max_index = std::max(std::get<1>(*begin), std::get<2>(*begin));
    ++begin;
    for (; begin != end; ++begin) {
        max_index = std::max(
            max_index, std::max(std::get<1>(*begin), std::get<2>(*begin)));
    }
    return max_index;
}
} // namespace


Heisenberg::Heisenberg(spec_type edges, std::shared_ptr<SpinBasis const> basis)
    : _edges{std::move(edges)}
    , _basis{std::move(basis)}
    , _max_index{std::numeric_limits<unsigned>::max()}
    , _pool{sizeof(buffer_type::element_type) * _edges.size(), 32U}
{
    TCM_CHECK(_basis != nullptr, std::invalid_argument,
              fmt::format("basis should not be nullptr (None)"));
    for (auto const& edge : _edges) {
        auto const coupling = std::get<0>(edge);
        TCM_CHECK(
            std::isnormal(coupling.real()) && std::isfinite(coupling.imag()),
            std::invalid_argument,
            fmt::format("invalid coupling: {} + {}j; expected a normal (i.e. "
                        "neither zero, subnormal, infinite or NaN) float",
                        coupling.real(), coupling.imag()));
    }
    if (!_edges.empty()) {
        _max_index = find_max_index(std::begin(_edges), std::end(_edges));
    }
}

auto Heisenberg::get_buffer() const -> buffer_type
{
    return buffer_type{static_cast<buffer_type::pointer>(_pool.malloc()),
                       pool_deleter_type{&_pool}};
}

auto Heisenberg::operator()(SpinBasis::StateT const spin) const
    -> std::pair<buffer_type, size_t>
{
    auto buffer = get_buffer();
    auto size   = size_t{0};
    (*this)(spin, [&buffer, &size](auto const s, auto const c) {
        buffer[size++] = std::make_pair(s, c);
    });
    return std::make_pair(std::move(buffer), size);
}

} // namespace v2

TCM_NAMESPACE_END
