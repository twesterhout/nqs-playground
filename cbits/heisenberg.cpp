#include "heisenberg.hpp"
#include "parallel.hpp"

TCM_NAMESPACE_BEGIN

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

TCM_EXPORT Heisenberg::Heisenberg(spec_type                        edges,
                                  std::shared_ptr<SpinBasis const> basis)
    : _edges{std::move(edges)}
    , _basis{std::move(basis)}
    , _max_index{std::numeric_limits<unsigned>::max()}
    , _is_real{true}
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
        if (coupling.imag() != 0) { _is_real = false; }
    }
    if (!_edges.empty()) {
        _max_index = find_max_index(std::begin(_edges), std::end(_edges));
    }
}

template <class T, class>
auto Heisenberg::operator()(gsl::span<T const> x, gsl::span<T> y) const -> void
{
    TCM_CHECK(
        x.size() == y.size() && y.size() == _basis->number_states(),
        std::invalid_argument,
        fmt::format("vectors have invalid sizes: {0}, {1}; expected {2}, {2}",
                    x.size(), y.size(), _basis->number_spins()));
    if constexpr (!is_complex<T>::value) {
        TCM_CHECK(_is_real, std::runtime_error,
                  "cannot apply a complex-valued Hamiltonian to a "
                  "real-valued vector");
    }
    auto&      executor = global_executor();
    auto const states   = _basis->states();
    auto const chunk_size =
        std::max(500UL, states.size() / (20UL * executor.num_workers()));

    struct __attribute__((visibility("hidden"), aligned(64))) Task {
        Heisenberg const&     self;
        T const*              x_p;
        T*                    y_p;
        Symmetry::UInt const* states_p;

        auto operator()(uint64_t const j) const -> void
        {
            auto acc = T{0};
            self(states_p[j], [&acc, this](auto const spin, auto const coeff) {
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
    taskflow.parallel_for(uint64_t{0}, y.size(), uint64_t{1}, std::cref(task),
                          chunk_size);
    executor.run(taskflow).wait();
}

#define DEFINE_OPERATOR_CALL(T)                                                \
    template TCM_EXPORT auto Heisenberg::operator()(                           \
        gsl::span<T const> x, gsl::span<T> y) const->void

DEFINE_OPERATOR_CALL(float);
DEFINE_OPERATOR_CALL(double);
DEFINE_OPERATOR_CALL(std::complex<float>);
DEFINE_OPERATOR_CALL(std::complex<double>);

#undef DEFINE_OPERATOR_CALL

TCM_NAMESPACE_END
