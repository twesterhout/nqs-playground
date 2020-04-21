#include "operator.hpp"
#include "parallel.hpp"

TCM_NAMESPACE_BEGIN

namespace {
/// Finds the largest index used in `_edges`.
///
/// \precondition Range must not be empty.
template <class Iter> auto find_max_index(Iter begin, Iter end)
{
    TCM_ASSERT(begin != end, "Range is empty");
    // This implementation is quite inefficient, but it's not on the hot
    // bath, so who cares ;)
    auto max_index = std::max(std::get<0>(*begin), std::get<1>(*begin));
    ++begin;
    for (; begin != end; ++begin) {
        max_index = std::max(
            max_index, std::max(std::get<0>(*begin), std::get<1>(*begin)));
    }
    return max_index;
}
} // namespace

TCM_EXPORT
Interaction::Interaction(std::array<std::array<complex_type, 4>, 4> matrix,
                         std::vector<edge_type>                     edges)
    : _edges{std::move(edges)}
{
    TCM_CHECK(!_edges.empty(), std::invalid_argument,
              fmt::format("'edges' should be non-empty. It makes little sense "
                          "to have 'Interaction' with no edges since it will "
                          "do nothing but waste cpu cycles."));
    auto const check = [](auto const& value, auto const _i, auto const _j) {
        auto const okay = [](auto const x) {
            switch (std::fpclassify(x)) {
            case FP_NORMAL:
            case FP_ZERO: return true;
            default: return false;
            }
        };
        TCM_CHECK(okay(value.real()) && okay(value.imag()),
                  std::invalid_argument,
                  fmt::format("matrix[{}, {}] is invalid: {} + {}j; expected "
                              "either a normal or zero value",
                              _i, _j, value.real(), value.imag()));
    };
    for (auto j = 0U; j < 4U; ++j) {
        for (auto i = 0U; i < 4U; ++i) {
            check(matrix[i][j], i, j);
            _matrix[i + j * 4U] = matrix[i][j];
        }
    }
}

TCM_EXPORT auto Interaction::is_real() const noexcept -> bool
{
    using std::begin, std::end;
    return std::all_of(begin(_matrix), end(_matrix),
                       [](auto const& x) { return x.imag() == 0.0; });
}

TCM_EXPORT auto Interaction::max_index() const noexcept -> uint16_t
{
    using std::begin, std::end;
    return find_max_index(begin(_edges), end(_edges));
}

TCM_EXPORT Operator::Operator(std::vector<Interaction>         interactions,
                              std::shared_ptr<BasisBase const> basis)
    : _interactions{std::move(interactions)}
    , _basis{std::move(basis)}
    , _is_real{true}
{
    TCM_CHECK(_basis != nullptr, std::invalid_argument,
              fmt::format("basis should not be nullptr (None)"));
    TCM_CHECK(!_interactions.empty(), std::invalid_argument,
              fmt::format("interactions should be non-empty"));
    using std::begin, std::end;
    auto max_index =
        std::accumulate(begin(_interactions), end(_interactions), uint16_t{0},
                        [](auto const max, auto const& x) {
                            return std::max(max, x.max_index());
                        });
    TCM_CHECK(max_index < _basis->number_spins(), std::invalid_argument,
              fmt::format("system is too small: 'basis' specifies {} spins, "
                          "but biggest index in 'interactions' is {}",
                          _basis->number_spins(), max_index));

    _is_real = std::accumulate(
        begin(_interactions), end(_interactions), _basis->is_real(),
        [](auto const acc, auto const& x) { return acc && x.is_real(); });
}

template <class T, class>
auto Operator::operator()(gsl::span<T const> x, gsl::span<T> y) const -> void
{
    auto const* basis = dynamic_cast<SmallSpinBasis const*>(_basis.get());
    TCM_CHECK(
        basis != nullptr, std::runtime_error,
        fmt::format(
            "Function call operator is not supported for Hamiltonians "
            "constructed with BigSpinBasis. If your system contains no more "
            "than 64 spins, reconstruct the Hamiltonian with SmallSpinBasis."));
    TCM_CHECK(
        x.size() == y.size() && y.size() == basis->number_states(),
        std::invalid_argument,
        fmt::format(
            "vectors have invalid sizes: [{0}], [{1}]; expected [{2}], [{2}]",
            x.size(), y.size(), basis->number_states()));
    if constexpr (!is_complex<T>::value) {
        TCM_CHECK(
            _is_real, std::runtime_error,
            "cannot apply a complex-valued Operator to a real-valued vector");
    }
    auto const states = basis->states();

    struct alignas(64) Task {
        Operator const& self;
        T const*        x_p;
        T*              y_p;
        uint64_t const* states_p;

        auto operator()(uint64_t const j) const -> void
        {
            auto const* basis =
                static_cast<SmallSpinBasis const*>(self._basis.get());
            auto acc = T{0};
            self.call_impl<SmallSpinBasis>(
                states_p[j],
                [&acc, basis, this](auto const spin, auto const coeff) {
                    if constexpr (is_complex<T>::value) {
                        acc += static_cast<T>(std::conj(coeff))
                               * x_p[basis->index(spin)];
                    }
                    else {
                        acc += static_cast<T>(coeff.real())
                               * x_p[basis->index(spin)];
                    }
                });
            y_p[j] = acc;
        }
    } task{*this, x.data(), y.data(), states.data()};

    auto const chunk_size = std::max(
        500UL,
        states.size() / (20UL * static_cast<unsigned>(omp_get_max_threads())));
    omp_parallel_for(std::cref(task), 0UL, y.size(), chunk_size);
}

#define DEFINE_OPERATOR_CALL(T)                                                \
    template TCM_EXPORT auto Operator::operator()(gsl::span<T const> x,        \
                                                  gsl::span<T> y) const->void

DEFINE_OPERATOR_CALL(float);
DEFINE_OPERATOR_CALL(double);
DEFINE_OPERATOR_CALL(std::complex<float>);
DEFINE_OPERATOR_CALL(std::complex<double>);

#undef DEFINE_OPERATOR_CALL

template <class T> constexpr auto to_torch_dtype() noexcept;

template <> constexpr auto to_torch_dtype<float>() noexcept
{
    return torch::kFloat32;
}
template <> constexpr auto to_torch_dtype<double>() noexcept
{
    return torch::kFloat64;
}

template <class T>
auto Operator::_to_sparse() const -> std::tuple<torch::Tensor, torch::Tensor>
{
    auto const* basis = dynamic_cast<SmallSpinBasis const*>(_basis.get());
    TCM_CHECK(basis != nullptr, std::runtime_error,
              fmt::format(
                  "_to_sparse is not supported for Operators constructed "
                  "with BigSpinBasis. If your system contains no more than 64 "
                  "spins, reconstruct the Hamiltonian with SmallSpinBasis."));
    if constexpr (!is_complex<T>::value) {
        TCM_CHECK(
            _is_real, std::runtime_error,
            "cannot convert a complex-valued Operator to a real-valued matrix");
    }
    auto const states = basis->states();

    std::vector<std::tuple<T, uint64_t, uint64_t>> triplets;
    for (auto i = uint64_t{0}; i < states.size(); ++i) {
        call_impl<SmallSpinBasis>(
            states[i],
            [&triplets, &basis, i](auto const spin, auto const coeff) {
                auto const j = basis->index(spin);
                if constexpr (is_complex<T>::value) {
                    triplets.push_back({static_cast<T>(coeff), j, i});
                }
                else {
                    triplets.push_back({static_cast<T>(coeff.real()), j, i});
                }
            });
    }
    std::sort(std::begin(triplets), std::end(triplets),
              [](auto const& a, auto const& b) {
                  if (std::get<1>(a) < std::get<1>(b)) { return true; }
                  else if (std::get<1>(a) == std::get<1>(b)) {
                      return std::get<2>(a) < std::get<2>(b);
                  }
                  return false;
              });

    auto const n       = static_cast<int64_t>(triplets.size());
    auto       indices = torch::empty(std::initializer_list<int64_t>{n, 2},
                                torch::TensorOptions{}.dtype(torch::kInt64));
    std::transform(std::begin(triplets), std::end(triplets),
                   static_cast<std::array<int64_t, 2>*>(indices.data_ptr()),
                   [](auto const& t) -> std::array<int64_t, 2> {
                       return {static_cast<int64_t>(std::get<1>(t)),
                               static_cast<int64_t>(std::get<2>(t))};
                   });
    torch::Tensor values;
    if constexpr (is_complex<T>::value) {
        values = torch::empty(std::initializer_list<int64_t>{n, 2},
                              torch::TensorOptions{}.dtype(
                                  to_torch_dtype<typename T::value_type>()));
    }
    else {
        values =
            torch::empty(std::initializer_list<int64_t>{n},
                         torch::TensorOptions{}.dtype(to_torch_dtype<T>()));
    }
    std::transform(std::begin(triplets), std::end(triplets),
                   static_cast<T*>(values.data_ptr()),
                   [](auto const& t) { return std::get<0>(t); });
    return std::make_tuple(std::move(values), std::move(indices));
}

template TCM_EXPORT auto Operator::_to_sparse<double>() const
    -> std::tuple<torch::Tensor, torch::Tensor>;

TCM_NAMESPACE_END
