#include "nqs.hpp"

// #include <boost/stacktrace.hpp>

TCM_NAMESPACE_BEGIN

// [Errors] {{{
namespace detail {

TCM_NORETURN auto assert_fail(char const* expr, char const* file,
                              size_t const line, char const* function,
                              std::string const& msg) noexcept -> void
{
    std::fprintf(
        stderr,
        TCM_BUG_MESSAGE
        "\nAssertion failed at %s:%zu: %s: \"%s\" evaluated to false: %s\n",
        file, line, function, expr, msg.c_str());
    std::terminate();
}

auto make_what_message(char const* file, size_t const line,
                       char const* function, std::string const& description)
    -> std::string
{
    return fmt::format("{}:{}: {}: {}", file, line, function, description);
}

auto spin_configuration_to_string(gsl::span<float const> spin) -> std::string
{
    std::ostringstream msg;
    msg << '[';
    if (spin.size() > 0) {
        msg << spin[0];
        for (auto i = size_t{1}; i < spin.size(); ++i) {
            msg << ", " << spin[i];
        }
    }
    msg << ']';
    return msg.str();
}
} // namespace detail
// }}}

// [SpinVector] {{{
namespace detail {
namespace {
    template <
        int ExtraFlags,
        class = std::enable_if_t<ExtraFlags & pybind11::array::c_style
                                 || ExtraFlags & pybind11::array::f_style> /**/>
    auto copy_to_numpy_array(SpinVector const&                     spin,
                             pybind11::array_t<float, ExtraFlags>& out) -> void
    {
        TCM_CHECK_DIM(out.ndim(), 1);
        TCM_CHECK_SHAPE(out.shape(0), spin.size());

        auto const spin2float = [](Spin const s) noexcept->float
        {
            return s == Spin::up ? 1.0f : -1.0f;
        };

        auto access = out.template mutable_unchecked<1>();
        for (auto i = 0u; i < spin.size(); ++i) {
            access(i) = spin2float(spin[i]);
        }
    }
} // unnamed namespace
} // namespace detail

auto SpinVector::numpy() const
    -> pybind11::array_t<float, pybind11::array::c_style>
{
    pybind11::array_t<float, pybind11::array::c_style> out{size()};
    detail::copy_to_numpy_array(*this, out);
    return out;
}

auto SpinVector::tensor() const -> torch::Tensor
{
    auto out = detail::make_tensor<float>(size());
    copy_to({out.data<float>(), size()});
    return out;
}

SpinVector::SpinVector(gsl::span<float const> spin)
{
    check_range(spin);
    copy_from(spin);
    TCM_ASSERT(is_valid(), "Bug! Post-condition violated");
}

SpinVector::SpinVector(torch::Tensor const& spins)
{
    TCM_CHECK_DIM(spins.dim(), 1);
    TCM_CHECK(spins.is_contiguous(), std::invalid_argument,
              "input tensor must be contiguous");
    TCM_CHECK_TYPE(spins.scalar_type(), torch::kFloat32);
    auto buffer = gsl::span<float const>{spins.data<float>(),
                                         static_cast<size_t>(spins.size(0))};
    check_range(buffer);
    copy_from(buffer);
    TCM_ASSERT(is_valid(), "Bug! Post-condition violated");
}

SpinVector::SpinVector(pybind11::str str)
{
    // PyUnicode_Check macro uses old style casts
#if defined(TCM_GCC)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
    // Borrowed from pybind11/pytypes.h
    pybind11::object temp = str;
    if (PyUnicode_Check(str.ptr())) {
        temp = pybind11::reinterpret_steal<pybind11::object>(
            PyUnicode_AsUTF8String(str.ptr()));
        if (!temp)
            throw std::runtime_error{
                "Unable to extract string contents! (encoding issue)"};
    }
    char*             buffer;
    pybind11::ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(temp.ptr(), &buffer, &length)) {
        throw std::runtime_error{
            "Unable to extract string contents! (invalid type)"};
    }
    TCM_CHECK(length <= static_cast<pybind11::ssize_t>(max_size()),
              std::overflow_error,
              fmt::format("spin chain too long {}; expected <={}", length,
                          max_size()));

    _data.as_ints = _mm_set1_epi32(0);
    _data.size    = static_cast<std::uint16_t>(length);
    for (auto i = 0u; i < _data.size; ++i) {
        auto const s = buffer[i];
        TCM_CHECK(s == '0' || s == '1', std::domain_error,
                  fmt::format("invalid spin '{}'; expected '0' or '1'", s));
        unsafe_at(i) = (s == '1') ? Spin::up : Spin::down;
    }
#if defined(TCM_GCC)
#    pragma GCC diagnostic pop
#endif
    TCM_ASSERT(is_valid(), "Bug! Post-condition violated");
}

auto unpack_to_tensor(gsl::span<SpinVector const> src, torch::Tensor dst)
    -> void
{
    if (src.empty()) { return; }

    auto const size         = src.size();
    auto const number_spins = src[0].size();
    TCM_ASSERT(std::all_of(std::begin(src), std::end(src),
                           [number_spins](auto const& x) {
                               return x.size() == number_spins;
                           }),
               "Input range contains variable size spin chains");
    TCM_ASSERT(dst.dim() == 2, fmt::format("Invalid dimension {}", dst.dim()));
    TCM_ASSERT(size == static_cast<size_t>(dst.size(0)),
               fmt::format("Sizes don't match: size={}, dst.size(0)={}", size,
                           dst.size(0)));
    TCM_ASSERT(static_cast<int64_t>(number_spins) == dst.size(1),
               fmt::format("Sizes don't match: number_spins={}, dst.size(1)={}",
                           number_spins, dst.size(1)));
    TCM_ASSERT(dst.is_contiguous(), "Output tensor must be contiguous");

    auto const chunks_16     = number_spins / 16;
    auto const rest_16       = number_spins % 16;
    auto const rest_8        = number_spins % 8;
    auto const copy_cheating = [chunks = chunks_16 + (rest_16 != 0)](
                                   SpinVector const& spin, float* out) {
        for (auto i = 0u; i < chunks; ++i, out += 16) {
            detail::unpack(spin._data.spin[i], out);
        }
    };

    auto* data = dst.data<float>();
    for (auto i = size_t{0}; i < size - 1; ++i, data += number_spins) {
        copy_cheating(src[i], data);
    }
    src[size - 1].copy_to({data, number_spins});
}
// }}}

// [Heisenberg] {{{
Heisenberg::Heisenberg(std::vector<edge_type> edges, real_type const coupling)
    : _edges{std::move(edges)}
    , _coupling{coupling}
    , _max_index{std::numeric_limits<unsigned>::max()}
{
    TCM_CHECK(std::isnormal(coupling), std::invalid_argument,
              fmt::format("invalid coupling: {}; expected a normal (i.e. "
                          "neither zero, subnormal, infinite or NaN) float",
                          coupling));
    if (!_edges.empty()) {
        _max_index = find_max_index(std::begin(_edges), std::end(_edges));
    }
}
// }}}

// [Polynomial] {{{
Polynomial::Polynomial(std::shared_ptr<Heisenberg const> hamiltonian,
                       std::vector<Term>                 terms)
    : _current{}
    , _old{}
    , _hamiltonian{std::move(hamiltonian)}
    , _terms{std::move(terms)}
    , _basis{}
    , _coeffs{}
    , _lock{}
{
    TCM_CHECK(_hamiltonian != nullptr, std::invalid_argument,
              "hamiltonian must not be nullptr (or None)");
    auto const estimated_size =
        std::min(static_cast<size_t>(std::round(
                     std::pow(_hamiltonian->size() / 2, _terms.size()))),
                 size_t{4096});
    _old.reserve(estimated_size);
    _current.reserve(estimated_size);
}

auto Polynomial::operator()(complex_type const coeff, SpinVector const spin)
    -> Polynomial&
{
    TCM_CHECK(_lock.try_lock(), std::runtime_error,
              fmt::format("This function is not thread-safe, why are you "
                          "calling it from multiple threads?"));

    using std::swap;
    TCM_CHECK(std::isfinite(coeff.real()) && std::isfinite(coeff.imag()),
              std::runtime_error,
              fmt::format("invalid coefficient ({}, {}); expected a finite "
                          "(i.e. either normal, subnormal or zero)",
                          coeff.real(), coeff.imag()));
    TCM_CHECK(_hamiltonian->max_index() < spin.size(), std::out_of_range,
              fmt::format("spin configuration too short {}; expected >{}",
                          spin.size(), _hamiltonian->max_index()));
    if (_terms.empty()) {
        _old.clear();
        _old.emplace(spin, coeff);
        save_results(_old, torch::nullopt);
        return *this;
    }

    // The zeroth iteration: goal is to perform `_old := coeff * (H - root)|spin⟩`
    {
        // `|_old⟩ := - coeff * root|spin⟩`
        _old.clear();
        _old.emplace(spin, -_terms[0].root * coeff);
        // `|_old⟩ += coeff * H|spin⟩`
        (*_hamiltonian)(coeff, spin, _old);
    }
    // Other iterations
    TCM_ASSERT(_current.empty(), "Bug!");
    for (auto i = size_t{1}; i < _terms.size(); ++i) {
        auto const  root    = _terms[i].root;
        auto const& epsilon = _terms[i - 1].epsilon;
        if (epsilon.has_value()) {
            // Performs `|_current⟩ := (H - root)P[epsilon]|_old⟩` in two steps:
            // 1) `|_current⟩ := - root * P[epsilon]|_old⟩`
            for (auto const& item : _old) {
                if (std::abs(item.second.real()) >= *epsilon) {
                    _current.emplace(item.first, -root * item.second);
                }
            }
            // 2) `|_current⟩ += H P[epsilon]|_old⟩`
            for (auto const& item : _old) {
                if (std::abs(item.second.real()) >= *epsilon) {
                    (*_hamiltonian)(item.second, item.first, _current);
                }
            }
        }
        else {
            // Performs `|_current⟩ := (H - root)|_old⟩` in two steps:
            // 1) `|_current⟩ := - root|_old⟩`
            for (auto const& item : _old) {
                _current.emplace(item.first, -root * item.second);
            }
            // 2) `|_current⟩ += H |_old⟩`
            for (auto const& item : _old) {
                (*_hamiltonian)(item.second, item.first, _current);
            }
        }
        // |_old⟩ := |_current⟩, but to not waste allocated memory, we use
        // `swap + clear` instead.
        swap(_old, _current);
        _current.clear();
    }
    // Final filtering, i.e. `P[epsilon] |_old⟩`
    save_results(_old, _terms.back().epsilon);

    // TODO(twesterhout): This one is important!
    _lock.unlock();
    return *this;
}
// }}}

// [PolynomialState] {{{
namespace detail {
/// Sets all `xs` to `0`.
template <size_t N>
inline auto zero_results(float (&xs)[N]) TCM_NOEXCEPT -> void
{
    constexpr auto vector_size = size_t{8};
    static_assert(N % vector_size == 0, "");
    TCM_ASSERT(boost::alignment::is_aligned(32, xs),
               "Input not aligned properly");
    for (auto i = size_t{0}; i < N / vector_size; ++i) {
        _mm256_stream_ps(xs + i * vector_size, _mm256_set1_ps(0.0f));
    }
}

/// Returns the sum of all `xs`.
template <size_t N>
inline auto sum_results(float (&xs)[N]) TCM_NOEXCEPT -> float
{
    constexpr auto vector_size = size_t{8};
    static_assert(N % vector_size == 0, "");
    TCM_ASSERT(boost::alignment::is_aligned(32, xs),
               "Input not aligned properly");
    auto sum = _mm256_set1_ps(0.0f);
    for (auto i = size_t{0}; i < N / vector_size; ++i) {
        sum = _mm256_add_ps(sum, _mm256_load_ps(xs + i * vector_size));
    }
    return detail::hadd(sum);
}

auto PolynomialState::Worker::operator()(int64_t const batch_index) -> float
{
    TCM_ASSERT(batch_index >= 0, "Invalid index");
    auto const i    = static_cast<size_t>(batch_index);
    auto const size = _polynomial->vectors().size();
    TCM_ASSERT(i * _batch_size < size, "Index out of bounds");
    return ((i + 1) * _batch_size <= size) ? forward_propagate_batch(i)
                                           : forward_propagate_rest(i);
}

auto PolynomialState::Worker::forward_propagate_batch(size_t const i) -> float
{
    TCM_ASSERT((i + 1) * _batch_size <= _polynomial->vectors().size(),
               "Index out of bounds");
    // Stores the `i`th batch of `_poly.vectors()` into `_buffer`.
    auto const vectors = gsl::span<SpinVector const>{_polynomial->vectors()};
    unpack_to_tensor(vectors.subspan(i * _batch_size, _batch_size), _buffer);
    // Forward propagates the batch through the network.
    auto output = _forward(_buffer).view({-1});
    // Extracts the `i`th batch of `_poly.coefficients()`.
    auto coefficients = _polynomial->coefficients().slice(
        /*dim=*/0, /*start=*/static_cast<int64_t>(i * _batch_size),
        /*end=*/static_cast<int64_t>((i + 1) * _batch_size), /*step=*/1);
    // Computes the final result.
    return torch::dot(std::move(output), std::move(coefficients)).item<float>();
}

auto PolynomialState::Worker::forward_propagate_rest(size_t const i) -> float
{
    auto const vectors = gsl::span<SpinVector const>{_polynomial->vectors()};
    auto const size    = vectors.size();
    auto const rest    = size - i * _batch_size;
    TCM_ASSERT(rest < _batch_size, "Go use forward_propagate_batch instead");
    TCM_ASSERT(i * _batch_size + rest == size, "Precondition violated");
    TCM_ASSERT(_buffer.is_variable(), "");
    // Stores part of batch which we're given into `_input`.
    unpack_to_tensor(
        /*source=*/vectors.subspan(i * _batch_size, rest),
        /*destination=*/
        _buffer.slice(/*dim=*/0, /*start=*/0,
                      /*end=*/static_cast<int64_t>(rest),
                      /*step=*/1));
    // Fills the remaining part of the batch with spin ups.
    _buffer.slice(/*dim=*/0, /*start=*/static_cast<int64_t>(rest),
                  /*end=*/static_cast<int64_t>(_batch_size),
                  /*step=*/1) = 1.0f;
    // Forward progates the batch through out network `_psi`. Only the first
    // `rest` components contain meaningful info.
    auto output = _forward(_buffer)
                      .slice(/*dim=*/0, /*start=*/0,
                             /*end=*/static_cast<int64_t>(rest), /*rest=*/1)
                      .view({-1});
    // Extracts part of the `n`th batch of `_poly.coefficients()`.
    auto coefficients = _polynomial->coefficients().slice(
        /*dim=*/0, /*start=*/static_cast<int64_t>(i * _batch_size),
        /*end=*/static_cast<int64_t>(i * _batch_size + rest), /*step=*/1);
    // Computes the final result.
    return torch::dot(std::move(output), std::move(coefficients)).item<float>();
}

auto PolynomialState::operator()(SpinVector const input) -> float
{
    using MicroSecondsT =
        std::chrono::duration<real_type, std::chrono::microseconds::period>;
    TCM_ASSERT(!_workers.empty(), "There are no workers");
    auto const batch_size = _workers[0].batch_size();
    auto const num_spins  = _workers[0].number_spins();
    TCM_CHECK_SHAPE(input.size(), static_cast<int64_t>(num_spins));

    auto time_point = std::chrono::steady_clock::now();
    _poly(real_type{1}, input);
    _poly_time(
        MicroSecondsT(std::chrono::steady_clock::now() - time_point).count());

    time_point = std::chrono::steady_clock::now();
    alignas(32) float results[32];
    zero_results(results);
    auto factory = [this, &results](unsigned const i) {
        TCM_ASSERT(i < 32, "");
        TCM_ASSERT(i < _workers.size(), "");
        struct Body {
            Worker& worker;
            float&  result;

            Body(Body const&) = delete;
            constexpr Body(Body&&) noexcept = default;
            Body& operator=(Body const&) = delete;
            Body& operator=(Body&&) = delete;

            auto operator()(int64_t const n) -> void { result += worker(n); }
        };
        return Body{_workers[i], results[i]};
    };
    static_assert(std::is_nothrow_copy_constructible<decltype(factory)>::value,
                  TCM_BUG_MESSAGE);
    auto const number_batches = (_poly.size() + batch_size - 1) / batch_size;
    parallel_for_lazy(0, static_cast<int64_t>(number_batches),
                      std::move(factory), /*cutoff=*/1,
                      /*num_threads=*/_workers.size());
    auto const sum = sum_results(results);
    _psi_time(
        MicroSecondsT(std::chrono::steady_clock::now() - time_point).count());
    return sum;
}

auto PolynomialState::time_poly() const -> std::pair<real_type, real_type>
{
    return {_poly_time.mean(), std::sqrt(_poly_time.variance())};
}

auto PolynomialState::time_psi() const -> std::pair<real_type, real_type>
{
    return {_psi_time.mean(), std::sqrt(_psi_time.variance())};
}
} // namespace detail
// }}}

// [Random] {{{
namespace detail {
inline auto really_need_that_random_seed_now() -> uint64_t
{
    std::random_device                      random_device;
    std::uniform_int_distribution<uint64_t> dist;
    auto const seed = dist(random_device);
    return seed;
}
} // namespace detail

auto global_random_generator() -> RandomGenerator&
{
    static thread_local RandomGenerator generator{
        detail::really_need_that_random_seed_now()};
    return generator;
}
// }}}

// [RandomFlipper] {{{
constexpr RandomFlipper::index_type RandomFlipper::number_flips;

RandomFlipper::RandomFlipper(SpinVector const initial_spin,
                             RandomGenerator& generator)
    : _storage(initial_spin.size())
    , _generator{std::addressof(generator)}
    , _i{0}
{
    using std::begin;
    using std::end;
    TCM_CHECK(initial_spin.size() >= number_flips, std::invalid_argument,
              fmt::format("requested number of spin-flips exceeds the number "
                          "of spins in the system: {} > {}.",
                          number_flips, initial_spin.size()));
    std::iota(begin(_storage), end(_storage), 0);
    auto const number_ups =
        static_cast<size_t>(std::partition(begin(_storage), end(_storage),
                                           [spin = initial_spin](auto const i) {
                                               return spin[i] == Spin::up;
                                           })
                            - begin(_storage));
    _ups   = gsl::span<index_type>{_storage.data(), number_ups};
    _downs = gsl::span<index_type>{_storage.data() + number_ups,
                                   _storage.size() - number_ups};
    TCM_CHECK((_ups.size() >= number_flips / 2)
                  && (_downs.size() >= number_flips / 2),
              std::invalid_argument,
              fmt::format("initial spin is invalid. Given {} spins up and {} "
                          "spins down, it's impossible to perform {} "
                          "spin-flips and still preserve the magnetisation.",
                          _ups.size(), _downs.size(), number_flips));
    shuffle();
}

auto RandomFlipper::shuffle() -> void
{
    using std::begin;
    using std::end;
    std::shuffle(begin(_ups), end(_ups), *_generator);
    std::shuffle(begin(_downs), end(_downs), *_generator);
}
// }}}

// [ChainResult] {{{
auto ChainResult::extract_vectors() const -> torch::Tensor
{
    return detail::unpack_to_tensor(
        std::begin(_samples), std::end(_samples),
        [](auto const& x) -> SpinVector const& { return x.spin; });
}

auto ChainResult::extract_values() const -> torch::Tensor
{
    auto out      = detail::make_tensor<float>(_samples.size());
    auto accessor = out.accessor<float, 1>();
    for (auto i = size_t{0}; i < _samples.size(); ++i) {
        accessor[static_cast<int64_t>(i)] = _samples[i].value;
    }
    return out;
}

auto ChainResult::extract_count() const -> torch::Tensor
{
    auto out      = detail::make_tensor<int64_t>(_samples.size());
    auto accessor = out.accessor<int64_t, 1>();
    for (auto i = size_t{0}; i < _samples.size(); ++i) {
        accessor[static_cast<int64_t>(i)] =
            static_cast<int64_t>(_samples[i].count);
    }
    return out;
}

auto merge(ChainResult const& _x, ChainResult const& _y) -> ChainResult
{
    auto const&           x = _x.samples();
    auto const&           y = _y.samples();
    ChainResult::SamplesT buffer;
    buffer.reserve(x.size() + y.size());

    std::merge(std::begin(x), std::end(x), std::begin(y), std::end(y),
               std::back_inserter(buffer));
    buffer.erase(
        compress(std::begin(buffer), std::end(buffer), std::equal_to<void>{},
                 [](auto& acc, auto const& value) { return acc.merge(value); }),
        std::end(buffer));
    return ChainResult{std::move(buffer)};
}

auto merge(std::vector<ChainResult>&& results) -> ChainResult
{
    if (results.empty()) { return {}; }
    for (auto i = size_t{1}; i < results.size(); ++i) {
        results[0] = merge(results[0], results[i]);
    }
    return std::move(results[0]);
}
// }}}

auto parallel_sample_some(std::string const& filename,
                          Polynomial const& polynomial, Options const& options,
                          std::tuple<unsigned, unsigned> num_threads)
    -> ChainResult
{
    std::vector<ChainResult> results(options.steps[0]);

    auto func = [&filename, &polynomial, &options,
                 num_threads = static_cast<int>(std::get<1>(num_threads)),
                 results     = results.data()](int64_t const i) {
#pragma omp critical
        std::cout << fmt::format("Thread {} calculating {}...\n", omp_get_thread_num(), i);

        results[i] = sample_some(filename, Polynomial{polynomial, SplitTag{}},
                                 options, num_threads);
    };
    static_assert(std::is_nothrow_copy_constructible<decltype(func)>::value,
                  TCM_BUG_MESSAGE);
    detail::parallel_for(
        0, options.steps[0], std::move(func), /*cutoff=*/1,
        /*num_threads=*/static_cast<int>(std::get<0>(num_threads)));
    return merge(std::move(results));
}

TCM_NAMESPACE_END
