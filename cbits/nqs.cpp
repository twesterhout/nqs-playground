#include "nqs.hpp"

TCM_NAMESPACE_BEGIN

// [Errors] {{{
namespace detail {
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
    auto out = detail::make_f32_tensor(size());
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
{
    if (_hamiltonian == nullptr) {
        throw std::invalid_argument{
            "Polynomial(shared_ptr<Heisenberg>, vector<pair<complex_type, "
            "optional<real_type>>>): hamiltonian must not be nullptr"};
    }
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
    return *this;
}
// }}}

// [PolynomialState] {{{
template <class Factory>
auto parallel_for_lazy(int64_t begin, int64_t end, Factory factory,
                       int64_t cutoff) -> void
{
    static_assert(std::is_nothrow_copy_constructible<Factory>::value,
                  "Factory must be nothrow copy constructible to be usable "
                  "with OpenMP's firstprivate.");
    TCM_CHECK(begin <= end, std::invalid_argument,
              fmt::format("invalid range [{}, {})", begin, end));
    if (begin == end) { return; }
    if (end - begin <= cutoff) {
        auto f = factory();
        for (auto i = begin; i < end; ++i) {
            f(i);
        }
        return;
    }

    std::atomic_flag   err_flag{ATOMIC_FLAG_INIT};
    std::exception_ptr err_ptr{nullptr};
#pragma omp parallel default(none) firstprivate(begin, end, factory)           \
    shared(err_flag, err_ptr)
    {
        auto const num_threads  = omp_get_num_threads();
        auto const thread_id    = omp_get_thread_num();
        auto const size         = end - begin;
        auto const chunk_size   = size / num_threads;
        auto const rest         = size % num_threads;
        auto const thread_begin = begin + thread_id * chunk_size;
        if (thread_begin < end) {
            try {
                auto f = factory();
                for (auto i = thread_begin; i < thread_begin + chunk_size;
                     ++i) {
                    f(i);
                }
                if (thread_id < rest) {
                    f(begin + num_threads * chunk_size + thread_id);
                }
            }
            catch (...) {
                if (!err_flag.test_and_set()) {
                    err_ptr = std::current_exception();
                }
            }
        }
    }
    if (err_ptr != nullptr) { std::rethrow_exception(err_ptr); }
}

namespace detail {
struct _PolynomialState::Body {
  private:
    ForwardT const&             _forward;
    gsl::span<SpinVector const> _vectors;
    torch::Tensor               _coefficients;
    torch::Tensor               _buffer;
    size_t                      _batch_size;
    float&                      _sum;

  public:
    Body(ForwardT const& forward, Polynomial const& polynomial,
         size_t const batch_size, size_t const number_spins, float& dest)
        : _forward{forward}
        , _vectors{polynomial.vectors()}
        , _coefficients{polynomial.coefficients()}
        , _buffer{detail::make_f32_tensor(batch_size, number_spins)}
        , _batch_size{batch_size}
        , _sum{dest}
    {}

    Body(Body const&) = delete;
    Body(Body&&)      = default;
    Body& operator=(Body const&) = delete;
    Body& operator=(Body&&) = delete;

    auto operator()(int64_t const _i) -> void
    {
        TCM_ASSERT(_i >= 0, "Invalid index");
        auto const i    = static_cast<size_t>(_i);
        auto const size = _vectors.size();
        TCM_CHECK(i * _batch_size < size, std::logic_error,
                  fmt::format("invalid batch index {}; expected <{}", i,
                              size / _batch_size));
        if ((i + 1) * _batch_size <= size) {
            _sum += forward_propagate_batch(i);
        }
        else {
            _sum += forward_propagate_rest(i);
        }
    }

    TCM_NOINLINE auto forward_propagate_batch(size_t i) -> float;
    TCM_NOINLINE auto forward_propagate_rest(size_t i) -> float;
};

auto _PolynomialState::Body::forward_propagate_batch(size_t const i) -> float
{
    TCM_ASSERT((i + 1) * _batch_size <= _vectors.size(), "Index out of bounds");
    // Stores the `i`th batch of `_poly.vectors()` into `_buffer`.
    auto* data = _vectors.data() + i * _batch_size;
    detail::unpack_to_tensor(/*first=*/data, /*last=*/data + _batch_size,
                             /*destination=*/_buffer);
    // Forward propagates the batch through the network.
    auto output = _forward(_buffer).view({-1});
    // Extracts the `i`th batch of `_poly.coefficients()`.
    auto coefficients = _coefficients.slice(
        /*dim=*/0, /*start=*/static_cast<int64_t>(i * _batch_size),
        /*end=*/static_cast<int64_t>((i + 1) * _batch_size), /*step=*/1);
    // Computes the final result.
    // NOTE: This can't be optimised much because `_forward` is a user supplied
    // function which might return tensors of wrong shape.
    return torch::dot(std::move(output), std::move(coefficients)).item<float>();
}

auto _PolynomialState::Body::forward_propagate_rest(size_t const i) -> float
{
    auto const size = _vectors.size();
    auto const rest = size - i * _batch_size;
    TCM_ASSERT(rest < _batch_size, "Go use forward_propagate_batch instead");
    TCM_ASSERT(i * _batch_size + rest == size, "Precondition violated");
    // Stores part of batch which we're given into `_input`.
    auto* data = _vectors.data() + i * _batch_size;
    detail::unpack_to_tensor(
        /*first=*/data, /*last=*/data + rest,
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
    auto coefficients = _coefficients.slice(
        /*dim=*/0, /*start=*/static_cast<int64_t>(i * _batch_size),
        /*end=*/static_cast<int64_t>(i * _batch_size + rest), /*step=*/1);
    // Computes the final result.
    return torch::dot(std::move(output), std::move(coefficients)).item<float>();
}

auto _PolynomialState::time_poly() const -> std::pair<real_type, real_type>
{
    return {_poly_time.mean(), std::sqrt(_poly_time.variance())};
}

auto _PolynomialState::time_psi() const -> std::pair<real_type, real_type>
{
    return {_psi_time.mean(), std::sqrt(_psi_time.variance())};
}

template <size_t N>
inline auto zero_results(float (&xs)[N]) TCM_NOEXCEPT -> void
{
    constexpr auto vector_size = size_t{8};
    static_assert(N % vector_size == 0, "");
    TCM_ASSERT(reinterpret_cast<uintptr_t>(xs) % 32 == 0,
               "Input not aligned properly");
    for (auto i = size_t{0}; i < N / vector_size; ++i) {
        _mm256_stream_ps(xs + i * vector_size, _mm256_set1_ps(0.0f));
    }
}

template <size_t N>
inline auto sum_results(float (&xs)[N]) TCM_NOEXCEPT -> float
{
    constexpr auto vector_size = size_t{8};
    static_assert(N % vector_size == 0, "");
    TCM_ASSERT(reinterpret_cast<uintptr_t>(xs) % 32 == 0,
               "Input not aligned properly");
    auto sum = _mm256_set1_ps(0.0f);
    for (auto i = size_t{0}; i < N / vector_size; ++i) {
        sum = _mm256_add_ps(sum, _mm256_load_ps(xs + i * vector_size));
    }
    return detail::hadd(sum);
}

auto _PolynomialState::operator()(SpinVector const input,
                                  ForwardT const&  forward,
                                  std::string const& filename) -> float
{
    using MicroSecondsT =
        std::chrono::duration<real_type, std::chrono::microseconds::period>;
    // TODO(twesterhout): It'd be nice to time how much time this function
    // spends applying `_poly` to `input` and how much on forward propagation
    // through `_psi`.
    auto time_start = std::chrono::steady_clock::now();
    _poly(real_type{1}, input);
    auto time_interval =
        MicroSecondsT(std::chrono::steady_clock::now() - time_start);
    _poly_time(time_interval.count());

    std::vector<ForwardT> forward_functions;
    for (auto i = 0; i < 64; ++i) {
        forward_functions.emplace_back(detail::load_forward_fn(filename));
    }

    time_start                = std::chrono::steady_clock::now();
    auto const size           = _poly.size();
    auto const number_spins   = input.size();
    auto const number_batches = (size + batch_size() - 1) / batch_size();

    alignas(32) float results[64];
    zero_results(results);
    auto factory = [/*forward = std::cref(forward), &filename,*/ p = forward_functions.data(), poly = std::cref(_poly),
                    batch_size = batch_size(), number_spins = number_spins,
                    &results]() -> Body {
        auto const i = omp_get_thread_num();
        TCM_CHECK(i < 64, std::runtime_error,
                  fmt::format("too many OpenMP threads: {}; expected <=128.",
                              omp_get_num_threads()));
        return {p[i], poly, batch_size, number_spins, results[i]};
    };
    static_assert(std::is_nothrow_copy_constructible<decltype(factory)>::value,
                  "");
    parallel_for_lazy(0, static_cast<int64_t>(number_batches),
                      std::move(factory), 10);
    auto const sum = sum_results(results);

    time_interval =
        MicroSecondsT(std::chrono::steady_clock::now() - time_start);
    _psi_time(time_interval.count());
    return sum;
}
} // namespace detail

#if 0
auto PolynomialState::operator()(SpinVector const input) -> float
{
    using MicroSecondsT =
        std::chrono::duration<real_type, std::chrono::microseconds::period>;
    // TODO(twesterhout): Add a lookup in the table shared between multiple
    // parallel Markov chains.
    // Upon construction of the state, we don't know the number of spins in the
    // system. We extract it from the input.
    if (!_input.defined() || _input.size(1) != input.size()) {
        _input = detail::make_f32_tensor(batch_size(), input.size());
    }
    // TODO(twesterhout): It'd be nice to time how much time this function
    // spends applying `_poly` to `input` and how much on forward propagation
    // through `_psi`.
    auto time_start = std::chrono::steady_clock::now();
    _poly(real_type{1}, input);
    auto time_interval =
        MicroSecondsT(std::chrono::steady_clock::now() - time_start);
    _poly_time(time_interval.count());

    time_start                = std::chrono::steady_clock::now();
    auto const size           = _poly.size();
    auto const number_batches = size / batch_size();
    auto const rest           = size % batch_size();
    auto       sum            = 0.0f;
    for (auto i = size_t{0}; i < number_batches; ++i) {
        sum += forward_propagate_batch(i);
    }
    if (rest != 0) { sum += forward_propagate_rest(number_batches, rest); }
    time_interval =
        MicroSecondsT(std::chrono::steady_clock::now() - time_start);
    _psi_time(time_interval.count());
    return sum;
}

auto PolynomialState::forward_propagate_batch(size_t const n) -> float
{
    TCM_ASSERT(_input.defined(),
               "You should allocate the storage before calling this function");
    TCM_ASSERT((n + 1) * _batch_size <= _poly.size(), "Index out of bounds");
    // Stores the `n`th batch of `_poly.vectors()` into `_input`.
    auto* data = _poly.vectors().data() + n * batch_size();
    detail::unpack_to_tensor(/*first=*/data, /*last=*/data + batch_size(),
                             /*destination=*/_input);
    // Forward propagates the batch through our network `_psi`.
    auto output = _psi(_input).view({-1});
    // Extracts the `n`th batch of `_poly.coefficients()`.
    auto coefficients = _poly.coefficients().slice(
        /*dim=*/0, /*start=*/static_cast<int64_t>(n * batch_size()),
        /*end=*/static_cast<int64_t>((n + 1) * batch_size()), /*step=*/1);
    // Computes the final result.
    // NOTE: This can't be optimised much because `_psi` is a user supplied
    // function which might return tensors of wrong shape.
    return torch::dot(std::move(output), std::move(coefficients)).item<float>();
}

auto PolynomialState::forward_propagate_rest(size_t const n, size_t const rest)
    -> float
{
    TCM_ASSERT(_input.defined(),
               "You should allocate the storage before calling this function");
    TCM_ASSERT(rest < batch_size(), "Go use forward_propagate_batch instead");
    TCM_ASSERT(n * batch_size() + rest == _poly.size(),
               "Precondition violated");
    // Stores part of batch which we're given into `_input`.
    auto* data = _poly.vectors().data() + n * batch_size();
    detail::unpack_to_tensor(
        /*first=*/data, /*last=*/data + rest,
        /*destination=*/
        _input.slice(/*dim=*/0, /*start=*/0, /*end=*/static_cast<int64_t>(rest),
                     /*step=*/1));
    // Fills the remaining part of the batch with spin ups.
    _input.slice(/*dim=*/0, /*start=*/static_cast<int64_t>(rest),
                 /*end=*/static_cast<int64_t>(batch_size()),
                 /*step=*/1) = 1.0f;
    // Forward progates the batch through out network `_psi`. Only the first
    // `rest` components contain meaningful info.
    auto output = _psi(_input)
                      .slice(/*dim=*/0, /*start=*/0,
                             /*end=*/static_cast<int64_t>(rest), /*rest=*/1)
                      .view({-1});
    // Extracts part of the `n`th batch of `_poly.coefficients()`.
    auto coefficients = _poly.coefficients().slice(
        /*dim=*/0, /*start=*/static_cast<int64_t>(n * batch_size()),
        /*end=*/static_cast<int64_t>(n * batch_size() + rest), /*step=*/1);
    // Computes the final result.
    return torch::dot(std::move(output), std::move(coefficients)).item<float>();
}
#endif
// }}}

// [Random] {{{
namespace detail {
inline auto really_need_that_random_seed_now() -> uint64_t
{
    std::random_device                      random_device;
    std::uniform_int_distribution<uint64_t> dist;
    return dist(random_device);
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

auto random_spin(size_t const size, int const magnetisation,
                 RandomGenerator& generator) -> SpinVector
{
    return SpinVector::random(size, magnetisation, generator);
#if 0
    TCM_CHECK(size <= SpinVector::max_size(), std::invalid_argument,
              fmt::format("invalid size {}; expected <={}", size,
                          SpinVector::max_size()));
    TCM_CHECK(
        static_cast<size_t>(std::abs(magnetisation)) <= size,
        std::invalid_argument,
        fmt::format("magnetisation exceeds the number of spins: |{}| > {}",
                    magnetisation, size));
    TCM_CHECK((static_cast<int>(size) + magnetisation) % 2 == 0,
              std::runtime_error,
              fmt::format("{} spins cannot have a magnetisation of {}. `size + "
                          "magnetisation` must be even",
                          size, magnetisation));
    float      buffer[SpinVector::max_size()];
    auto const spin = gsl::span<float>{buffer, size};
    auto const number_ups =
        static_cast<size_t>((static_cast<int>(size) + magnetisation) / 2);
    auto const middle = std::begin(spin) + number_ups;
    std::fill(std::begin(spin), middle, 1.0f);
    std::fill(middle, std::end(spin), -1.0f);
    std::shuffle(std::begin(spin), std::end(spin), generator);
    auto compact_spin = SpinVector{spin};
    TCM_ASSERT(compact_spin.magnetisation() == magnetisation, "");
    return compact_spin;
#endif
}

// [ChainResult] {{{
auto ChainResult::extract_vectors() const -> torch::Tensor
{
    return detail::unpack_to_tensor(
        std::begin(_samples), std::end(_samples),
        [](auto const& x) -> SpinVector const& { return x.spin; });
}

auto ChainResult::extract_values() const -> torch::Tensor
{
    auto out      = detail::make_f32_tensor(_samples.size());
    auto accessor = out.accessor<float, 1>();
    for (auto i = size_t{0}; i < _samples.size(); ++i) {
        accessor[static_cast<int64_t>(i)] = _samples[i].value;
    }
    return out;
}

auto ChainResult::extract_count() const -> torch::Tensor
{
    auto out      = detail::make_i64_tensor(_samples.size());
    auto accessor = out.accessor<int64_t, 1>();
    for (auto i = size_t{0}; i < _samples.size(); ++i) {
        accessor[static_cast<int64_t>(i)] = _samples[i].count;
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
// }}}

TCM_NAMESPACE_END
