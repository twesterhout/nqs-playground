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

#include "polynomial.hpp"
#include "parallel.hpp"
#include <boost/align/is_aligned.hpp>
#include <torch/extension.h>

#if defined(TCM_GCC)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wundef"
#    pragma GCC diagnostic ignored "-Wold-style-cast"
#elif defined(TCM_CLANG)
#    pragma clang diagnostic push
#endif
#include <pybind11/stl.h>
#if defined(TCM_GCC)
#    pragma GCC diagnostic pop
#elif defined(TCM_CLANG)
#    pragma clang diagnostic pop
#endif

TCM_NAMESPACE_BEGIN

auto keys(QuantumState const& psi) -> aligned_vector<SpinVector>
{
    using std::begin, std::end;
    aligned_vector<SpinVector> spins;
    spins.reserve(psi.size());
    std::transform(begin(psi), end(psi), std::back_inserter(spins),
                   [](auto const& item) { return item.first; });
    return spins;
}

auto values(QuantumState const& psi, bool only_real) -> torch::Tensor
{
    using std::begin, std::end;
    if (only_real) {
        auto  coeffs = detail::make_tensor<float>(psi.size());
        auto* data   = reinterpret_cast<float*>(coeffs.data_ptr());
        std::transform(begin(psi), end(psi), data, [](auto const& item) {
            return static_cast<float>(item.second.real());
        });
        return coeffs;
    }
    else {
        auto  coeffs = detail::make_tensor<float>(psi.size(), 2);
        auto* data = reinterpret_cast<std::complex<float>*>(coeffs.data_ptr());
        std::transform(begin(psi), end(psi), data,
                       [](auto const& item) { return item.second; });
        return coeffs;
    }
}

auto items(QuantumState const& psi, bool only_real)
    -> std::pair<aligned_vector<SpinVector>, torch::Tensor>
{
    return std::make_pair(keys(psi), values(psi, only_real));
}

Heisenberg::Heisenberg(spec_type edges)
    : _edges{std::move(edges)}
    , _max_index{std::numeric_limits<unsigned>::max()}
{
    for (auto const& edge : _edges) {
        auto const coupling = std::get<0>(edge);
        TCM_CHECK(std::isnormal(coupling), std::invalid_argument,
                  fmt::format("invalid coupling: {}; expected a normal (i.e. "
                              "neither zero, subnormal, infinite or NaN) float",
                              coupling));
    }
    if (!_edges.empty()) {
        _max_index = find_max_index(std::begin(_edges), std::end(_edges));
    }
}

Polynomial::Polynomial(std::shared_ptr<Heisenberg const> hamiltonian,
                       std::vector<complex_type> roots, bool const normalising)
    : _current{}
    , _old{}
    , _hamiltonian{std::move(hamiltonian)}
    , _roots{std::move(roots)}
    , _normalising{normalising}
{
    TCM_CHECK(_hamiltonian != nullptr, std::invalid_argument,
              "hamiltonian must not be nullptr (or None)");
    TCM_CHECK(!_roots.empty(), std::invalid_argument,
              "zero-degree polynomials are not supported");
    auto const estimated_size =
        std::min(static_cast<size_t>(std::round(
                     std::pow(_hamiltonian->size() / 2, _roots.size()))),
                 size_t{16384});
    _old.reserve(estimated_size);
    _current.reserve(estimated_size);
}

auto Polynomial::operator()(complex_type coeff, SpinVector const spin)
    -> QuantumState const&
{
    TCM_CHECK(std::isfinite(coeff.real()) && std::isfinite(coeff.imag()),
              std::runtime_error,
              fmt::format("invalid coefficient ({}, {}); expected a finite "
                          "(i.e. either normal, subnormal or zero)",
                          coeff.real(), coeff.imag()));
    TCM_CHECK(_hamiltonian->max_index() < spin.size(), std::out_of_range,
              fmt::format("spin configuration too short {}; expected >{}",
                          spin.size(), _hamiltonian->max_index()));
    // `|_old⟩ := - coeff * root|spin⟩`
    _old.clear();
    _old.emplace(spin, -coeff * _roots[0]);
    // `|_old⟩ += coeff * H|spin⟩`
    (*_hamiltonian)(coeff, spin, _old);
    return kernel<1>();
#if 0
    using std::swap;
    TCM_CHECK(std::isfinite(coeff.real()) && std::isfinite(coeff.imag()),
              std::runtime_error,
              fmt::format("invalid coefficient ({}, {}); expected a finite "
                          "(i.e. either normal, subnormal or zero)",
                          coeff.real(), coeff.imag()));
    TCM_CHECK(_hamiltonian->max_index() < spin.size(), std::out_of_range,
              fmt::format("spin configuration too short {}; expected >{}",
                          spin.size(), _hamiltonian->max_index()));
    if (_roots.empty()) {
        _old.clear();
        _old.emplace(spin, coeff);
        return _old;
    }

    // The zeroth iteration: goal is to perform `_old := coeff * (H - root)|spin⟩`
    {
        // `|_old⟩ := - coeff * root|spin⟩`
        _old.clear();
        _old.emplace(spin, -_roots[0] * _scale * coeff);
        // `|_old⟩ += coeff * H|spin⟩`
        (*_hamiltonian)(_scale * coeff, spin, _old);
    }
    // Other iterations
    TCM_ASSERT(_current.empty(), "Bug!");
    for (auto i = size_t{1}; i < _roots.size(); ++i) {
        // Performs `|_current⟩ := (H - root)|_old⟩` in two steps:
        // 1) `|_current⟩ := - root|_old⟩`
        for (auto const& item : _old) {
            _current.emplace(item.first, -_roots[i] * _scale * item.second);
        }
        // 2) `|_current⟩ += H |_old⟩`
        for (auto const& item : _old) {
            (*_hamiltonian)(_scale * item.second, item.first, _current);
        }
        // |_old⟩ := |_current⟩, but to not waste allocated memory, we use
        // `swap + clear` instead.
        swap(_old, _current);
        _current.clear();
    }
    return _old;
#endif
}

auto Polynomial::iteration(complex_type root, QuantumState& current,
                           QuantumState const& old) const -> void
{
    TCM_ASSERT(current.empty(), "Bug!");
    if (_normalising) {
        auto norm = real_type{0};
        // Performs `|current⟩ := (H - root)|old⟩ / ‖old‖₂` in two steps:
        // 1) `|current⟩ := - root|old⟩`
        for (auto const& item : old) {
            current.emplace(item.first, -root * item.second);
            norm += std::norm(item.second);
        }
        // `|current⟩ := |current⟩ / ‖old‖₂`
        auto const scale = real_type{1} / std::sqrt(norm);
        for (auto& item : current) {
            item.second *= scale;
        }
        // 2) `|current⟩ += H |old⟩ / ‖old‖₂`
        for (auto const& item : old) {
            (*_hamiltonian)(item.second * scale, item.first, current);
        }
    }
    else {
        // Performs `|current⟩ := (H - root)|old⟩` in two steps:
        // 1) `|current⟩ := - root|old⟩`
        for (auto const& item : old) {
            current.emplace(item.first, -root * item.second);
        }
        // 2) `|current⟩ += H |old⟩`
        for (auto const& item : old) {
            (*_hamiltonian)(item.second, item.first, current);
        }
    }
}

template <size_t Offset> auto Polynomial::kernel() -> QuantumState const&
{
    using std::swap;
    for (auto i = Offset; i < _roots.size(); ++i) {
        // `|_current⟩ := (H - root)|_old⟩`
        iteration(_roots[i], _current, _old);
        // |_old⟩ := |_current⟩, but to not waste allocated memory, we use
        // `swap + clear` instead.
        swap(_old, _current);
        _current.clear();
    }
    return _old;
}

auto Polynomial::operator()(QuantumState const& state) -> QuantumState const&
{
    if (std::addressof(state) == std::addressof(_old)) { return kernel<0>(); }
    iteration(_roots[0], /*current=*/_old, /*old=*/state);
    return kernel<1>();
}

#if 0
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
#endif

#if 0
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
    auto const vectors =
        _polynomial->vectors().subspan(i * _batch_size, _batch_size);
    unpack_to_tensor(std::begin(vectors), std::end(vectors), _buffer);
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
    auto const size    = _polynomial->vectors().size();
    auto const rest    = size - i * _batch_size;
    TCM_ASSERT(rest < _batch_size, "Go use forward_propagate_batch instead");
    TCM_ASSERT(i * _batch_size + rest == size, "Precondition violated");
    TCM_ASSERT(_buffer.is_variable(), "");
    // Stores part of batch which we're given into `_input`.
    auto const vectors = _polynomial->vectors().subspan(i * _batch_size, rest);
    unpack_to_tensor(std::begin(vectors), std::end(vectors),
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
                      /*num_threads=*/static_cast<int>(_workers.size()));
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
#endif

auto load_forward_fn(std::string const& filename) -> ForwardT
{
    struct Function {
        torch::jit::script::Module _module;
        torch::jit::script::Method _method;
        torch::Tensor              _buffer;

        explicit Function(torch::jit::script::Module&& m)
            : _module{std::move(m)}
            , _method{_module.get_method("forward")}
            , _buffer{}
        {}

        auto operator()(gsl::span<SpinVector const> spins) -> torch::Tensor
        {
            TCM_CHECK(!spins.empty(), std::invalid_argument,
                      "empty batches are not supported");
            auto const batch_size  = static_cast<int64_t>(spins.size());
            auto const system_size = static_cast<int64_t>(spins[0].size());
            TCM_ASSERT(
                std::all_of(spins.begin() + 1, spins.end(),
                            [system_size](auto const& s) {
                                return s.size() == system_size;
                            }),
                "all spin configurations in the batch must have the same size");
            if (!_buffer.defined()) {
                _buffer = detail::make_tensor<float>(batch_size, system_size);
            }
            _buffer.resize_({batch_size, system_size});
            unpack_to_tensor(spins.begin(), spins.end(), _buffer);
            auto       output = _method({_buffer}).toTensor();
            auto const shape  = output.sizes();
            TCM_CHECK(((shape.size() == 1 && shape[0] == batch_size)
                       || (shape.size() == 2 && shape[0] == batch_size
                           && (shape[1] == 1 || shape[1] == 2))),
                      std::runtime_error,
                      fmt::format("output tensor has invalid shape: {}; "
                                  "expected [{}], [{}, 1] or [{}, 2]",
                                  fmt::join(shape, ", "), batch_size,
                                  batch_size, batch_size));
            return output;
        }
    };
    return [f = std::make_shared<Function>(torch::jit::load(filename))](
               auto const& x) { return (*f)(x); };
}

#if 0
auto load_forward_fn(std::string const& filename, size_t count)
    -> std::vector<ForwardT>
{
    static_assert(std::is_nothrow_move_constructible<ForwardT>::value,
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    static_assert(std::is_nothrow_move_assignable<ForwardT>::value,
                  TCM_STATIC_ASSERT_BUG_MESSAGE);

    std::vector<ForwardT> modules;
    modules.resize(count);
    std::atomic_flag   err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr err_ptr  = nullptr;

    auto* modules_ptr = modules.data();
#pragma omp parallel for num_threads(count) default(none)                      \
    firstprivate(count, modules_ptr) shared(filename, err_flag, err_ptr)
    for (auto i = size_t{0}; i < count; ++i) {
        try {
            modules_ptr[i] = load_forward_fn(filename);
        }
        catch (...) {
            if (!err_flag.test_and_set()) {
                err_ptr = std::current_exception();
            }
        }
    }
    if (err_ptr != nullptr) { std::rethrow_exception(err_ptr); }
    return modules;
}
#endif

auto bind_heisenberg(pybind11::module m) -> void
{
    namespace py = pybind11;
    py::class_<Heisenberg, std::shared_ptr<Heisenberg>>(m, "Heisenberg")
        .def(py::init<Heisenberg::spec_type>(), py::arg{"edges"},
             R"EOF(
                 Creates an isotropic Heisenberg Hamiltonian from a list of edges.

                 :param edges: A list of tuples ``(coupling, i, j)``.
             )EOF")
        .def(
            "__len__", [](Heisenberg const& self) { return self.size(); },
            R"EOF(
                 Returns the number of edges in the graph.
            )EOF")
        .def_property_readonly(
            "edges", [](Heisenberg const& self) { return self.edges(); },
            R"EOF(
                 Returns graph edges

                 .. warning:: This function copies the edges
            )EOF");
}

auto bind_explicit_state(pybind11::module m) -> void
{
    py::class_<QuantumState>(m, "ExplicitState",
                             R"EOF(
            Quantum state |ψ⟩=∑cᵢ|σᵢ⟩ backed by a table {(σᵢ, cᵢ)}.
        )EOF")
        .def("__getitem__",
             [](QuantumState const& self, SpinVector const& spin) {
                 auto i = self.find(spin);
                 if (i != self.end()) { return i->second; }
                 throw py::key_error{};
             })
        .def("__setitem__",
             [](QuantumState& self, SpinVector const& spin,
                complex_type const& value) {
                 TCM_CHECK(
                     std::isfinite(value.real()) && std::isfinite(value.imag()),
                     std::runtime_error,
                     fmt::format(
                         "invalid value ({}, {}); expected a finite (i.e. "
                         "either normal, subnormal or zero) complex float",
                         value.real(), value.imag()));
                 auto i = self.find(spin);
                 if (i != self.end()) { i->second = value; }
                 throw py::key_error{};
             })
        .def(
            "__len__", [](QuantumState const& self) { return self.size(); },
            R"EOF(Returns number of elements in |ψ⟩.)EOF")
        .def(
            "keys", [](QuantumState const& self) { return keys(self); },
            R"EOF(Returns basis vectors {|σᵢ⟩} as a ``numpy.ndarray``.)EOF")
        .def(
            "values",
            [](QuantumState const& self, bool only_real) {
                return values(self, only_real);
            },
            pybind11::arg{"only_real"} = true,
            R"EOF(
                Returns coefficients {cᵢ} or {Re[cᵢ]} (depending on the value
                of ``only_real``) as a ``torch.Tensor``.
            )EOF");
}

auto bind_polynomial(pybind11::module m) -> void
{
    namespace py = pybind11;
    py::class_<Polynomial, std::shared_ptr<Polynomial>>(m, "Polynomial",
                                                        R"EOF(
            Represents polynomials in H.
        )EOF")
        .def(py::init([](std::shared_ptr<Heisenberg const> h,
                         std::vector<complex_type> roots, bool normalising) {
                 return std::make_shared<Polynomial>(
                     std::move(h), std::move(roots), normalising);
             }),
             py::arg{"hamiltonian"}, py::arg{"roots"},
             py::arg{"normalising"} = false,
             R"EOF(
                 Given a Hamiltonian H and roots {rᵢ} (i ∈ {0, 1, ..., n-1})
                 constructs the following polynomial

                     (H - rₙ₋₁)...(H - r₂)(H - r₁)(H - r₀)

                 Even though each rᵢ is complex, after expanding the brackets
                 __all coefficients should be real__.
             )EOF")
        // .def_property_readonly(
        //     "size", [](Polynomial const& self) { return self.size(); },
        //     R"EOF(
        //         Returns the number of different spin configurations in the current
        //         state. This attribute is only useful after applying the polynomial
        //         to a state |σ⟩.
        //     )EOF")
        // .def(
        //     "__len__", [](Polynomial const& self) { return self.size(); },
        //     R"EOF(Same as ``size``.)EOF")
        .def(
            "__call__",
            [](Polynomial& self, complex_type coeff, SpinVector spin) {
                return self(coeff, spin);
            },
            py::arg{"coeff"}, py::arg{"spin"},
            py::return_value_policy::reference_internal,
            R"EOF(

            )EOF")
        .def(
            "__call__",
            [](Polynomial& self, QuantumState const& state) {
                return self(state);
            },
            py::arg{"state"}, py::return_value_policy::reference_internal,
            R"EOF(

            )EOF");
    // .def(
    //     "vectors",
    //     [](Polynomial const& p) {
    //         auto const spins = p.vectors();
    //         return unpack_to_tensor(std::begin(spins), std::end(spins));
    //     },
    //     py::return_value_policy::move)
    // .def("coefficients", [](Polynomial const& p) {
    //     return torch::Tensor{p.coefficients()};
    // });
}

#if 0
auto bind_polynomial_state(pybind11::module m) -> void
{
    namespace py = pybind11;
    py::class_<PolynomialState>(m, "TargetState")
        .def(py::init([](std::string const& filename, Polynomial const& poly,
                         std::tuple<size_t, size_t> dim, int num_threads) {
#if 0
            return std::make_unique<PolynomialState>(
                ::TCM_NAMESPACE::detail::load_forward_fn(filename), poly,
                batch_size, filename);
#else
            if (num_threads <= 0) { num_threads = omp_get_max_threads(); }
            return std::make_unique<PolynomialState>(
                tcm::detail::load_forward_fn(
                    filename, static_cast<unsigned>(num_threads)),
                Polynomial{poly, SplitTag{}}, dim);
#endif
        }), py::arg{"filename"}, py::arg{"poly"}, py::arg{"dim"}, py::arg{"num_threads"})
        .def(py::init([](AmplitudeNet const& net, Polynomial const& poly,
                         std::tuple<size_t, size_t> dim, int num_threads) {
            if (num_threads <= 0) { num_threads = omp_get_max_threads(); }
            std::vector<ForwardT> forward;
            for (auto i = 0; i < num_threads; ++i) {
                forward.emplace_back([m = std::make_shared<AmplitudeNet>(net)](
                                         auto const& x) { return (*m)(x); });
            }
            return std::make_unique<PolynomialState>(
                std::move(forward), Polynomial{poly, SplitTag{}}, dim);
        }), py::arg{"net"}, py::arg{"poly"}, py::arg{"dim"}, py::arg{"num_threads"})
        .def("__call__",
             [](PolynomialState& state, SpinVector const input) {
                 // mkl_set_num_threads(1);
                 // torch::NoGradGuard no_grad;
                 return state(input);
             }, py::call_guard<py::gil_scoped_release>())
        .def("__call__",
             [](PolynomialState& state, torch::Tensor const& input) {
                 auto const dim = input.dim();
                 TCM_CHECK_DIM(dim, 1, 2);
                 if (dim == 1) {
                     auto out = tcm::detail::make_tensor<float>(1);
                     out[0]   = state(SpinVector{input});
                     return out;
                 }
                 auto const size = input.size(0);
                 auto       out =
                     tcm::detail::make_tensor<float>(static_cast<size_t>(size));
                 auto out_accessor = out.accessor<float, 1>();
                 for (auto i = int64_t{0}; i < size; ++i) {
                     out_accessor[i] = state(SpinVector{input[i]});
                 }
                 return out;
             })
        .def_property_readonly(
            "time_poly",
            [](PolynomialState const& self) { return self.time_poly(); })
        .def_property_readonly("time_psi", [](PolynomialState const& self) {
            return self.time_psi();
        });

}
#endif

TCM_NAMESPACE_END
