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

#include "nqs.hpp"
#include "trim.hpp"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace pybind11 {
namespace detail {
    template <class T> struct type_caster<gsl::span<T>> {
      public:
        /**
         * This macro establishes the name 'inty' in
         * function signatures and declares a local variable
         * 'value' of type inty
         */
        PYBIND11_TYPE_CASTER(gsl::span<T>, _("numpy.ndarray[")
                                               + npy_format_descriptor<T>::name
                                               + _("]"));

        /**
         * Conversion part 1 (Python->C++): convert a PyObject into a inty
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool)
        {
            using ArrayT = array_t<std::remove_const_t<T>, array::c_style>;
            if (!ArrayT::check_(src))
                return false;
            auto array = ArrayT::ensure(src);
            if (!array || array.ndim() != 1) return false;
            auto size = static_cast<size_t>(array.shape()[0]);
            value     = gsl::span<T>{
                std::is_const_v<T> ? array.data() : array.mutable_data(), size};
            return true;
        }
    };
} // namespace detail
} // namespace pybind11

namespace py = pybind11;

namespace {
auto bind_polynomial_state(py::module m) -> void
{
    using namespace tcm;

    py::class_<PolynomialStateV2>(m, "PolynomialState")
        .def(py::init([](std::shared_ptr<Polynomial> polynomial,
                         std::string const&          state,
                         std::pair<size_t, size_t>   input_shape) {
            return std::make_unique<PolynomialStateV2>(
                std::move(polynomial), load_forward_fn(state), input_shape);
        }))
        .def("__call__", [](PolynomialStateV2&                          self,
                            py::array_t<SpinVector, py::array::c_style> spins) {
            return self({spins.data(0), static_cast<size_t>(spins.shape(0))});
        });
}

template <class T, int Flags>
decltype(auto) to_const(py::array_t<T, Flags>& array)
{
    // NOTE: Yes, it's not nice to use pybind11::detail, but currently,
    // there's no other way...
    pybind11::detail::array_proxy(array.ptr())->flags &=
        ~pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_;
    return array;
}

template <class T>
auto span_to_numpy(gsl::span<T> xs, py::object base) -> py::array
{
    using type = std::remove_const_t<T>;
    auto a     = py::array_t<type>{xs.size(), xs.data(), base};
    if (std::is_const<T>::value) { to_const(a); }
    return a;
}

auto states_to_numpy(
    typename ::TCM_NAMESPACE::detail::BasisCache::StatesT const& states,
    py::object base) -> py::array
{
    using ::TCM_NAMESPACE::detail::BasisCache;
    return span_to_numpy(gsl::span{states.data(), states.size()}, base);
}

auto states_from_numpy(py::array array)
{
    return array.cast<typename ::TCM_NAMESPACE::detail::BasisCache::StatesT>();
}

auto ranges_to_numpy(
    typename ::TCM_NAMESPACE::detail::BasisCache::RangesT const& ranges,
    py::object base) -> py::array
{
    auto a =
        py::array_t<uint64_t>{{static_cast<int64_t>(ranges.size()), 2L},
                              reinterpret_cast<uint64_t const*>(ranges.data()),
                              base};
    to_const(a);
    return a;
}

auto ranges_from_numpy(py::array array)
{
    TCM_CHECK(
        array.ndim() == 2 && array.shape(1) == 2, std::invalid_argument,
        fmt::format(
            "array has invalid shape: [{}]; expected a 2D array with 2 columns",
            fmt::join(array.shape(), array.shape() + array.ndim(), ", ")));
    typename ::TCM_NAMESPACE::detail::BasisCache::RangesT ranges;
    ranges.reserve(static_cast<size_t>(array.shape(0)));
    auto const* p =
        static_cast<std::pair<uint64_t, uint64_t> const*>(array.data());
    for (auto i = int64_t{0}; i < array.shape(0); ++i, ++p) {
        ranges.push_back(*p);
    }
    return ranges;
}

template <class T, class Allocator>
auto vector_to_numpy(std::vector<T, Allocator>&& xs) -> py::array
{
    using V         = std::vector<T, Allocator>;
    auto const data = xs.data();
    auto const size = xs.size();
    auto       base = py::capsule{new V{std::move(xs)},
                            [](void* p) { delete static_cast<V*>(p); }};
    return py::array_t<T>{size, data, std::move(base)};
}

auto bind_spin_basis(py::module m) -> void
{
    using namespace tcm;
    m = m.def_submodule("v2");
    std::vector<std::string> keep_alive;

#define DOC(str) trim(keep_alive, str)

    py::class_<Symmetry>(m, "Symmetry", DOC(R"EOF(
        A symmetry operator with optimised ``__call__`` member function. It
        stores three things: a Benes network to permute bits, periodicity of
        the permutation, and the symmetry sector.
        )EOF"))
        .def(py::init(
                 [](std::pair<std::array<uint64_t, 6>, std::array<uint64_t, 6>>
                             network,
                    unsigned sector, unsigned periodicity) {
                     return Symmetry{
                         {network.first, network.second}, sector, periodicity};
                 }),
             DOC(R"EOF(
             Initialises the symmetry given a Benes network, symmetry sector, and
             periodicity of the operator.

             :param network: a Benes network for permuting 64-bit integers. It
                is given as a pair of lists of masks. *Validity of the network is
                not checked!*
             :param sector: symmetry sector. It must be in ``[0, periodicity)``.
                Eigenvalue of the operator is then ``exp(-2πi * sector/periodicity)``.
             :param periodicity: a positive integer specifying the periodicity
                of the operator.
             )EOF"),
             py::arg{"network"}, py::arg{"sector"}, py::arg{"periodicity"})
        .def_property_readonly("sector", &Symmetry::sector)
        .def_property_readonly("periodicity", &Symmetry::periodicity)
        .def_property_readonly("eigenvalue", &Symmetry::eigenvalue)
        .def(
            "__call__",
            [](Symmetry const& self, uint64_t const x) { return self(x); },
            DOC(R"EOF(
            Permutes bits of ``x`` according to the underlying permutation.
            )EOF"),
            py::arg{"x"})
        .def(py::pickle(
            [](Symmetry const& self) { return self._state_as_tuple(); },
            [](decltype(std::declval<Symmetry const&>()._state_as_tuple())
                   const& state) {
                return Symmetry{{std::get<0>(state), std::get<1>(state)},
                                std::get<2>(state),
                                std::get<3>(state)};
            }));

    py::class_<SpinBasis, std::shared_ptr<SpinBasis>>(m, "SpinBasis")
        .def(py::init<std::vector<Symmetry>, unsigned,
                      std::optional<unsigned>>(),
             DOC(R"EOF(
             Initialises the basis without constructing the internal cache.
             Memory usage thus does not increase with system size (only with
             number of symmetries, but this is negligible).

             :param symmetries: a list of symmetries. It must be either empty or
                form a group. This is unchecked, but bad stuff will happen is
                this requirement is not fulfilled.
             :param number_spins: number of particles in the system.
             :param hamming_weight: Hamming weight of states, i.e. number of
                 spin ups. Specifying it is equivalent to restricting the
                 Hilbert space to some magnetisation sector.
             )EOF"),
             py::arg{"symmetries"}, py::arg{"number_spins"},
             py::arg{"hamming_weight"})
        .def("build", &SpinBasis::build, DOC(R"EOF(
             Constructs a list of representative vectors. After this operation,
             methods such as ``index``, ``number_states``, and ``states`` can be
             used.
             )EOF"))
        .def_property_readonly(
            "states",
            [](py::object self) {
                return span_to_numpy(self.cast<SpinBasis const&>().states(),
                                     self);
            },
            DOC(R"EOF(
            Array of representative states. This property is available only
            after a call to ``build()``.
            )EOF"))
        .def_property_readonly("number_spins", &SpinBasis::number_spins,
                               DOC(R"EOF(
            Number of particles in the system.
            )EOF"))
        .def_property_readonly("number_states", &SpinBasis::number_states,
                               DOC(R"EOF(
            Number of states in the basis. This property is available only
            after a call to ``build()``.
            )EOF"))
        .def("index", &SpinBasis::index, DOC(R"EOF(
             Given a representative state ``x``, returns its index in the array
             of representatives (i.e. ``self.states``).
             )EOF"),
             py::arg{"x"})
        .def("normalisation", &SpinBasis::normalisation, DOC(R"EOF(
             Given a representative state ``x``, returns its "norm".
             )EOF"),
             py::arg{"x"})
        .def("representative", &SpinBasis::representative, DOC(R"EOF(
             Given a state ``x``, returns its representative.
             )EOF"))
        .def("full_info", &SpinBasis::full_info, DOC(R"EOF(
             Given a state ``x``, returns a tuple ``(r, λ, N)`` where ``r`` is
             the representative of ``x``, ``N`` is the "norm" of ``r``, and λ is
             the eigenvalue of a group element ``g`` such that ``gr == x``.
             )EOF"))
        .def(py::pickle(
            [](py::object _self) {
                auto const& self = _self.cast<SpinBasis const&>();
                auto const& [symmetries, number_spins, hamming_weight, cache] =
                    self._get_state();
                auto cache_state = py::none();
                if (cache.has_value()) {
                    auto const& [states, ranges] = cache->_get_state();
                    cache_state =
                        py::make_tuple(states_to_numpy(states, _self),
                                       ranges_to_numpy(ranges, _self));
                }
                return py::make_tuple(symmetries, number_spins, hamming_weight,
                                      cache_state);
            },
            [](py::tuple state) {
                TCM_CHECK(state.size() == 4, std::runtime_error,
                          fmt::format("state has wrong length: {}; expected a "
                                      "state of length 4",
                                      state.size()));
                auto symmetries     = state[0].cast<std::vector<Symmetry>>();
                auto number_spins   = state[1].cast<unsigned>();
                auto hamming_weight = state[2].cast<std::optional<unsigned>>();
                using ::TCM_NAMESPACE::detail::BasisCache;
                auto cache = std::optional<BasisCache>{std::nullopt};
                if (!state[3].is_none()) {
                    auto cache_state = py::tuple{state[3]};
                    auto states = states_from_numpy(py::array{cache_state[0]});
                    auto ranges = ranges_from_numpy(py::array{cache_state[1]});
                    cache.emplace(std::move(states), std::move(ranges));
                }
                return std::make_shared<SpinBasis>(
                    std::move(symmetries), number_spins,
                    std::move(hamming_weight), std::move(cache));
            }))
        .def("expand", [](SpinBasis const& self, gsl::span<float const> xs,
                          gsl::span<SpinBasis::StateT const> states) {
            py::array_t<float> ys{states.size()};
            ::TCM_NAMESPACE::v2::expand_states<float>(
                self, xs, states, {ys.mutable_data(), states.size()});
            return ys;
        });

    m.def(
        "unpack",
        [](py::array_t<SpinBasis::StateT, py::array::c_style> states,
           unsigned const number_spins, std::optional<torch::Tensor> out) {
            TCM_CHECK(0 < number_spins && number_spins <= 64,
                      std::invalid_argument,
                      fmt::format("invalid number_spins: {}; expected a "
                                  "positive integer smaller than 64",
                                  number_spins));
            auto const* shape = states.shape();
            TCM_CHECK(
                states.ndim() == 1, std::invalid_argument,
                fmt::format("states has invalid shape: [{}]; expected a "
                            "one-dimensional array",
                            fmt::join(shape, shape + states.ndim(), ", ")));
            if (!out.has_value()) {
                out.emplace(::TCM_NAMESPACE::detail::make_tensor<float>(
                    shape[0], number_spins));
            }
            auto const in = gsl::span<SpinBasis::StateT const>{
                static_cast<SpinBasis::StateT const*>(states.data()),
                static_cast<size_t>(shape[0])};
            v2::unpack(in.begin(), in.end(), number_spins, *out);
            return *std::move(out);
        },
        DOC(R"EOF(
            Given a 1D array of states, returns a 2D tensor of float32 where
            each bit is replaced by a float (0 becomes -1.0 and 1 becomes 1.0).

            This operation is uses SIMD intrinsics under the hood and is
            relatively efficient. It is thus okay to call it on every batch
            before propagating it through a neural network.
            )EOF"),
        py::arg{"states"}, py::arg{"number_spins"},
        py::arg{"out"} = py::none{}, py::call_guard<py::gil_scoped_release>());

    m.def(
        "unpack",
        [](py::array_t<SpinBasis::StateT, py::array::c_style> states,
           py::array_t<int64_t, py::array::c_style>           indices,
           unsigned const number_spins, std::optional<torch::Tensor> out) {
            TCM_CHECK(0 < number_spins && number_spins <= 64,
                      std::invalid_argument,
                      fmt::format("invalid number_spins: {}; expected a "
                                  "positive integer smaller than 64",
                                  number_spins));
            auto const* states_shape = states.shape();
            TCM_CHECK(
                states.ndim() == 1, std::invalid_argument,
                fmt::format("states has invalid shape: [{}]; expected a "
                            "one-dimensional array",
                            fmt::join(states_shape,
                                      states_shape + states.ndim(), ", ")));
            auto const* indices_shape = indices.shape();
            TCM_CHECK(
                indices.ndim() == 1, std::invalid_argument,
                fmt::format("indices has invalid shape: [{}]; expected a "
                            "one-dimensional array",
                            fmt::join(indices_shape,
                                      indices_shape + indices.ndim(), ", ")));
            auto const in = gsl::span<int64_t const>{
                static_cast<int64_t const*>(indices.data()),
                static_cast<size_t>(indices_shape[0])};
            if (!out.has_value()) {
                out.emplace(::TCM_NAMESPACE::detail::make_tensor<float>(
                    indices_shape[0], number_spins));
            }
            auto const projection =
                [p = static_cast<SpinBasis::StateT const*>(states.data()),
                 n = states_shape[0]](auto const index) {
                    TCM_CHECK(
                        0 <= index && index <= n, std::out_of_range,
                        fmt::format("indices contains an index which is out of "
                                    "range: {}; expected an index in [{}, {})",
                                    index, 0, n));
                    return p[index];
                };
            v2::unpack(in.begin(), in.end(), number_spins, *out, projection);
            return *std::move(out);
        },
        DOC(R"EOF(
            Given a 1D array of states, returns a 2D tensor of float32 where
            each bit is replaced by a float (0 becomes -1.0 and 1 becomes 1.0).

            This operation is uses SIMD intrinsics under the hood and is
            relatively efficient. It is thus okay to call it on every batch
            before propagating it through a neural network.
            )EOF"),
        py::arg{"states"}, py::arg{"indices"}, py::arg{"number_spins"},
        py::arg{"out"} = py::none{}, py::call_guard<py::gil_scoped_release>());

    py::class_<SpinDataset>(m, "SpinDataset")
        .def(py::init<torch::Tensor, torch::Tensor, size_t>(), py::arg{"spins"},
             py::arg{"values"}, py::arg{"number_spins"})
        .def("fetch",
             [](SpinDataset const& self, int64_t const first,
                int64_t const last, py::object device, bool const pin_memory) {
                 return self.fetch(
                     first, last,
                     torch::python::detail::py_object_to_device(device),
                     pin_memory);
             })
        .def("fetch", [](SpinDataset const&             self,
                         gsl::span<int64_t const> const indices,
                         py::object device, bool const pin_memory) {
            return self.fetch(
                indices, torch::python::detail::py_object_to_device(device),
                pin_memory);
        });

    py::class_<ChunkLoader>(m, "ChunkLoader")
        .def(py::init([](SpinDataset dataset, size_t chunk_size, bool shuffle,
                         py::object device, bool pin_memory) {
                 return std::make_unique<ChunkLoader>(
                     std::move(dataset), chunk_size, shuffle,
                     torch::python::detail::py_object_to_device(device),
                     pin_memory);
             }),
             py::arg{"dataset"}, py::arg{"chunk_size"}, py::arg{"shuffle"},
             py::arg{"device"}, py::arg{"pin_memory"})
        .def("__iter__",
             [](py::object self) {
                 self.cast<ChunkLoader&>().reset();
                 return self;
             })
        .def("__next__", [](ChunkLoader& self) {
            auto batch = self.next();
            if (batch.has_value()) { return *std::move(batch); }
            throw py::stop_iteration{};
        });

    py::class_<v2::Heisenberg, std::shared_ptr<v2::Heisenberg>>(m, "Heisenberg")
        .def(py::init<v2::Heisenberg::spec_type,
                      std::shared_ptr<SpinBasis const>>())
        // .def_property_readonly("edges", &v2::Heisenberg::edges)
        .def_property_readonly("basis", &v2::Heisenberg::basis)
        .def_property_readonly("is_real", &v2::Heisenberg::is_real)
        .def("__call__",
             [](v2::Heisenberg const& self, SpinBasis::StateT const x) {
                 auto [buffer, size] = self(x);
                 return std::vector<decltype(buffer)::element_type>(
                     buffer.get(), buffer.get() + size);
             })
        .def("__call__",
             [](v2::Heisenberg const&                                 self,
                py::array_t<std::complex<double>, py::array::c_style> x,
                py::array_t<std::complex<double>, py::array::c_style> y) {
                 self(
                     gsl::span<std::complex<double> const>{
                         x.data(), static_cast<size_t>(x.shape(0))},
                     gsl::span<std::complex<double>>{
                         y.mutable_data(), static_cast<size_t>(y.shape(0))});
             })
        .def("__call__",
             [](v2::Heisenberg const&                   self,
                py::array_t<double, py::array::c_style> x,
                py::array_t<double, py::array::c_style> y) {
                 self(gsl::span<double const>{x.data(),
                                              static_cast<size_t>(x.shape(0))},
                      gsl::span<double>{y.mutable_data(),
                                        static_cast<size_t>(y.shape(0))});
             })
        .def("__call__", [](v2::Heisenberg const&                  self,
                            py::array_t<float, py::array::c_style> x,
                            py::array_t<float, py::array::c_style> y) {
            self(gsl::span<float const>{x.data(),
                                        static_cast<size_t>(x.shape(0))},
                 gsl::span<float>{y.mutable_data(),
                                  static_cast<size_t>(y.shape(0))});
        });

    py::class_<v2::QuantumState>(m, "ExplicitState",
                                 R"EOF(
            Quantum state |ψ⟩=∑cᵢ|σᵢ⟩ backed by a table {(σᵢ, cᵢ)}.
        )EOF")
        .def("__getitem__",
             [](v2::QuantumState const&           self,
                v2::QuantumState::key_type const& spin) {
                 auto i = self.find(spin);
                 if (i != self.end()) { return i->second; }
                 throw py::key_error{};
             })
        .def("__setitem__",
             [](v2::QuantumState& self, v2::QuantumState::key_type const& spin,
                complex_type const& value) {
                 TCM_CHECK(
                     std::isfinite(value.real()) && std::isfinite(value.imag()),
                     std::runtime_error,
                     fmt::format(
                         "invalid value {} + {}j; expected a finite (i.e. "
                         "either normal, subnormal or zero) complex float",
                         value.real(), value.imag()));
                 auto i = self.find(spin);
                 if (i != self.end()) { i->second = value; }
                 throw py::key_error{};
             })
        .def(
            "__len__", [](v2::QuantumState const& self) { return self.size(); },
            R"EOF(Returns number of elements in |ψ⟩.)EOF")
        .def(
            "keys",
            [](v2::QuantumState const& self) {
                return vector_to_numpy(keys(self));
            },
            R"EOF(Returns basis vectors {|σᵢ⟩} as a ``numpy.ndarray``.)EOF")
        .def(
            "values",
            [](v2::QuantumState const& self, bool only_real) {
                return values(self, only_real);
            },
            pybind11::arg{"only_real"} = false,
            R"EOF(
                Returns coefficients {cᵢ} or {Re[cᵢ]} (depending on the value
                of ``only_real``) as a ``torch.Tensor``.
            )EOF");

    py::class_<v2::Polynomial, std::shared_ptr<v2::Polynomial>>(m, "Polynomial")
        .def(py::init([](std::shared_ptr<v2::Heisenberg const> h,
                         std::vector<complex_type> roots, bool normalising) {
                 return std::make_shared<v2::Polynomial>(
                     std::move(h), std::move(roots), normalising);
             }),
             py::arg{"hamiltonian"}, py::arg{"roots"},
             py::arg{"normalising"} = false, DOC(R"EOF(
             Given a Hamiltonian H and roots {rᵢ} (i ∈ {0, 1, ..., n-1})
             constructs the following polynomial

                 P = (H - rₙ₋₁)...(H - r₂)(H - r₁)(H - r₀)

             :param hamiltonian: Hamiltonian H.
             :param roots: a list of roots {rᵢ}.
             :param normalising: if True, scaling coefficients will be between
                brackets to ensure that the norm of P is around 1.
             )EOF"))
        .def_property_readonly("hamiltonian", &v2::Polynomial::hamiltonian)
        .def(
            "__call__",
            [](v2::Polynomial& self, complex_type coeff,
               SpinBasis::StateT spin) { return self(coeff, spin); },
            py::arg{"coeff"}, py::arg{"spin"},
            py::return_value_policy::reference_internal)
        .def(
            "__call__",
            [](v2::Polynomial& self, v2::QuantumState const& state) {
                return self(state);
            },
            py::arg{"state"}, py::return_value_policy::reference_internal);

    py::class_<v2::PolynomialState>(m, "PolynomialState")
        .def(py::init([](std::shared_ptr<v2::Polynomial> polynomial,
                         torch::jit::script::Method      forward,
                         unsigned const                  internal_batch_size) {
            struct Function {
                using StateT = v2::PolynomialState::StateT;
                torch::jit::script::Method _method;
                torch::Tensor              _buffer;
                SpinBasis const*           _basis;

                auto operator()(gsl::span<StateT const> spins) -> torch::Tensor
                {
                    TCM_CHECK(!spins.empty(), std::invalid_argument,
                              "empty batches are not supported");
                    auto const batch_size = static_cast<int64_t>(spins.size());
                    auto const system_size =
                        static_cast<int64_t>(_basis->number_spins());
                    if (!_buffer.defined()) {
                        _buffer = ::TCM_NAMESPACE::detail::make_tensor<float>(
                            batch_size, system_size);
                    }
                    _buffer.resize_({batch_size, system_size});
                    v2::unpack(spins.begin(), spins.end(),
                               _basis->number_spins(), _buffer);
                    auto       output = _method({_buffer}).toTensor();
                    auto const shape  = output.sizes();
                    TCM_CHECK((shape.size() == 2 && shape[0] == batch_size
                               && shape[1] == 2),
                              std::runtime_error,
                              fmt::format("output tensor has invalid "
                                          "shape: [{}]; expected [{}, 2]",
                                          fmt::join(shape, ", "), batch_size));
                    return output;
                }
            };

            auto* basis = polynomial->hamiltonian()->basis().get();
            return std::make_unique<v2::PolynomialState>(
                std::move(polynomial), Function{std::move(forward), {}, basis},
                internal_batch_size);
        }))
        .def(py::init(
            [](std::shared_ptr<v2::Polynomial> polynomial,
               std::function<torch::Tensor(std::vector<SpinBasis::StateT>)>
                              forward,
               unsigned const internal_batch_size) {
                auto f = std::make_shared<decltype(forward)>(forward);
                return std::make_unique<v2::PolynomialState>(
                    std::move(polynomial),
                    [f](auto spins) {
                        return (*f)(std::vector<SpinBasis::StateT>{
                            spins.begin(), spins.end()});
                    },
                    internal_batch_size);
            }))
        .def("__call__",
             [](v2::PolynomialState&                               self,
                py::array_t<SpinBasis::StateT, py::array::c_style> states) {
                 TCM_CHECK(
                     states.ndim() == 1, std::invalid_argument,
                     fmt::format("states has invalid shape: [{}]; expected a "
                                 "one-dimensional array",
                                 fmt::join(states.shape(),
                                           states.shape() + states.ndim(),
                                           ", ")));
                 auto xs = gsl::span<SpinBasis::StateT const>{
                     states.data(), static_cast<size_t>(states.shape(0))};
                 return self(xs);
             });

    m.def("get_representative",
          [](std::vector<Symmetry> const& symmetries, uint64_t const x) {
              return tcm::detail::find_representative(symmetries.begin(),
                                                      symmetries.end(), x);
          });

    m.def("get_info", [](std::vector<Symmetry> const& symmetries,
                         uint64_t const               x) {
        std::vector<std::pair<uint64_t, double>> workspace(symmetries.size());
        return tcm::detail::get_info(symmetries.begin(), symmetries.end(), x);
    });

#if 0
    m.def("generate_states", [](std::vector<Symmetry> const& symmetries,
                                unsigned                     number_spins,
                                std::optional<unsigned>      hamming_weight) {
        return vector_to_numpy(
            tcm::detail::generate_states(symmetries.begin(), symmetries.end(),
                                         number_spins, hamming_weight));
    });
#endif

    m.def("generate_ranges",
          [](std::vector<Symmetry::UInt> const& states, unsigned number_spins) {
              return tcm::detail::generate_ranges<4>(
                  states.begin(), states.end(), number_spins);
          });

    m.def("test_script", [](torch::jit::script::Module const& module) {
        py::print("Yes!");
    });

    m.def("test_method", [](torch::jit::script::Method const& module) {
        py::print("Yes!");
    });

    m.def("test_load_script", [](std::string const& module) {
        return torch::jit::load(module);
    });

    m.def("generate_states_parallel",
          [](std::vector<Symmetry> const& symmetries, unsigned number_spins,
             std::optional<unsigned> hamming_weight) {
              return vector_to_numpy(
                  ::TCM_NAMESPACE::detail::generate_states_parallel(
                      symmetries, number_spins, std::move(hamming_weight)));
          });

#undef DOC
}
} // namespace


#if defined(TCM_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-prototypes"
#endif
PYBIND11_MODULE(_C_nqs, m)
{
#if defined(TCM_CLANG)
#    pragma clang diagnostic pop
#endif
    m.doc() = R"EOF()EOF";

    using namespace tcm;

    bind_spin(m.ptr());
    bind_spin_basis(m);
    bind_heisenberg(m);
    bind_explicit_state(m);
    bind_polynomial(m);
    // bind_options(m);
    // bind_chain_result(m);
    // bind_sampling(m);
    // bind_networks(m);
    // bind_dataloader(m);
    bind_monte_carlo(m.ptr());
    bind_polynomial_state(m);
}
