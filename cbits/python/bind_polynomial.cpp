#include "bind_polynomial.hpp"
#include "../heisenberg.hpp"
#include "../polynomial.hpp"
#include "../trim.hpp"
#include "pybind11_helpers.hpp"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/script.h>

namespace py = pybind11;

TCM_NAMESPACE_BEGIN

template <class Hamiltonian>
auto make_polynomial(py::module m, char const* class_name)
{
    py::class_<Polynomial<Hamiltonian>,
               std::shared_ptr<Polynomial<Hamiltonian>>>(m, "Polynomial")
        .def(py::init<std::shared_ptr<Hamiltonian const>,
                      std::vector<complex_type>, bool>(),
             py::arg{"hamiltonian"}, py::arg{"roots"},
             py::arg{"normalising"} = false)
        .def(
            "__call__",
            [](Polynomial<Hamiltonian>& self, bits512 const& spin)
                -> QuantumState const& { return self(spin); },
            py::return_value_policy::reference_internal);
}

TCM_EXPORT auto bind_polynomial(PyObject* _module) -> void
{
    auto m = py::module{py::reinterpret_borrow<py::object>(_module)};

    std::vector<std::string> keep_alive;
#define DOC(str) trim(keep_alive, str)

    py::class_<QuantumState>(m, "ExplicitState", DOC(R"EOF(
            Quantum state |ψ⟩=∑cᵢ|σᵢ⟩ backed by a table {(σᵢ, cᵢ)}.
        )EOF"))
        .def("__contains__",
             [](QuantumState const& self, QuantumState::key_type const& spin) {
                 return self.find(spin) != self.end();
             })
        .def("__getitem__",
             [](QuantumState const& self, QuantumState::key_type const& spin) {
                 auto i = self.find(spin);
                 if (i != self.end()) { return i->second; }
                 throw py::key_error{};
             })
        .def("__setitem__",
             [](QuantumState& self, QuantumState::key_type const& spin,
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
            "__len__", [](QuantumState const& self) { return self.size(); },
            R"EOF(Returns number of elements in |ψ⟩.)EOF")
        .def(
            "__iter__",
            [](QuantumState const& self) {
                return py::make_iterator(self.begin(), self.end());
            },
            py::keep_alive<0, 1>());

    make_polynomial<Heisenberg>(m, "Polynomial");
    // .def(
    //     "keys",
    //     [](QuantumState const& self) {
    //         return vector_to_numpy(keys(self));
    //     },
    //     R"EOF(Returns basis vectors {|σᵢ⟩} as a ``numpy.ndarray``.)EOF")
    // .def(
    //     "values",
    //     [](QuantumState const& self, bool only_real) {
    //         return values(self, only_real);
    //     },
    //     pybind11::arg{"only_real"} = false,
    //     R"EOF(
    //         Returns coefficients {cᵢ} or {Re[cᵢ]} (depending on the value
    //         of ``only_real``) as a ``torch.Tensor``.
    //     )EOF");

#undef DOC
}

TCM_NAMESPACE_END
