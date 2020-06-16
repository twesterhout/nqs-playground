#include "bind_polynomial.hpp"
#include "../heisenberg.hpp"
#include "../simple_polynomial.hpp"
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

// template <class Hamiltonian>
auto make_polynomial(py::module m, char const* class_name)
{
#if 0
    py::class_<Polynomial<Hamiltonian>,
               std::shared_ptr<Polynomial<Hamiltonian>>>(m, "Polynomial")
        .def(py::init<std::shared_ptr<Hamiltonian const>,
                      std::vector<complex_type>, bool, int>(),
             py::arg{"hamiltonian"}, py::arg{"roots"},
             py::arg{"normalising"} = false, py::arg{"num_threads"} = -1)
        .def(
            "__call__",
            [](Polynomial<Hamiltonian>& self, bits512 const& spin)
                -> v2::QuantumState const& { return self(spin); },
            py::return_value_policy::reference_internal);
#else
    py::class_<Polynomial, std::shared_ptr<Polynomial>>(m, "Polynomial")
        .def(py::init([](std::shared_ptr<Heisenberg const> hamiltonian,
                         std::vector<complex_type> roots, bool normalising) {
                 auto const max_states = static_cast<uint64_t>(std::ceil(
                     std::pow(hamiltonian->max_states() + 1, roots.size())));
                 return std::make_shared<Polynomial>(
                     [f_ptr = std::move(hamiltonian)](auto&&... args) {
                         return (*f_ptr)(std::forward<decltype(args)>(args)...);
                     },
                     std::move(roots), normalising, max_states);
             }),
             py::arg{"hamiltonian"}, py::arg{"roots"},
             py::arg{"normalising"} = false);
    // .def(
    //     "__call__",
    //     [](Polynomial<Hamiltonian>& self, bits512 const& spin)
    //         -> v2::QuantumState const& { return self(spin); },
    //     py::return_value_policy::reference_internal);

#endif
}

TCM_EXPORT auto bind_polynomial(PyObject* _module) -> void
{
    auto m = py::module{py::reinterpret_borrow<py::object>(_module)};

    std::vector<std::string> keep_alive;
#define DOC(str) trim(keep_alive, str)

#if 0
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
#else
    // py::class_<v2::QuantumState>(m, "ExplicitState", DOC(R"EOF(
    //         Quantum state |ψ⟩=∑cᵢ|σᵢ⟩ backed by a Dict[int, complex].
    //     )EOF"));
    // .def(
    //     "__contains__",
    //     [](v2::QuantumState const& self, bits512 const& spin) {
    //         auto const& table = self.unsafe_locked_table();
    //         return table.count(spin) > 0;
    //     },
    //     py::arg{"spin"})
    // .def(
    //     "__getitem__",
    //     [](v2::QuantumState const& self, bits512 const& spin) {
    //         auto const& table = self.unsafe_locked_table();
    //         auto        i     = table.find(spin);
    //         if (i != table.cend()) { return i->second; }
    //         throw py::key_error{};
    //     },
    //     py::arg{"spin"})
    // .def(
    //     "__len__",
    //     [](v2::QuantumState const& self) {
    //         return self.unsafe_locked_table().size();
    //     },
    //     R"EOF(Returns number of elements in |ψ⟩.)EOF")
    // .def(
    //     "__iter__",
    //     [](v2::QuantumState const& self) {
    //         auto const& table = self.unsafe_locked_table();
    //         return py::make_iterator(table.cbegin(), table.cend());
    //     },
    //     py::keep_alive<0, 1>());
#endif

    make_polynomial(m, "Polynomial");

#undef DOC
}

TCM_NAMESPACE_END
