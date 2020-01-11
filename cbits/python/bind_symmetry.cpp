#include "bind_symmetry.hpp"
#include "../symmetry.hpp"
#include "../trim.hpp"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

TCM_NAMESPACE_BEGIN

auto bind_symmetry(PyObject* _module) -> void
{
    namespace py = pybind11;
    auto m       = py::module{py::reinterpret_borrow<py::object>(_module)};

    std::vector<std::string> keep_alive;
#define DOC(str) trim(keep_alive, str)

    py::class_<Symmetry>(m, "Symmetry", DOC(R"EOF(
        A symmetry operator with optimised ``__call__`` member function. It
        stores three things: a Benes network to permute bits, periodicity of
        the permutation, and the symmetry sector.
        )EOF"))
        .def(py::init([](std::pair<std::array<uint64_t, 6>,
                                   std::array<uint64_t, 6>> const& network,
                         unsigned const sector, unsigned const periodicity) {
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
        .def_property_readonly("sector", &Symmetry::sector, DOC(R"EOF(
             Returns the symmetry sector to which we have restricted the Hilbert
             space. It is an integer between 0 and :attr:`periodicity`.)EOF"))
        .def_property_readonly("periodicity", &Symmetry::periodicity, DOC(R"EOF(
             Returns the periodicity of the underlying permutation operator
             ``P``. I.e. a number ``n`` such that ``P^n == id``.)EOF"))
        .def_property_readonly("eigenvalue", &Symmetry::eigenvalue, DOC(R"EOF(
             Returns the eigenvalue of the operator in sector :attr:`sector`.)EOF"))
        .def("__call__", &Symmetry::operator(), DOC(R"EOF(
             Shuffles bits of ``x`` according to the underlying permutation.)EOF"),
             py::arg{"x"})
        .def(py::pickle(
            [](Symmetry const& self) { return self._state_as_tuple(); },
            [](decltype(std::declval<Symmetry const&>()._state_as_tuple())
                   const& state) {
                return Symmetry{{std::get<0>(state), std::get<1>(state)},
                                std::get<2>(state),
                                std::get<3>(state)};
            }));
#undef DOC
}

TCM_NAMESPACE_END
