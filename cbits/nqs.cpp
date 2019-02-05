#include "nqs.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_C_nqs, m)
{
    m.doc() = R"EOF()EOF";

    using ::TCM_NAMESPACE::Heisenberg;
    using ::TCM_NAMESPACE::Polynomial;
    using ::TCM_NAMESPACE::SpinVector;

    py::class_<SpinVector>(m, "CompactSpin")
        .def(py::init<py::array_t<float>>())
        .def(py::init<py::array_t<double>>())
        .def(py::init<py::array_t<int>>())
        .def(py::init<py::array_t<long>>())
        .def(
            "__copy__", [](SpinVector const& x) { return SpinVector{x}; },
            R"EOF(Copies the current spin configuration.)EOF")
        .def(
            "__deepcopy__", [](SpinVector const& x) { return SpinVector{x}; },
            R"EOF(Same as ``self.__copy__()``.)EOF")
        .def("__len__", &SpinVector::size, R"EOF(
            Returns the number of spins in the vector.
        )EOF")
        .def_property_readonly("size", &SpinVector::size,
                               R"EOF(Same as ``self.__len__()``.)EOF")
        .def("__int__",
             [](SpinVector const& x) { return static_cast<std::size_t>(x); })
        .def("__str__",
             [](SpinVector const& x) { return static_cast<std::string>(x); })
        .def("__getitem__",
             [](SpinVector const& x, unsigned const i) {
                 return x.at(i) == ::TCM_NAMESPACE::Spin::up ? 1.0f : -1.0f;
             })
        .def("__setitem__",
             [](SpinVector& x, unsigned const i, float const spin) {
                 auto const float2spin = [](auto const s) {
                     if (s == -1.0f) { return ::TCM_NAMESPACE::Spin::down; }
                     if (s == 1.0f) { return ::TCM_NAMESPACE::Spin::up; }
                     throw std::invalid_argument{
                         "Invalid spin: expected either -1 or +1, but got "
                         + std::to_string(s) + "."};
                 };
                 x.at(i) = float2spin(spin);
             })
        .def("numpy",
             [](SpinVector const&                      x,
                py::array_t<float, py::array::c_style> out) {
                 return x.numpy(out);
             })
        .def("numpy",
             [](SpinVector const&                       x,
                py::array_t<double, py::array::c_style> out) {
                 return x.numpy(out);
             })
        .def("__hash__", &SpinVector::hash, R"EOF(
            Returns the hash of the spin configuration.
        )EOF");

    py::class_<Heisenberg>(m, "Heisenberg")
        .def(py::init<std::vector<std::pair<unsigned, unsigned>>>())
        .def(
            py::init<std::vector<std::pair<unsigned, unsigned>>, long double>())
        .def("__len__", &Heisenberg::size)
        .def_property(
            "coupling", [](Heisenberg const& h) { return h.coupling(); },
            [](Heisenberg& h, long double coupling) { h.coupling(coupling); })
        .def("edges", &Heisenberg::edges);

    py::class_<Polynomial>(m, "Polynomial")
        .def(py::init<Heisenberg, std::vector<long double>>())
        .def_property_readonly("size", &Polynomial::size)
        .def("print", &Polynomial::print)
        .def("__call__", &Polynomial::operator())
        .def("keys",
             [](Polynomial const&                      p,
                py::array_t<float, py::array::c_style> out) {
                 return p.keys(out);
             })
        .def("values", [](Polynomial const& p, py::array_t<float> out) {
            return p.values(out);
        });
}
