#include "bind_heisenberg.hpp"
#include "../heisenberg.hpp"
#include "../trim.hpp"

#include <fmt/format.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

TCM_NAMESPACE_BEGIN

template <class T> auto make_call_function()
{
    namespace py = pybind11;
    return [](Heisenberg const& self, py::array_t<T, py::array::c_style> x,
              std::optional<py::array_t<T, py::array::c_style>> y) {
        TCM_CHECK(
            x.ndim() == 1, std::domain_error,
            fmt::format("x has invalid shape: [{}]; expected a vector",
                        fmt::join(x.shape(), x.shape() + x.ndim(), ", ")));
        auto src =
            gsl::span<T const>{x.data(), static_cast<size_t>(x.shape()[0])};
        auto out = y.value_or(py::array_t<T, py::array::c_style>{src.size()});
        if (y.has_value()) {
            TCM_CHECK(y->ndim() == 1, std::domain_error,
                      fmt::format(
                          "y has invalid shape: [{}]; expected a vector",
                          fmt::join(y->shape(), y->shape() + y->ndim(), ", ")));
        }
        auto dst = gsl::span<T>{out.mutable_data(),
                                static_cast<size_t>(out.shape()[0])};
        self(src, dst);
        return out;
    };
}

auto bind_heisenberg(PyObject* _module) -> void
{
    namespace py = pybind11;
    auto m       = py::module{py::reinterpret_borrow<py::object>(_module)};

    py::options options;
    options.disable_function_signatures();

    std::vector<std::string> keep_alive;
#define DOC(str) trim(keep_alive, str)

    py::class_<Heisenberg, std::shared_ptr<Heisenberg>>(m, "Heisenberg")
        .def(
            py::init<Heisenberg::spec_type, std::shared_ptr<SpinBasis const>>(),
            DOC(R"EOF(
            Constructs the Heisenberg Hamiltonian.

            :param specs: is a list of couplings. Each element is a tuple
                ``(cᵢⱼ, i, j)``. This defines a term ``cᵢⱼ·σᵢ⊗σⱼ``.
            :param basis: specifies the Hilbert space basis on which the
                Hamiltonian is defined.)EOF"))
        .def_property_readonly(
            "edges",
            [](Heisenberg const& self) {
                auto edges = self.edges();
                return std::vector<Heisenberg::edge_type>{edges.begin(),
                                                          edges.end()};
            },
            DOC(R"EOF(Returns the list of couplings)EOF"))
        .def_property_readonly("basis", &Heisenberg::basis, DOC(R"EOF(
            Returns the Hilbert space basis on which the Hamiltonian is defined.)EOF"))
        .def_property_readonly("is_real", &Heisenberg::is_real, DOC(R"EOF(
            Returns whether the Hamiltonian is real, i.e. whether all couplings
            ``{cᵢⱼ}`` are real.)EOF"))
        .def("__call__", make_call_function<float>(), py::arg{"x"}.noconvert(),
             py::arg{"y"}.noconvert() = py::none())
        .def("__call__", make_call_function<double>(), py::arg{"x"}.noconvert(),
             py::arg{"y"}.noconvert() = py::none())
        .def("__call__", make_call_function<std::complex<float>>(),
             py::arg{"x"}.noconvert(), py::arg{"y"}.noconvert() = py::none())
        .def("__call__", make_call_function<std::complex<double>>(),
             py::arg{"x"}.noconvert(), py::arg{"y"}.noconvert() = py::none());

#undef DOC
}

TCM_NAMESPACE_END
