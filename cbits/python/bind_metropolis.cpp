#include "bind_metropolis.hpp"
#include "../metropolis.hpp"
#include "../spin_basis.hpp"
#include "../trim.hpp"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

TCM_NAMESPACE_BEGIN

auto bind_metropolis(PyObject* _module) -> void
{
    namespace py = pybind11;
    auto m       = py::module{py::reinterpret_borrow<py::object>(_module)};

    std::vector<std::string> keep_alive;
#define DOC(str) trim(keep_alive, str)

    py::class_<MetropolisKernel>(m, "MetropolisKernel")
        .def(py::init<std::shared_ptr<SpinBasis const>>(), DOC(R"EOF(
            Constructs Metropolis transition kernel.

            :param basis: specifies the Hilbert space basis.)EOF"))
        .def_property_readonly("basis", &MetropolisKernel::basis,
                               DOC(R"EOF(Returns the Hilbert space basis.)EOF"))
        .def(
            "__call__",
            [](MetropolisKernel const& self, torch::Tensor x) {
                return self(x);
            },
            py::arg{"x"}.noconvert());

#undef DOC
}

TCM_NAMESPACE_END
