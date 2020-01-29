#include "bind_jacobian.hpp"
#include "../jacobian.hpp"
#include "../trim.hpp"

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/script.h>

TCM_NAMESPACE_BEGIN

auto bind_jacobian(PyObject* _module) -> void
{
    namespace py = pybind11;
    auto m       = py::module{py::reinterpret_borrow<py::object>(_module)};

    std::vector<std::string> keep_alive;
#define DOC(str) trim(keep_alive, str)

    m.def("_jacobian", &jacobian, DOC(R"EOF(
          Computes the jacobian of ``module(inputs)`` with respect to module
          parameters and stores the result in ``out``.

          :param module: Jacobian of this module is be computed.
          :param inputs: Points at which the Jacobian is evaluated.
          :param out: Output tensor where the Jacobian is stored.)EOF"),
          py::arg{"module"}.noconvert(), py::arg{"inputs"}.noconvert(),
          py::arg{"out"}.noconvert() = py::none(),
          // NOTE: releasing GIL here is important, see
          //     https://github.com/pytorch/pytorch/issues/32045
          py::call_guard<py::gil_scoped_release>());

#undef DOC
}

TCM_NAMESPACE_END
