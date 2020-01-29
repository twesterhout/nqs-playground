#include "bind_cuda.hpp"
#include "../trim.hpp"

#include <fmt/format.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

TCM_NAMESPACE_BEGIN

auto bind_cuda(PyObject* _module) -> void
{
    namespace py = pybind11;
    auto m       = py::module{py::reinterpret_borrow<py::object>(_module)};

}

TCM_NAMESPACE_END
