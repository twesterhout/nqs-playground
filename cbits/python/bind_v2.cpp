#include "bind_v2.hpp"
#include "pybind11_helpers.hpp"

namespace py = pybind11;

TCM_NAMESPACE_BEGIN

auto bind_v2(PyObject* _module) -> void
{
    auto m = py::module{py::reinterpret_borrow<py::object>(_module)};

    std::vector<std::string> keep_alive;
#define DOC(str) trim(keep_alive, str)

    m.def("is_operator_real",
          [](ls_operator const& op) { return ls_operator_is_real(&op); });

#undef DOC
}

TCM_NAMESPACE_END
