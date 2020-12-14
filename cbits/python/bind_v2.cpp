#include "bind_v2.hpp"
#include "../v2/accumulator.hpp"
#include "../wrappers.hpp"
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

    m.def(
        "log_apply",
        [](torch::Tensor spins, ls_operator const& _op, v2::ForwardT psi,
           uint64_t batch_size) {
            auto [op, max_required_size] = view_as_operator(_op);
            return v2::apply(std::move(spins), std::move(op), std::move(psi),
                             max_required_size, batch_size);
        },
        py::call_guard<py::gil_scoped_release>());

#undef DOC
}

TCM_NAMESPACE_END
