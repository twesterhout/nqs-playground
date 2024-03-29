#include "zanella.hpp"
#include "accumulator.hpp"
#include <torch/extension.h>
#include <pybind11/functional.h>

namespace pybind11::detail {
template <> struct type_caster<ls_spin_basis> {
  private:
    ls_spin_basis* payload;

  public:
    auto load(handle src, bool /*convert*/) -> bool
    {
        if (src.is_none()) { return false; }
        auto basis_type = module_::import("lattice_symmetries").attr("SpinBasis");
        if (!isinstance(src, basis_type)) { return false; }
        payload =
            reinterpret_cast<ls_spin_basis*>(src.attr("_payload").attr("value").cast<intptr_t>());
        return true;
    }

    operator ls_spin_basis*() { return payload; }
    operator ls_spin_basis&() { return *payload; }

    static constexpr auto name            = _("lattice_symmetries.SpinBasis");
    template <class T> using cast_op_type = pybind11::detail::cast_op_type<T>;
};

template <> struct type_caster<ls_operator> {
  private:
    ls_operator* payload;

  public:
    auto load(handle src, bool /*convert*/) -> bool
    {
        if (src.is_none()) { return false; }
        auto operator_type = module_::import("lattice_symmetries").attr("Operator");
        if (!isinstance(src, operator_type)) { return false; }
        payload =
            reinterpret_cast<ls_operator*>(src.attr("_payload").attr("value").cast<intptr_t>());
        return true;
    }

    operator ls_operator*() { return payload; }
    operator ls_operator&() { return *payload; }

    static constexpr auto name            = _("lattice_symmetries.Operator");
    template <class T> using cast_op_type = pybind11::detail::cast_op_type<T>;
};
} // namespace pybind11::detail

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using namespace tcm;
    py::class_<ZanellaGenerator>(m, "ZanellaGenerator")
        .def(py::init<ls_spin_basis const&, std::vector<std::pair<unsigned, unsigned>>>(),
             py::arg{"basis"}.noconvert(), py::arg{"edges"}, R"EOF(
            :param basis: specifies the Hilbert space basis.)EOF")
        .def(
            "__call__",
            [](ZanellaGenerator const& self, torch::Tensor x) { return self(std::move(x)); },
            py::arg{"x"}.noconvert(), py::call_guard<py::gil_scoped_release>());

    m.def("log_apply", &log_apply, py::call_guard<py::gil_scoped_release>());
}
