#include "bind_v2.hpp"
#include "../trim.hpp"
#include "../v2/accumulator.hpp"
#include "../v2/zanella.hpp"
#include "../wrappers.hpp"
#include "pybind11_helpers.hpp"

namespace py = pybind11;

TCM_NAMESPACE_BEGIN

auto bind_v2(PyObject* _module) -> void
{
    auto m = py::module{py::reinterpret_borrow<py::object>(_module)};

    std::vector<std::string> keep_alive;
#define DOC(str) trim(keep_alive, str)

    m.def("is_operator_real", [](ls_operator const& op) { return ls_operator_is_real(&op); });

    m.def(
        "log_apply",
        [](torch::Tensor spins, ls_operator const& _op, v2::ForwardT psi, uint64_t batch_size) {
            auto [op, max_required_size] = view_as_operator(_op);
            return v2::apply(std::move(spins), std::move(op), std::move(psi), max_required_size,
                             batch_size);
        },
        py::call_guard<py::gil_scoped_release>());

    // Currently, we only access the generator from one thread
    m.def(
        "manual_seed", [](uint64_t const seed) { global_random_generator().seed(seed); },
        DOC(R"EOF(Seed the random number generator used by nqs_playground.)EOF"), py::arg{"seed"});

    m.def(
        "random_spin", [](ls_spin_basis const& basis) { return random_spin(basis); },
        py::arg{"basis"}.noconvert());

    py::class_<ZanellaGenerator>(m, "ZanellaGenerator")
        .def(py::init<ls_spin_basis const&>(), py::arg{"basis"}.noconvert(), DOC(R"EOF(
            :param basis: specifies the Hilbert space basis.)EOF"))
        .def(
            "__call__",
            [](ZanellaGenerator const& self, torch::Tensor x) { return self(std::move(x)); },
            py::arg{"x"}.noconvert());

    m.def(
        "zanella_choose_samples",
        [](torch::Tensor weights, int64_t const number_samples, double const time_step,
           py::object device) {
            return zanella_choose_samples(std::move(weights), number_samples, time_step,
                                          torch::python::detail::py_object_to_device(device));
        },
        DOC(R"EOF(
        )EOF"),
        py::arg{"weights"}.noconvert(), py::arg{"number_samples"}, py::arg{"time_step"}.noconvert(),
        py::arg{"device"}.noconvert());

#undef DOC
}

TCM_NAMESPACE_END
