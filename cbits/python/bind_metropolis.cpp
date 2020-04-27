#include "bind_metropolis.hpp"
#include "../metropolis.hpp"
#include "../spin_basis.hpp"
#include "../trim.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

TCM_NAMESPACE_BEGIN

auto bind_metropolis(PyObject* _module) -> void
{
    namespace py = pybind11;
    auto m       = py::module{py::reinterpret_borrow<py::object>(_module)};

    std::vector<std::string> keep_alive;
#define DOC(str) trim(keep_alive, str)

    py::class_<MetropolisKernel>(m, "MetropolisKernel")
        .def(py::init<std::shared_ptr<BasisBase const>>(), DOC(R"EOF(
            Constructs Metropolis transition kernel.

            :param basis: specifies the Hilbert space basis.)EOF"))
        .def_property_readonly("basis", &MetropolisKernel::basis,
                               DOC(R"EOF(Returns the Hilbert space basis.)EOF"))
        .def(
            "__call__",
            [](MetropolisKernel const& self, torch::Tensor x) {
                return self(std::move(x));
            },
            py::arg{"x"}.noconvert());

    py::class_<ProposalGenerator>(m, "_ProposalGenerator")
        .def(py::init<std::shared_ptr<BasisBase const>>(), DOC(R"EOF(
            :param basis: specifies the Hilbert space basis.)EOF"))
        .def_property_readonly("basis", &ProposalGenerator::basis,
                               DOC(R"EOF(Returns the Hilbert space basis.)EOF"))
        .def(
            "__call__",
            [](ProposalGenerator const& self, torch::Tensor x) {
                return self(std::move(x));
            },
            py::arg{"x"}.noconvert());

    m.def("_pick_next_state_index", &_pick_next_state_index,
          py::arg{"jump_rates"}.noconvert(),
          py::arg{"counts"}.noconvert(),
          py::arg{"out"}.noconvert() = py::none());
    m.def("_calculate_jump_rates",
          [](torch::Tensor current, torch::Tensor possible,
             std::vector<int64_t> const& counts, py::object device) {
              return _calculate_jump_rates(std::move(current), std::move(possible),
                  counts, torch::python::detail::py_object_to_device(device));
          },
          py::arg{"current_log_prob"},
          py::arg{"possible_log_prob"},
          py::arg{"counts"},
          py::arg{"target_device"});
    m.def("_add_waiting_time_", &_add_waiting_time_, py::arg{"t"}.noconvert(), py::arg{"rates"}.noconvert());
    m.def("_store_ready_samples_", &_store_ready_samples_);

#undef DOC
}

TCM_NAMESPACE_END
