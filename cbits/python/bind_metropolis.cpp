#include "bind_metropolis.hpp"
#include "../metropolis.hpp"
#include "../spin_basis.hpp"
#include "../tabu.hpp"
#include "../trim.hpp"
#include "pybind11_helpers.hpp"

#include "../cpu/kernels.hpp"

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

    // Currently, we only access the generator from one thread
    m.def(
        "manual_seed",
        [](uint64_t const seed) { global_random_generator().seed(seed); },
        DOC(R"EOF(Seed the random number generator used by nqs_playground.)EOF"),
        py::arg{"seed"});

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

    m.def("tabu_process", &tabu_process);
#if 0
    m.def("zanella_next_state_index", &zanella_next_state_index, DOC(R"EOF(

          )EOF"),
          py::arg{"jump_rates"}.noconvert(), py::arg{"counts"}.noconvert(),
          py::arg{"out"}.noconvert() = py::none());
#endif

    m.def(
        "zanella_next_state_index",
        [](torch::Tensor jump_rates, torch::Tensor jump_rates_sum,
           std::vector<int64_t> const& counts, py::object device) {
            return zanella_next_state_index(
                std::move(jump_rates), std::move(jump_rates_sum), counts,
                torch::python::detail::py_object_to_device(device));
        },
        py::arg{"jump_rates"}.noconvert(),
        py::arg{"jump_rates_sum"}.noconvert(), py::arg{"counts"},
        py::arg{"device"}.noconvert());

    m.def("zanella_jump_rates", &zanella_jump_rates, DOC(R"EOF(

        )EOF"),
          py::arg{"current_log_prob"}.noconvert(),
          py::arg{"possible_log_prob"}.noconvert(), py::arg{"counts"});

    m.def("zanella_waiting_time", &zanella_waiting_time, DOC(R"EOF(
          Calculate waiting time for current state.

          :param rates: Λs
          :param out: If specified, the result is stored into it.
          :return:)EOF"),
          py::arg{"rates"}.noconvert(), py::arg{"out"}.noconvert() = py::none(),
          py::call_guard<py::gil_scoped_release>());

    m.def(
        "zanella_choose_samples",
        [](torch::Tensor weights, int64_t number_samples, double time_step,
           py::object device) {
            return zanella_choose_samples(
                std::move(weights), number_samples, time_step,
                torch::python::detail::py_object_to_device(device));
        },
        DOC(R"EOF(
        )EOF"),
        py::arg{"weights"}.noconvert(), py::arg{"number_samples"},
        py::arg{"time_step"}.noconvert(), py::arg{"device"}.noconvert());

#undef DOC
}

TCM_NAMESPACE_END
