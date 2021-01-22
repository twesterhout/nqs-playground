// Copyright (c) 2019-2021, Tom Westerhout
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "../common/accumulator.hpp"
#include "../common/metropolis.hpp"
#include "../common/wrappers.hpp"
#include "../common/zanella.hpp"
#include "pybind11_helpers.hpp"

namespace py = pybind11;

// #if defined(TCM_CLANG)
// #    pragma clang diagnostic push
// #    pragma clang diagnostic ignored "-Wmissing-prototypes"
// #endif
PYBIND11_MODULE(_C, m)
{
    using namespace TCM_NAMESPACE;
    // #if defined(TCM_CLANG)
    // #    pragma clang diagnostic pop
    // #endif

    m.doc() = R"EOF()EOF";

    std::vector<std::string> keep_alive;
#define DOC(str) trim(keep_alive, str)

    m.def("is_operator_real", [](ls_operator const& op) { return ls_operator_is_real(&op); });

    m.def(
        "log_apply",
        [](torch::Tensor spins, ls_operator const& _op, ForwardT psi, uint64_t batch_size) {
            auto [op, max_required_size] = view_as_operator(_op);
            return apply(std::move(spins), std::move(op), std::move(psi), max_required_size,
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

    py::class_<MetropolisGenerator>(m, "MetropolisGenerator")
        .def(py::init<ls_spin_basis const&>(), py::arg{"basis"}.noconvert(), DOC(R"EOF(
            :param basis: specifies the Hilbert space basis.)EOF"))
        .def(
            "__call__",
            [](MetropolisGenerator const& self, torch::Tensor x, py::object dtype) {
                return self(std::move(x), torch::python::detail::py_object_to_dtype(dtype));
            },
            py::arg{"x"}.noconvert(), py::arg{"dtype"}.noconvert());

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
