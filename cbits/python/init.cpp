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

PYBIND11_MODULE(_C, m)
{
    using namespace TCM_NAMESPACE;

    m.doc() = R"EOF()EOF";

    m.def("log_apply", [](torch::Tensor spins, py::object py_op, ForwardT psi,
                          uint64_t batch_size) {
        auto operator_type = py::module_::import("lattice_symmetries").attr("Operator");
        TCM_CHECK(isinstance(py_op, operator_type), std::invalid_argument,
                  "'operator' argument has wrong type; expected lattice_symmetries.Operator");
        auto const* _op =
            reinterpret_cast<ls_operator*>(py_op.attr("_payload").attr("value").cast<intptr_t>());
        TCM_ASSERT(_op != nullptr, "");
        auto [op, max_required_size] = view_as_operator(*_op);
        py::gil_scoped_release release;
        return apply(std::move(spins), std::move(op), std::move(psi), max_required_size,
                     batch_size);
    }, py::arg{"spins"}.noconvert(), py::arg{"operator"}.noconvert(), py::arg{"psi"}, py::arg{"batch_size"});

    m.def(
        "manual_seed", [](uint64_t const seed) { manual_seed(seed); },
        R"EOF(Seed random number generators used by nqs_playground.)EOF", py::arg{"seed"});

    m.def(
        "random_spin", [](ls_spin_basis const& basis) { return random_spin(basis); },
        py::arg{"basis"}.noconvert());

    py::class_<MetropolisGenerator>(m, "MetropolisGenerator")
        .def(py::init<ls_spin_basis const&>(), py::arg{"basis"}.noconvert(), R"EOF(
            :param basis: specifies the Hilbert space basis.)EOF")
        .def(
            "__call__",
            [](MetropolisGenerator const& self, torch::Tensor x, py::object dtype) {
                return self(std::move(x), torch::python::detail::py_object_to_dtype(dtype));
            },
            py::arg{"x"}.noconvert(), py::arg{"dtype"}.noconvert());

    py::class_<ZanellaGenerator>(m, "ZanellaGenerator")
        .def(py::init<ls_spin_basis const&, std::vector<std::pair<unsigned, unsigned>>>(),
             py::arg{"basis"}.noconvert(), py::arg{"edges"}, R"EOF(
            :param basis: specifies the Hilbert space basis.)EOF")
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
        R"EOF(
        )EOF",
        py::arg{"weights"}.noconvert(), py::arg{"number_samples"}, py::arg{"time_step"}.noconvert(),
        py::arg{"device"}.noconvert());
}
