#include "bind_operator.hpp"
#include "../operator.hpp"
#include "../trim.hpp"

#include <fmt/format.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

TCM_NAMESPACE_BEGIN

namespace {
template <class T> auto make_call_function()
{
    namespace py = pybind11;
    return [](Operator const& self, py::array_t<T, py::array::c_style> x,
              std::optional<py::array_t<T, py::array::c_style>> y) {
        TCM_CHECK(
            x.ndim() == 1, std::domain_error,
            fmt::format("x has invalid shape: [{}]; expected a vector",
                        fmt::join(x.shape(), x.shape() + x.ndim(), ", ")));
        auto src =
            gsl::span<T const>{x.data(), static_cast<size_t>(x.shape()[0])};
        auto out = [&y, n = src.size()]() {
            if (y.has_value()) {
                TCM_CHECK(
                    y->ndim() == 1, std::domain_error,
                    fmt::format(
                        "y has invalid shape: [{}]; expected a vector",
                        fmt::join(y->shape(), y->shape() + y->ndim(), ", ")));
                return std::move(*y);
            }
            return py::array_t<T, py::array::c_style>{n};
        }();
        auto dst = gsl::span<T>{out.mutable_data(),
                                static_cast<size_t>(out.shape()[0])};
        self(src, dst);
        return out;
    };
}
} // namespace

auto bind_operator(PyObject* _module) -> void
{
    namespace py = pybind11;
    auto m       = py::module{py::reinterpret_borrow<py::object>(_module)};

    py::options options;
    options.disable_function_signatures();

    std::vector<std::string> keep_alive;
#define DOC(str) trim(keep_alive, str)

    py::class_<Interaction>(m, "Interaction")
        .def(py::init<std::array<std::array<complex_type, 4>, 4>,
                      std::vector<Interaction::edge_type>>())
        .def_property_readonly("is_real", &Interaction::is_real)
        .def_property_readonly("max_index", &Interaction::max_index)
        .def_property_readonly(
            "matrix",
            [](Interaction const& self) {
                return py::array_t<complex_type, py::array::f_style>{
                    /*shape=*/{4, 4}, /*data=*/self.matrix()};
            })
        .def_property_readonly("edges",
                               [](Interaction const& self) {
                                   using std::begin, std::end;
                                   auto const edges = self.edges();
                                   return std::vector<Interaction::edge_type>{
                                       begin(edges), end(edges)};
                               })
        .def("diag", [](Interaction const& self, bits512 const& spin) {
            return self.diag(spin);
        });

#if 1
    py::class_<Operator, std::shared_ptr<Operator>>(m, "Operator")
        .def(py::init<std::vector<Interaction>,
                      std::shared_ptr<BasisBase const>>(),
             DOC(R"EOF(
            Constructs the Heisenberg Hamiltonian with given exchange couplings.
            Full Hamiltonian is: ``∑Jᵢⱼ·σᵢ⊗σⱼ``, where ``σ`` are vector spin
            operators.

            :param specs: is a list of couplings. Each element is a tuple
                ``(Jᵢⱼ, i, j)``. This defines a term ``Jᵢⱼ·σᵢ⊗σⱼ``.
            :param basis: specifies the Hilbert space basis on which the
                Hamiltonian is defined.)EOF"))
        .def_property_readonly(
            "interactions",
            [](Operator const& self) {
                using std::begin, std::end;
                auto is = self.interactions();
                return std::vector<Interaction>{begin(is), end(is)};
            },
            DOC(R"EOF(Returns the list of 'Interaction's)EOF"))
        .def_property_readonly("basis", &Operator::basis, DOC(R"EOF(
            Returns the Hilbert space basis on which the Hamiltonian is defined.)EOF"))
        .def_property_readonly("is_real", &Operator::is_real, DOC(R"EOF(
            Returns whether the Hamiltonian is real, i.e. whether all couplings
            ``{cᵢⱼ}`` are real.)EOF"))
        .def("__call__", make_call_function<float>(), py::arg{"x"}.noconvert(),
             py::arg{"y"}.noconvert() = py::none())
        .def("__call__", make_call_function<double>(), py::arg{"x"}.noconvert(),
             py::arg{"y"}.noconvert() = py::none())
        .def("__call__", make_call_function<std::complex<float>>(),
             py::arg{"x"}.noconvert(), py::arg{"y"}.noconvert() = py::none())
        .def("__call__", make_call_function<std::complex<double>>(),
             py::arg{"x"}.noconvert(), py::arg{"y"}.noconvert() = py::none())
        .def("to_csr", [](Operator const& self) {
            auto [values, indices] = self._to_sparse<double>();
            auto const n           = values.size(0);
            auto const csr_matrix =
                py::module::import("scipy.sparse").attr("csr_matrix");
            auto data = py::cast(values).attr("numpy")();
            auto row_indices =
                py::cast(torch::narrow(indices, /*dim=*/1, /*start=*/0,
                                       /*length=*/1))
                    .attr("squeeze")()
                    .attr("numpy")();
            auto col_indices =
                py::cast(torch::narrow(indices, /*dim=*/1, /*start=*/1,
                                       /*length=*/1))
                    .attr("squeeze")()
                    .attr("numpy")();
            auto const* basis =
                static_cast<SmallSpinBasis const*>(self.basis().get());
            return csr_matrix(
                std::make_tuple(data,
                                std::make_tuple(row_indices, col_indices)),
                std::make_tuple(basis->number_states(),
                                basis->number_states()));
        });
#endif

#undef DOC
}

TCM_NAMESPACE_END
