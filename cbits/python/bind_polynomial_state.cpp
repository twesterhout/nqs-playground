#include "bind_polynomial_state.hpp"
#include "../polynomial_state.hpp"
#include "../trim.hpp"
#include "pybind11_helpers.hpp"

#include <torch/extension.h>
#include <torch/script.h>

TCM_NAMESPACE_BEGIN

// namespace {
// auto make_forward_function(torch::jit::script::Method method,
//                            std::optional<unsigned> number_spins) -> v2::ForwardT
// {
//     static_cast<void>(number_spins);
//     // auto const n =
//     //     number_spins.has_value() ? static_cast<int64_t>(*number_spins) : -1L;
//     return [f = std::move(method)](auto x) mutable {
//         // if (x.scalar_type == torch::kInt64) { x = unpack(x, n); }
//         return f({std::move(x)}).toTensor();
//     };
// }
//
// auto make_forward_function(pybind11::object method) -> v2::ForwardT {}
// } // namespace

auto bind_polynomial_state(PyObject* _module) -> void
{
    namespace py = pybind11;
    auto m       = py::module{py::reinterpret_borrow<py::object>(_module)};

    std::vector<std::string> keep_alive;
#define DOC(str) trim(keep_alive, str)

    m.def(
        "apply",
        [](torch::Tensor spins, Heisenberg const& hamiltonian,
           v2::ForwardT forward) {
            return apply(std::move(spins), hamiltonian, std::move(forward));
        },
        py::call_guard<py::gil_scoped_release>());

#if 0
    m.def("apply", [](torch::Tensor spins, Polynomial<Heisenberg>& polynomial,
                      v2::ForwardT forward) {
        return apply(std::move(spins), polynomial, std::move(forward));
    });
#endif

    m.def(
        "diag",
        [](torch::Tensor spins, Heisenberg const& hamiltonian) {
            return diag(std::move(spins), hamiltonian);
        },
        DOC(R"EOF(
        Computes diagonal elements ``⟨s|H|s⟩``.

        :param spins: a tensor of packed spin configurations for which to
            compute diagonal matrix elements.
        :param hamiltonian: operator which matrix elements to compute.)EOF"),
        py::arg{"spins"}, py::arg{"hamiltonian"});

#undef DOC
}

TCM_NAMESPACE_END
