#include "bind_polynomial_state.hpp"
#include "../operators.hpp"
#include "../polynomial_state.hpp"
#include "../trim.hpp"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/script.h>

TCM_NAMESPACE_BEGIN

namespace {
auto make_forward_function(torch::jit::script::Method method,
                           std::optional<unsigned> number_spins) -> v2::ForwardT
{
    if (!number_spins.has_value()) {
        return [f = std::move(method)](auto x) mutable {
            return f({std::move(x)}).toTensor();
        };
    }
    return [f = std::move(method),
            n = static_cast<int64_t>(*number_spins)](auto x) mutable {
        return f({unpack(std::move(x), n)}).toTensor();
    };
}
} // namespace

auto bind_polynomial_state(PyObject* _module) -> void
{
    namespace py = pybind11;
    auto m       = py::module{py::reinterpret_borrow<py::object>(_module)};

    std::vector<std::string> keep_alive;
#define DOC(str) trim(keep_alive, str)

    m.def("apply", [](torch::Tensor spins, Heisenberg const& hamiltonian,
                      torch::jit::script::Method forward) {
        return apply(
            std::move(spins), hamiltonian,
            make_forward_function(std::move(forward),
                                  hamiltonian.basis()->number_spins()));
    });

#undef DOC
}

TCM_NAMESPACE_END
