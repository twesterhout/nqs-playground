#include "bind_spin_basis.hpp"
#include "../spin_basis.hpp"
#include "../trim.hpp"

#include <fmt/format.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

TCM_NAMESPACE_BEGIN

template <class T, int Flags>
decltype(auto) to_const(py::array_t<T, Flags>& array)
{
    // NOTE: Yes, it's not nice to use pybind11::detail, but currently,
    // there's no other way...
    pybind11::detail::array_proxy(array.ptr())->flags &=
        ~pybind11::detail::npy_api::NPY_ARRAY_WRITEABLE_;
    return array;
}

template <class T>
auto span_to_numpy(gsl::span<T> xs, py::object base) -> py::array
{
    using type = std::remove_const_t<T>;
    auto a     = py::array_t<type>{xs.size(), xs.data(), std::move(base)};
    if (std::is_const<T>::value) { to_const(a); }
    return static_cast<py::array>(std::move(a));
}

auto bind_spin_basis(PyObject* _module) -> void
{
    auto m = py::module{py::reinterpret_borrow<py::object>(_module)};

    std::vector<std::string> keep_alive;
#define DOC(str) trim(keep_alive, str)

    py::class_<SpinBasis, std::shared_ptr<SpinBasis>>(m, "SpinBasis")
        .def(py::init<std::vector<Symmetry>, unsigned,
                      std::optional<unsigned>>(),
             DOC(R"EOF(
             Initialises the basis without constructing the internal cache.
             Memory usage thus does not increase with system size (only with
             number of symmetries, but this is negligible).

             :param symmetries: a list of symmetries. It must be either empty or
                form a group. This is unchecked, but bad stuff will happen is
                this requirement is not fulfilled.
             :param number_spins: number of particles in the system.
             :param hamming_weight: Hamming weight of states, i.e. number of
                 spin ups. Specifying it is equivalent to restricting the
                 Hilbert space to some magnetisation sector.
             )EOF"),
             py::arg{"symmetries"}, py::arg{"number_spins"},
             py::arg{"hamming_weight"})
        .def("build", &SpinBasis::build, DOC(R"EOF(
             Constructs a list of representative vectors. After this operation,
             methods such as ``index``, ``number_states``, and ``states`` can be
             used.
             )EOF"))
        .def_property_readonly(
            "states",
            [](py::object self) {
                return span_to_numpy(self.cast<SpinBasis const&>().states(),
                                     self);
            },
            DOC(R"EOF(
            Array of representative states. This property is available only
            after a call to ``build()``.
            )EOF"))
        .def_property_readonly("number_spins", &SpinBasis::number_spins,
                               DOC(R"EOF(
            Number of particles in the system.
            )EOF"))
        .def_property_readonly("number_states", &SpinBasis::number_states,
                               DOC(R"EOF(
            Number of states in the basis. This property is available only
            after a call to ``build()``.
            )EOF"))
        .def_property_readonly("hamming_weight", &SpinBasis::hamming_weight,
                               DOC(R"EOF(
            Hamming weight of basis states. Returns ``None`` if it is not fixed.
            )EOF"))
        .def("index", &SpinBasis::index, DOC(R"EOF(
             Given a representative state ``x``, returns its index in the array
             of representatives (i.e. ``self.states``).
             )EOF"),
             py::arg{"x"})
#if 0
        .def("normalisation", &SpinBasis::normalisation, DOC(R"EOF(
             Given a representative state ``x``, returns its "norm".
             )EOF"),
             py::arg{"x"})
        .def("representative", &SpinBasis::representative, DOC(R"EOF(
             Given a state ``x``, returns its representative.
             )EOF"))
#endif
        .def("full_info", &SpinBasis::full_info, DOC(R"EOF(
             Given a state ``x``, returns a tuple ``(r, λ, N)`` where ``r`` is
             the representative of ``x``, ``N`` is the "norm" of ``r``, and λ is
             the eigenvalue of a group element ``g`` such that ``gr == x``.
             )EOF"))
#if 1
        .def(py::pickle(
            [](SpinBasis const& self) { return self._state_as_tuple(); },
            [](SpinBasis::PickleStateT const& state) {
                return SpinBasis::_from_tuple_state(state);
            }))
#endif
        .def("expand", [](SpinBasis const& self, torch::Tensor src,
                          torch::Tensor states) {
            return expand_states(self, src, states);
        });

#undef DOC
}

TCM_NAMESPACE_END
