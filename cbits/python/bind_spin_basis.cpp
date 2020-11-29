#include "bind_spin_basis.hpp"
#include "../spin_basis.hpp"
#include "../tensor_info.hpp"
#include "../trim.hpp"
#include "pybind11_helpers.hpp"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

TCM_NAMESPACE_BEGIN

template <class T, int Flags>
decltype(auto) to_const(py::array_t<T, Flags>& array)
{
    // NOTE: Yes, it's not nice to use pybind11::detail, but currently,
    // there's no other way...
    py::detail::array_proxy(array.ptr())->flags &=
        ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
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

    py::class_<BasisBase, std::shared_ptr<BasisBase>>(m, "BasisBase")
        .def_property_readonly("number_spins", &BasisBase::number_spins,
                               DOC(R"EOF(
            Number of particles in the system.)EOF"))
        .def_property_readonly("hamming_weight", &BasisBase::hamming_weight,
                               DOC(R"EOF(
            Hamming weight of basis states. ``None`` if it is not fixed.)EOF"))
        .def(
            "full_info",
            [](BasisBase const& self, bits512 const& spin) {
                return self.full_info(spin);
            },
            DOC(R"EOF(
            All available information about a basis state. Returns a tuple of

            1) representative state;
            2) character of the group element maps ``x`` to its representative;
            3) normalisation of the basis state (i.e. length of the
               corresponding orbit).)EOF"),
            py::arg{"spin"})
        .def(
            "representative",
            [](BasisBase const& self, torch::Tensor spins) {
                TCM_CHECK(spins.device().type() == c10::DeviceType::CPU,
                          std::invalid_argument,
                          "spins tensor must reside on the CPU");
                auto spins_info = obtain_tensor_info<bits512 const>(spins);
                auto out =
                    torch::empty({spins_info.size(), 8L},
                                 torch::TensorOptions{}.dtype(torch::kInt64));
                auto out_info = obtain_tensor_info<bits512>(out);
                for (auto i = 0L; i < spins_info.size(); ++i) {
                    out_info.data[i * out_info.stride()] =
                        std::get<0>(self.full_info(spins_info[i]));
                }
                return out;
            },
            py::arg{"spins"})
        .def(
            "norm",
            [](BasisBase const& self, bits512 const& spin) {
                return std::get<2>(self.full_info(spin));
            },
            DOC(R"EOF(
            Returns the normalisation of the basis state (i.e. length of the
            corresponding orbit). This is equivalent to
            ``self.full_info(x)[2]``.)EOF"),
            py::arg{"spin"})
        .def(
            "norm",
            [](BasisBase const& self, torch::Tensor spins) {
                TCM_CHECK(spins.device().type() == c10::DeviceType::CPU,
                          std::invalid_argument,
                          "spins tensor must reside on the CPU");
                auto spins_info = obtain_tensor_info<bits512 const>(spins);
                auto out =
                    torch::empty({spins_info.size()},
                                 torch::TensorOptions{}.dtype(torch::kFloat32));
                auto out_info = obtain_tensor_info<float>(out);
                for (auto i = 0L; i < spins_info.size(); ++i) {
                    out_info.data[i * out_info.stride()] = static_cast<float>(
                        std::get<2>(self.full_info(spins_info[i])));
                }
                return out;
            },
            DOC(R"EOF(
            Computes normalisation for a whole array of basis states. This is
            equivalent to calling ``self.norm`` for each element of ``spins``
            but is faster.

            :param spins: N x 8 tensor of int64. Each row must be contiguous and
                is treated as a single spin configuration.)EOF"),
            py::arg{"spins"});

    py::class_<SmallSpinBasis, BasisBase, std::shared_ptr<SmallSpinBasis>>(
        m, "SmallSpinBasis")
        .def(py::init<std::vector<v2::Symmetry<64>>, unsigned,
                      std::optional<unsigned>>(),
             DOC(R"EOF(
             Initialises Hilbert space basis for a system of <=64 spins.

             .. warning:: Do not use this function directly! Use
                          ``nqs_playground.SpinBasis`` instead.)EOF"),
             py::arg{"symmetries"}, py::arg{"number_spins"},
             py::arg{"hamming_weight"} = py::none())
        .def("build", &SmallSpinBasis::build, DOC(R"EOF(
             Constructs a list of representative vectors. After this operation,
             methods such as ``index``, ``number_states``, and ``states`` can be
             used.

             .. note:: This operation can potentially take a long time to
                       complete and use a lot of memory. Only use it for small
                       systems.)EOF"))
        .def_property_readonly(
            "states",
            [](py::object self) {
                return span_to_numpy(
                    self.cast<SmallSpinBasis const&>().states(), self);
            },
            DOC(R"EOF(
            Representative states as a 1D NumPy array of uint64. This property
            is available only after a call to ``build()``.)EOF"))
        .def_property_readonly("number_states", &SmallSpinBasis::number_states,
                               DOC(R"EOF(
            Number of states in the basis. This property is available only
            after a call to ``build()``.)EOF"))
        .def("index", &SmallSpinBasis::index, DOC(R"EOF(
             Given a representative state ``x``, returns its index in the array
             of representatives (i.e. ``self.states``).)EOF"),
             py::arg{"x"}.noconvert())
        .def(
            "index",
            [](SmallSpinBasis const& self, torch::Tensor spins) {
                auto spins_info = obtain_tensor_info<uint64_t const>(spins);
                auto out =
                    torch::empty({spins_info.size()},
                                 torch::TensorOptions{}.dtype(torch::kInt64));
                auto out_info = obtain_tensor_info<uint64_t>(out);
                for (auto i = 0L; i < spins_info.size(); ++i) {
                    out_info[i] = self.index(spins_info[i]);
                }
                return out;
            },
            py::arg{"xs"}.noconvert())
        .def(py::pickle(
            [](SmallSpinBasis const& self) { return self._internal_state(); },
            &SmallSpinBasis::_from_internal_state));

    py::class_<BigSpinBasis, BasisBase, std::shared_ptr<BigSpinBasis>>(
        m, "BigSpinBasis")
        .def(py::init<std::vector<v2::Symmetry<512>>, unsigned,
                      std::optional<unsigned>>(),
             DOC(R"EOF(
             Initialises Hilbert space basis for a system of >64 spins.

             .. warning:: Do not use this function directly! Use
                          ``nqs_playground.SpinBasis`` instead.)EOF"),
             py::arg{"symmetries"}, py::arg{"number_spins"},
             py::arg{"hamming_weight"} = py::none());

#undef DOC
}

TCM_NAMESPACE_END
