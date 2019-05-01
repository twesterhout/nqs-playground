// Copyright (c) 2019, Tom Westerhout
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

#include "spin.hpp"
#include <pybind11/stl.h>
#include <torch/extension.h>

TCM_NAMESPACE_BEGIN

namespace detail {
auto spin_configuration_to_string(gsl::span<float const> spin) -> std::string
{
    std::ostringstream msg;
    msg << '[';
    if (spin.size() > 0) {
        msg << spin[0];
        for (auto i = size_t{1}; i < spin.size(); ++i) {
            msg << ", " << spin[i];
        }
    }
    msg << ']';
    return msg.str();
}

namespace {
    template <
        int ExtraFlags,
        class = std::enable_if_t<ExtraFlags & pybind11::array::c_style
                                 || ExtraFlags & pybind11::array::f_style> /**/>
    auto copy_to_numpy_array(SpinVector const&                     spin,
                             pybind11::array_t<float, ExtraFlags>& out) -> void
    {
        TCM_CHECK_DIM(out.ndim(), 1);
        TCM_CHECK_SHAPE(out.shape(0), spin.size());

        auto const spin2float = [](Spin const s) noexcept->float
        {
            return s == Spin::up ? 1.0f : -1.0f;
        };

        auto access = out.template mutable_unchecked<1>();
        for (auto i = 0u; i < spin.size(); ++i) {
            access(i) = spin2float(spin[i]);
        }
    }
} // unnamed namespace
} // namespace detail

auto SpinVector::numpy() const
    -> pybind11::array_t<float, pybind11::array::c_style>
{
    pybind11::array_t<float, pybind11::array::c_style> out{size()};
    detail::copy_to_numpy_array(*this, out);
    return out;
}

auto SpinVector::tensor() const -> torch::Tensor
{
    auto out = detail::make_tensor<float>(size());
    copy_to({out.data<float>(), size()});
    return out;
}

SpinVector::SpinVector(gsl::span<float const> spin)
{
    check_range(spin);
    copy_from(spin);
    TCM_ASSERT(is_valid(), "Bug! Post-condition violated");
}

SpinVector::SpinVector(torch::Tensor const& spins)
{
    TCM_CHECK_DIM(spins.dim(), 1);
    TCM_CHECK(spins.is_contiguous(), std::invalid_argument,
              "input tensor must be contiguous");
    TCM_CHECK_TYPE(spins.scalar_type(), torch::kFloat32);
    auto buffer = gsl::span<float const>{spins.data<float>(),
                                         static_cast<size_t>(spins.size(0))};
    check_range(buffer);
    copy_from(buffer);
    TCM_ASSERT(is_valid(), "Bug! Post-condition violated");
}

SpinVector::SpinVector(pybind11::str str)
{
    // PyUnicode_Check macro uses old style casts
#if defined(TCM_GCC)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
    // Borrowed from pybind11/pytypes.h
    pybind11::object temp = str;
    if (PyUnicode_Check(str.ptr())) {
        temp = pybind11::reinterpret_steal<pybind11::object>(
            PyUnicode_AsUTF8String(str.ptr()));
        if (!temp)
            throw std::runtime_error{
                "Unable to extract string contents! (encoding issue)"};
    }
    char*             buffer;
    pybind11::ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(temp.ptr(), &buffer, &length)) {
        throw std::runtime_error{
            "Unable to extract string contents! (invalid type)"};
    }
    TCM_CHECK(length <= static_cast<pybind11::ssize_t>(max_size()),
              std::overflow_error,
              fmt::format("spin chain too long {}; expected <={}", length,
                          max_size()));

    _data.as_ints = _mm_set1_epi32(0);
    _data.size    = static_cast<std::uint16_t>(length);
    for (auto i = 0u; i < _data.size; ++i) {
        auto const s = buffer[i];
        TCM_CHECK(s == '0' || s == '1', std::domain_error,
                  fmt::format("invalid spin '{}'; expected '0' or '1'", s));
        unsafe_at(i) = (s == '1') ? Spin::up : Spin::down;
    }
#if defined(TCM_GCC)
#    pragma GCC diagnostic pop
#endif
    TCM_ASSERT(is_valid(), "Bug! Post-condition violated");
}

#if 0
auto unpack_to_tensor(gsl::span<SpinVector const> src, torch::Tensor dst)
    -> void
{
    // unpack_to_tensor(std::begin(src), std::end(src), dst,
    //                  [](auto const& x) -> SpinVector const& { return x; });
    if (src.empty()) { return; }

    auto const size         = src.size();
    auto const number_spins = src[0].size();
    TCM_ASSERT(std::all_of(std::begin(src), std::end(src),
                           [number_spins](auto const& x) {
                               return x.size() == number_spins;
                           }),
               "Input range contains variable size spin chains");
    TCM_ASSERT(dst.dim() == 2, fmt::format("Invalid dimension {}", dst.dim()));
    TCM_ASSERT(size == static_cast<size_t>(dst.size(0)),
               fmt::format("Sizes don't match: size={}, dst.size(0)={}", size,
                           dst.size(0)));
    TCM_ASSERT(static_cast<int64_t>(number_spins) == dst.size(1),
               fmt::format("Sizes don't match: number_spins={}, dst.size(1)={}",
                           number_spins, dst.size(1)));
    TCM_ASSERT(dst.is_contiguous(), "Output tensor must be contiguous");

    auto const chunks_16     = number_spins / 16;
    auto const rest_16       = number_spins % 16;
    auto const rest_8        = number_spins % 8;
    auto const copy_cheating = [chunks = chunks_16 + (rest_16 != 0)](
                                   SpinVector const& spin, float* out) {
        for (auto i = 0u; i < chunks; ++i, out += 16) {
            detail::unpack(spin._data.spin[i], out);
        }
    };

    auto* data = dst.data<float>();
    for (auto i = size_t{0}; i < size - 1; ++i, data += number_spins) {
        copy_cheating(src[i], data);
    }
    src[size - 1].copy_to({data, number_spins});
}
#endif

auto all_spins(unsigned n, optional<int> magnetisation)
    -> std::vector<SpinVector,
                   boost::alignment::aligned_allocator<SpinVector, 64>>
{
    using VectorT =
        std::vector<SpinVector,
                    boost::alignment::aligned_allocator<SpinVector, 64>>;
    if (magnetisation.has_value()) {
        TCM_CHECK(n <= SpinVector::max_size(), std::overflow_error,
                  fmt::format("invalid n: {}; expected <={}", n,
                              SpinVector::max_size()));
        TCM_CHECK(
            static_cast<unsigned>(std::abs(*magnetisation)) <= n,
            std::invalid_argument,
            fmt::format("magnetisation exceeds the number of spins: |{}| > {}",
                        *magnetisation, n));
        TCM_CHECK(
            (static_cast<int>(n) + *magnetisation) % 2 == 0, std::runtime_error,
            fmt::format("{} spins cannot have a magnetisation of {}. `n + "
                        "magnetisation` must be even",
                        n, *magnetisation));
        alignas(32) float buffer[SpinVector::max_size()];
        auto const        number_downs =
            static_cast<unsigned>((static_cast<int>(n) - *magnetisation) / 2);
        std::fill(buffer, buffer + number_downs, -1.0f);
        std::fill(buffer + number_downs, buffer + n, 1.0f);
        VectorT spins;
        // spins.reserve(???);
        auto s = gsl::span<float>{&buffer[0], n};
        do {
            spins.emplace_back(s, UnsafeTag{});
        } while (std::next_permutation(std::begin(s), std::end(s)));
        return spins;
    }
    else {
        static_assert(SpinVector::max_size() >= 26,
                      TCM_STATIC_ASSERT_BUG_MESSAGE);
        TCM_CHECK(n <= 26, std::overflow_error,
                  fmt::format("too many spins: {}; refuse to allocate more "
                              "than 1GB of storage",
                              n));
        auto const size = 1UL << static_cast<size_t>(n);
        VectorT    spins;
        spins.reserve(size);
        for (auto i = size_t{0}; i < size; ++i) {
            spins.emplace_back(n, i);
        }
        return spins;
    }
}

auto bind_spin(pybind11::module m) -> void
{
    namespace py = pybind11;

    py::class_<SpinVector>(m, "CompactSpin", R"EOF(
        Compact representation of spin configurations. Each spin is encoded in a one bit.
    )EOF")
        .def(py::init<unsigned, uint64_t>(), py::arg{"size"}, py::arg{"data"},
             R"EOF(
                 Creates a compact spin configuration from bits packed into an integer.

                 :param size: number of spins. This parameter can't be deduced
                              from ``data``, because that would discard the leading
                              zeros.
                 :param data: a sequence of bits packed into an int. The value of the
                              ``i``'th spin is given by the ``i``'th most significant
                              bit of ``data``.
             )EOF")
        .def(py::init<torch::Tensor const&>(), py::arg{"x"},
             R"EOF(
                 Creates a compact spin configuration from a tensor.

                 :param x: a one-dimensional tensor of ``float``. ``-1.0`` means
                           spin down and ``1.0`` means spin up.
             )EOF")
        .def(py::init<py::str>(), py::arg{"x"},
             R"EOF(
                 Creates a compact spin configuration from a string.

                 :param x: a string consisting of '0's and '1's. '0' means spin
                           down and '1' means spin up.
             )EOF")
        .def(py::init<py::array_t<float, py::array::c_style> const&>(),
             py::arg{"x"},
             R"EOF(
                 Creates a compact spin configuration from a numpy array.

                 :param x: a one-dimensional contiguous array of ``float``. ``-1.0``
                           means spin down and ``1.0`` means spin up.
             )EOF")
        .def("__copy__", [](SpinVector const& x) { return SpinVector{x}; },
             R"EOF(Copies the current spin configuration.)EOF")
        .def("__deepcopy__",
             [](SpinVector const& x, py::dict /*unused*/) {
                 return SpinVector{x};
             },
             py::arg{"memo"} = py::none(),
             R"EOF(Same as ``self.__copy__()``.)EOF")
        .def("__len__", [](SpinVector const& self) { return self.size(); },
             R"EOF(
                 Returns the number of spins in the spin configuration.
             )EOF")
        .def("__int__",
             [](SpinVector const& self) { return static_cast<size_t>(self); },
             R"EOF(
                 Implements ``int(self)``, i.e. conversion to ``int``.

                 .. warning::

                    This function does not work with spin configurations longer than 64.
             )EOF")
        .def("__str__",
             [](SpinVector const& self) {
                 return static_cast<std::string>(self);
             },
             py::return_value_policy::move,
             R"EOF(
                 Implements ``str(self)``, i.e. conversion to ``str``.
             )EOF")
        .def("__getitem__",
             [](SpinVector const& x, unsigned const i) {
                 return x.at(i) == Spin::up ? 1.0f : -1.0f;
             },
             py::arg{"i"},
             R"EOF(
                 Returns ``self[i]`` as a ``float``.
             )EOF")
        .def("__setitem__",
             [](SpinVector& x, unsigned const i, float const spin) {
                 auto const float2spin = [](auto const s) {
                     if (s == -1.0f) { return Spin::down; }
                     if (s == 1.0f) { return Spin::up; }
                     TCM_ERROR(
                         std::invalid_argument,
                         fmt::format(
                             "Invalid spin {}; expected either -1 or +1", s));
                 };
                 x.at(i) = float2spin(spin);
             },
             py::arg{"i"}, py::arg{"spin"},
             R"EOF(
                 Performs ``self[i] = spin``.

                 ``spin`` must be either ``-1.0`` or ``1.0``.
             )EOF")
        .def("__eq__",
             [](SpinVector const& x, SpinVector const& y) { return x == y; },
             py::is_operator())
        .def("__ne__",
             [](SpinVector const& x, SpinVector const& y) { return x != y; },
             py::is_operator())
        .def_property_readonly(
            "size", [](SpinVector const& self) { return self.size(); },
            R"EOF(Same as ``self.__len__()``.)EOF")
        .def_property_readonly(
            "magnetisation",
            [](SpinVector const& self) { return self.magnetisation(); },
            R"EOF(Returns the magnetisation)EOF")
        .def("numpy", [](SpinVector const& x) { return x.numpy(); },
             py::return_value_policy::move,
             R"EOF(Converts the spin configuration to a numpy.ndarray)EOF")
        .def("tensor", [](SpinVector const& x) { return x.tensor(); },
             py::return_value_policy::move,
             R"EOF(Converts the spin configuration to a torch.Tensor)EOF")
        .def("__hash__", &SpinVector::hash, R"EOF(
                Returns the hash of the spin configuration.
            )EOF");

    m.def("random_spin",
          [](unsigned const size, optional<int> magnetisation) {
              auto& generator = global_random_generator();
              if (magnetisation.has_value()) {
                  return SpinVector::random(size, *magnetisation, generator);
              }
              else {
                  return SpinVector::random(size, generator);
              }
          },
          py::arg{"n"}, py::arg{"magnetisation"} = py::none(),
          R"EOF(
              Generates a random spin configuration.
          )EOF");

    m.def("all_spins", &all_spins);
}

TCM_NAMESPACE_END
