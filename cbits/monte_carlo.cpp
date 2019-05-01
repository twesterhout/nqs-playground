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

#include "monte_carlo.hpp"
#include "common.hpp"

#include <gsl/gsl-lite.hpp>

#if defined(TCM_GCC)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wold-style-cast"
#    pragma GCC diagnostic ignored "-Wshadow"
#elif defined(TCM_CLANG)
#    pragma clang diagnostic push
#endif
#include <pybind11/iostream.h>
#if defined(TCM_GCC)
#    pragma GCC diagnostic pop
#elif defined(TCM_CLANG)
#    pragma clang diagnostic pop
#endif

TCM_NAMESPACE_BEGIN

constexpr RandomFlipper::index_type RandomFlipper::number_flips;

RandomFlipper::RandomFlipper(SpinVector const initial_spin,
                             RandomGenerator& generator)
    : _storage(initial_spin.size())
    , _generator{std::addressof(generator)}
    , _i{0}
{
    using std::begin;
    using std::end;
    TCM_CHECK(initial_spin.size() >= number_flips, std::invalid_argument,
              fmt::format("requested number of spin-flips exceeds the number "
                          "of spins in the system: {} > {}.",
                          number_flips, initial_spin.size()));
    std::iota(begin(_storage), end(_storage), 0);
    auto const number_ups =
        static_cast<size_t>(std::partition(begin(_storage), end(_storage),
                                           [spin = initial_spin](auto const i) {
                                               return spin[i] == Spin::up;
                                           })
                            - begin(_storage));
    _ups   = gsl::span<index_type>{_storage.data(), number_ups};
    _downs = gsl::span<index_type>{_storage.data() + number_ups,
                                   _storage.size() - number_ups};
    TCM_CHECK((_ups.size() >= number_flips / 2)
                  && (_downs.size() >= number_flips / 2),
              std::invalid_argument,
              fmt::format("initial spin is invalid. Given {} spins up and {} "
                          "spins down, it's impossible to perform {} "
                          "spin-flips and still preserve the magnetisation.",
                          _ups.size(), _downs.size(), number_flips));
    shuffle();
}

auto RandomFlipper::shuffle() -> void
{
    using std::begin;
    using std::end;
    std::shuffle(begin(_ups), end(_ups), *_generator);
    std::shuffle(begin(_downs), end(_downs), *_generator);
}

auto ChainResult::buffer_info() -> pybind11::buffer_info
{
    return pybind11::buffer_info{
        _samples.data(),             // Pointer to the buffer
        sizeof(ChainState),          // Size of one scalar
        ChainState::struct_format(), // Python struct-style format descriptor
        static_cast<int64_t>(size()) // Size of the buffer
    };
}

auto ChainResult::auto_shrink() -> void
{
    auto it =
        std::lower_bound(std::begin(_samples), std::end(_samples),
                         ChainState::magic_count(), [](auto const& x, size_t) {
                             return x.count != ChainState::magic_count();
                         });
    TCM_CHECK(std::all_of(it, std::end(_samples),
                          [](auto const& x) {
                              return x.count == ChainState::magic_count();
                          }),
              std::runtime_error, "This should not have happened");
    TCM_CHECK(std::all_of(std::begin(_samples), it,
                          [](auto const& x) {
                              return x.count != ChainState::magic_count();
                          }),
              std::runtime_error, "This should not have happened");
    _samples.erase(it, std::end(_samples));
}

namespace detail {
TCM_FORCEINLINE auto offsetof_value() noexcept -> int64_t
{
    constexpr auto x = ChainState::magic();
    return static_cast<char const*>(static_cast<void const*>(&x.value))
           - static_cast<char const*>(static_cast<void const*>(&x));
}
} // namespace detail

auto ChainResult::values() -> torch::Tensor
{
    static_assert(sizeof(ChainState) % sizeof(real_type) == 0,
                  TCM_STATIC_ASSERT_BUG_MESSAGE);
    auto const offset = detail::offsetof_value();
    auto*      data =
        static_cast<char*>(static_cast<void*>(_samples.data())) + offset;
    return torch::from_blob(
        /*data=*/data,
        /*sizes=*/{static_cast<int64_t>(_samples.size())},
        /*strides=*/{sizeof(ChainState) / sizeof(real_type)},
        /*options=*/torch::TensorOptions{torch::kFloat64});
#if 0
    auto out      = detail::make_tensor<float>(_samples.size());
    auto accessor = out.accessor<float, 1>();
    for (auto i = size_t{0}; i < _samples.size(); ++i) {
        accessor[static_cast<int64_t>(i)] =
            static_cast<float>(_samples[i].value);
    }
    return out;
#endif
}

auto ChainResult::vectors() const -> torch::Tensor
{
    return unpack_to_tensor(
        std::begin(_samples), std::end(_samples),
        [](auto const& x) -> SpinVector const& { return x.spin; });
}

auto ChainResult::counts() const -> torch::Tensor
{
    auto out      = detail::make_tensor<int64_t>(_samples.size());
    auto accessor = out.accessor<int64_t, 1>();
    for (auto i = size_t{0}; i < _samples.size(); ++i) {
        accessor[static_cast<int64_t>(i)] =
            static_cast<int64_t>(_samples[i].count);
    }
    return out;
}

auto ChainResult::values(torch::Tensor const& xs) -> void
{
    TCM_CHECK_DIM(xs.dim(), 1);
    TCM_CHECK_SHAPE(xs.size(0), static_cast<int64_t>(_samples.size()));
    TCM_CHECK_TYPE(xs.scalar_type(), torch::kFloat32);

    auto accessor = xs.accessor<float, 1>();
    for (auto i = size_t{0}; i < _samples.size(); ++i) {
        _samples[i].value =
            static_cast<real_type>(accessor[static_cast<int64_t>(i)]);
    }
}

auto merge(ChainResult const& _x, ChainResult const& _y) -> ChainResult
{
    auto const&           x = _x.samples();
    auto const&           y = _y.samples();
    ChainResult::SamplesT buffer;
    buffer.reserve(x.size() + y.size());

    std::merge(std::begin(x), std::end(x), std::begin(y), std::end(y),
               std::back_inserter(buffer));
    buffer.erase(compress(std::begin(buffer), std::end(buffer),
                          std::equal_to<void>{},
                          [](auto& acc, auto&& value) {
                              return acc.merge(std::move(value));
                          }),
                 std::end(buffer));
    return ChainResult{std::move(buffer)};
}

auto merge(std::vector<ChainResult>&& results) -> ChainResult
{
    if (results.empty()) { return {}; }
    for (auto i = size_t{1}; i < results.size(); ++i) {
        results[0] = merge(results[0], results[i]);
    }
    return std::move(results[0]);
}

auto sample_some(std::string const& filename, Polynomial const& polynomial,
                 Options const& options, int num_threads) -> ChainResult
{
    if (num_threads <= 0) { num_threads = omp_get_max_threads(); }
    PolynomialState state{
        load_forward_fn(filename, static_cast<size_t>(num_threads)),
        Polynomial{polynomial, SplitTag{}},
        std::make_tuple(options.batch_size, options.number_spins)};
    return sample_some(state, options);
}

namespace detail {
inline auto state_to_forward_fn(CombiningState const& psi, size_t count)
    -> std::vector<ForwardT>
{
    if (count == 0) { return {}; }

    std::vector<ForwardT> modules;
    modules.reserve(count);
    // Shallow copy
    modules.emplace_back(CombiningState{psi.amplitude(), psi.phase()});
    for (auto i = size_t{1}; i < count; ++i) {
        // Deep copy
        modules.emplace_back(CombiningState{psi});
    }
    return modules;
}
} // namespace detail

auto sample_some(CombiningState const& psi, Polynomial const& polynomial,
                 Options const& options, int num_threads) -> ChainResult
{
    if (num_threads <= 0) { num_threads = omp_get_max_threads(); }
    PolynomialState state{
        detail::state_to_forward_fn(psi, static_cast<size_t>(num_threads)),
        Polynomial{polynomial, SplitTag{}},
        std::make_tuple(options.batch_size, options.number_spins)};
    return sample_some(state, options);
}

namespace detail {
template <class F>
struct _NewState {
    F                forward;
    torch::Tensor    spin;
    gsl::span<float> buffer;

    _NewState(F function, size_t const number_spins)
        : forward{std::move(function)}
        , spin{detail::make_tensor<float>(number_spins)}
    {
        buffer = gsl::span<float>{spin.data<float>(), number_spins};
    }

    _NewState(_NewState const&) = delete;
    _NewState(_NewState&&)      = delete;
    _NewState& operator=(_NewState const&) = delete;
    _NewState& operator=(_NewState&&) = delete;

    auto operator()(SpinVector const& x) const -> float
    {
        TCM_ASSERT(x.size() == buffer.size(), "spin chain has wrong length");
        x.copy_to(buffer);
        return forward(spin).template item<float>();
    }
};
} // namespace detail

auto sample_difference(std::string const& new_state_filename,
                       std::string const& old_state_filename,
                       Polynomial const& polynomial, Options const& options,
                       int num_threads) -> ChainResult
{
    if (num_threads <= 0) { num_threads = omp_get_max_threads(); }
    detail::_NewState<ForwardT> new_state{load_forward_fn(new_state_filename),
                                          options.number_spins};
    PolynomialState   old_state{
        load_forward_fn(old_state_filename, static_cast<size_t>(num_threads)),
        Polynomial{polynomial, SplitTag{}},
        std::make_tuple(options.batch_size, options.number_spins)};
    return sample_some(
        [&new_state, &old_state](auto const& x) {
            return old_state(x) - new_state(x);
        },
        options);
}

auto sample_difference(CombiningState const& new_psi,
                       CombiningState const& old_psi,
                       Polynomial const& polynomial, Options const& options,
                       int num_threads) -> ChainResult
{
    if (num_threads <= 0) { num_threads = omp_get_max_threads(); }
    detail::_NewState<std::reference_wrapper<CombiningState const>> new_state{
        std::cref(new_psi), options.number_spins};
    PolynomialState old_state{
        detail::state_to_forward_fn(old_psi, static_cast<size_t>(num_threads)),
        Polynomial{polynomial, SplitTag{}},
        std::make_tuple(options.batch_size, options.number_spins)};
    return sample_some(
        [&new_state, &old_state](auto const& x) {
            return old_state(x) - new_state(x);
        },
        options);
}

auto sample_amplitude_difference(std::string const& new_state_filename,
                                 std::string const& old_state_filename,
                                 Polynomial const&  polynomial,
                                 Options const& options, int num_threads)
    -> ChainResult
{
    if (num_threads <= 0) { num_threads = omp_get_max_threads(); }
    detail::_NewState<ForwardT> new_state{load_forward_fn(new_state_filename),
                                          options.number_spins};
    PolynomialState   old_state{
        load_forward_fn(old_state_filename, static_cast<size_t>(num_threads)),
        Polynomial{polynomial, SplitTag{}},
        std::make_tuple(options.batch_size, options.number_spins)};
    return sample_some(
        [&new_state, &old_state](auto const& x) {
            auto const y = new_state(x);
            TCM_ASSERT(y >= 0, "");
            return std::abs(old_state(x)) - std::abs(y);
        },
        options);
}

auto sample_amplitude_difference(AmplitudeNet const&   new_psi,
                                 CombiningState const& old_psi,
                                 Polynomial const&     polynomial,
                                 Options const& options, int num_threads)
    -> ChainResult
{
    if (num_threads <= 0) { num_threads = omp_get_max_threads(); }
    detail::_NewState<std::reference_wrapper<AmplitudeNet const>> new_state{
        std::cref(new_psi), options.number_spins};
    PolynomialState old_state{
        detail::state_to_forward_fn(old_psi, static_cast<size_t>(num_threads)),
        Polynomial{polynomial, SplitTag{}},
        std::make_tuple(options.batch_size, options.number_spins)};
    return sample_some(
        [&new_state, &old_state](auto const& x) {
            auto const y = new_state(x);
            TCM_ASSERT(y >= 0, "");
            return std::abs(old_state(x)) - std::abs(y);
        },
        options);
}

auto bind_options(pybind11::module m) -> void
{
    namespace py = pybind11;
    py::class_<Options>(m, "ChainOptions")
        .def(py::init<unsigned, int, unsigned, std::array<unsigned, 4>>(),
             py::arg("number_spins"), py::arg("magnetisation"),
             py::arg("batch_size"), py::arg("steps"))
        .def_property_readonly(
            "number_spins",
            [](Options const& self) { return self.number_spins; })
        .def_property_readonly(
            "magnetisation",
            [](Options const& self) { return self.magnetisation; })
        .def_property_readonly("steps",
                               [](Options const& self) { return self.steps; })
        .def("__str__",
             [](Options const& self) { return fmt::format("{}", self); })
        .def("__repr__",
             [](Options const& self) { return fmt::format("{}", self); });
}

auto bind_chain_result(pybind11::module m) -> void
{
    namespace py = pybind11;
    py::class_<ChainState>(m, "ChainState")
        .def(py::init<SpinVector, real_type, size_t>(), py::arg{"spin"},
             py::arg{"value"}, py::arg{"count"})
        .def_property_readonly("spin",
                               [](ChainState const& self) { return self.spin; })
        .def_property_readonly(
            "value", [](ChainState const& self) { return self.value; })
        .def_property_readonly(
            "count", [](ChainState const& self) { return self.count; })
        .def("__str__",
             [](ChainState const& self) {
                 return fmt::format("({}, {}, {})",
                                    static_cast<std::string>(self.spin),
                                    self.value, self.count);
             })
        .def("__repr__", [](ChainState const& self) {
            return fmt::format("ChainState(spin={}, value={}, count={})",
                               static_cast<std::string>(self.spin), self.value,
                               self.count);
        });

    py::class_<ChainResult, std::shared_ptr<ChainResult>>(m, "ChainResult",
                                                          py::buffer_protocol())
        .def("__len__", &ChainResult::size)
        .def_property_readonly("number_spins", &ChainResult::number_spins)
        .def("values", [](ChainResult& self) { return self.values(); })
        .def("values",
             [](ChainResult& self, torch::Tensor new_values) {
                 return self.values(new_values);
             })
        .def("vectors", [](ChainResult const& self) { return self.vectors(); })
        .def("counts", [](ChainResult const& self) { return self.counts(); })
        .def_buffer(&ChainResult::buffer_info)
        .def_static(
            "_allocate",
            [](size_t const size) {
                ChainResult::SamplesT samples(size, ChainState::magic());
                return std::make_shared<ChainResult>(std::move(samples));
            })
        .def("_shrink", [](ChainResult& self) { self.auto_shrink(); })
        .def("merge", [](ChainResult& self, ChainResult const& other) {
            self = merge(self, other);
        });
}

auto bind_sampling(pybind11::module m) -> void
{
    namespace py = pybind11;
    m.def("sample_target_state",
          [](CombiningState const& psi, Polynomial const& polynomial,
             Options const& options, int num_threads) {
              return sample_some(psi, polynomial, options, num_threads);
          },
          py::arg{"psi"}, py::arg{"polynomial"}, py::arg{"options"},
          py::arg{"num_threads"} =
              -1 // , py::call_guard<py::gil_scoped_release>()
    );

    m.def("sample_target_state",
          [](std::string const& filename, Polynomial const& polynomial,
             Options const& options, int num_threads) {
              return sample_some(filename, polynomial, options, num_threads);
          },
          py::arg{"filename"}, py::arg{"polynomial"}, py::arg{"options"},
          py::arg{"num_threads"} =
              -1 // , py::call_guard<py::gil_scoped_release>()
    );

    m.def("sample_difference",
          [](CombiningState const& current, CombiningState const& old,
             Polynomial const& polynomial, Options const& options,
             int num_threads) {
              // TODO(twesterhout): Remove this and add gil_scoped_release
              py::scoped_ostream_redirect stream(
                  std::cout,                               // std::ostream&
                  py::module::import("sys").attr("stdout") // Python output
              );
              return sample_difference(current, old, polynomial, options,
                                       num_threads);
          },
          py::arg{"current"}, py::arg{"old"}, py::arg{"polynomial"},
          py::arg{"options"},
          py::arg{"num_threads"} =
              -1 // , py::call_guard<py::gil_scoped_release>()
    );

    m.def("sample_difference",
          [](std::string const& current, std::string const& old,
             Polynomial const& polynomial, Options const& options,
             int num_threads) {
              // TODO(twesterhout): Remove this and add gil_scoped_release
              py::scoped_ostream_redirect stream(
                  std::cout,                               // std::ostream&
                  py::module::import("sys").attr("stdout") // Python output
              );
              return sample_difference(current, old, polynomial, options,
                                       num_threads);
          },
          py::arg{"current"}, py::arg{"old"}, py::arg{"polynomial"},
          py::arg{"options"},
          py::arg{"num_threads"} =
              -1 // , py::call_guard<py::gil_scoped_release>()
    );

    m.def("sample_amplitude_difference",
          [](AmplitudeNet const& current, CombiningState const& old,
             Polynomial const& polynomial, Options const& options,
             int num_threads) {
              // TODO(twesterhout): Remove this and add gil_scoped_release
              py::scoped_ostream_redirect stream(
                  std::cout,                               // std::ostream&
                  py::module::import("sys").attr("stdout") // Python output
              );
              return sample_amplitude_difference(current, old, polynomial,
                                                 options, num_threads);
          },
          py::arg{"current"}, py::arg{"old"}, py::arg{"polynomial"},
          py::arg{"options"},
          py::arg{"num_threads"} =
              -1 // , py::call_guard<py::gil_scoped_release>()
    );

    m.def("sample_amplitude_difference",
          [](std::string const& current, std::string const& old,
             Polynomial const& polynomial, Options const& options,
             int num_threads) {
              // TODO(twesterhout): Remove this and add gil_scoped_release
              py::scoped_ostream_redirect stream(
                  std::cout,                               // std::ostream&
                  py::module::import("sys").attr("stdout") // Python output
              );
              return sample_amplitude_difference(current, old, polynomial,
                                                 options, num_threads);
          },
          py::arg{"current"}, py::arg{"old"}, py::arg{"polynomial"},
          py::arg{"options"},
          py::arg{"num_threads"} =
              -1 // , py::call_guard<py::gil_scoped_release>()
    );
}

TCM_NAMESPACE_END
