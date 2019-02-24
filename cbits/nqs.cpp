#include "nqs.hpp"
#include <ska_sort.hpp>

#include <boost/sort/spreadsort/integer_sort.hpp>
#include <parallel/algorithm>
// #include <pybind11/numpy.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <pybind11/functional.h>

// #if defined(TCM_PYTORCH_EXTENSION)
// #include <torch/extension.h>
// #include <torch/script.h>
// #endif

#if defined(TCM_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-W"
#pragma GCC diagnostic warning "-Wall"
#pragma GCC diagnostic warning "-Wextra"
#pragma GCC diagnostic warning "-Wcast-align"
#pragma GCC diagnostic warning "-Wcast-qual"
#pragma GCC diagnostic warning "-Wctor-dtor-privacy"
#pragma GCC diagnostic warning "-Wdisabled-optimization"
#pragma GCC diagnostic warning "-Wformat=2"
#pragma GCC diagnostic warning "-Winit-self"
#pragma GCC diagnostic warning "-Wlogical-op"
#pragma GCC diagnostic warning "-Wmissing-declarations"
#pragma GCC diagnostic warning "-Wmissing-include-dirs"
#pragma GCC diagnostic warning "-Wnoexcept"
#pragma GCC diagnostic warning "-Wold-style-cast"
#pragma GCC diagnostic warning "-Woverloaded-virtual"
#pragma GCC diagnostic warning "-Wredundant-decls"
#pragma GCC diagnostic warning "-Wshadow"
#pragma GCC diagnostic warning "-Wsign-conversion"
#pragma GCC diagnostic warning "-Wsign-promo"
#pragma GCC diagnostic warning "-Wstrict-null-sentinel"
#pragma GCC diagnostic warning "-Wstrict-overflow=5"
#pragma GCC diagnostic warning "-Wswitch-default"
#pragma GCC diagnostic warning "-Wundef"
#pragma GCC diagnostic warning "-Wunused"
#endif

TCM_NAMESPACE_BEGIN

namespace detail {
// [errors.implementation] {{{
auto error_wrong_dim(char const* function, int64_t const dim,
                     int64_t const expected) -> void
{
    TCM_ASSERT(dim != expected, "What are you complaining about?");
    throw std::domain_error{std::string{function} + ": wrong dimension "
                            + std::to_string(dim) + "; expected "
                            + std::to_string(expected)};
}

auto error_wrong_dim(char const* function, int64_t const dim,
                     int64_t const expected1, int64_t const expected2) -> void
{
    TCM_ASSERT(dim != expected1 && dim != expected2,
               "What are you complaining about?");
    TCM_ASSERT(expected1 != expected2, "Use the other overload instead");
    throw std::domain_error{std::string{function} + ": wrong dimension "
                            + std::to_string(dim) + "; expected either "
                            + std::to_string(expected1) + " or "
                            + std::to_string(expected2)};
}

auto error_wrong_shape(char const* function, int64_t const shape,
                       int64_t const expected) -> void
{
    TCM_ASSERT(shape != expected, "What are you complaining about?");
    throw std::domain_error{std::string{function} + ": wrong shape ["
                            + std::to_string(shape) + "]; expected ["
                            + std::to_string(expected) + "]"};
}

auto error_wrong_shape(char const*                         function,
                       std::tuple<int64_t, int64_t> const& shape,
                       std::tuple<int64_t, int64_t> const& expected) -> void
{
    TCM_ASSERT(shape != expected, "What are you complaining about?");
    throw std::domain_error{std::string{function} + ": wrong shape ["
                            + std::to_string(std::get<0>(shape)) + ", "
                            + std::to_string(std::get<1>(shape))
                            + "]; expected ["
                            + std::to_string(std::get<0>(expected)) + ", "
                            + std::to_string(std::get<1>(expected)) + "]"};
}

auto error_not_contiguous(char const* function) -> void
{
    throw std::invalid_argument{std::string{function}
                                + ": input must be contiguous"};
}

auto error_wrong_type(char const* function, torch::ScalarType const type,
                      torch::ScalarType const expected) -> void
{
    TCM_ASSERT(type != expected, "What are you complaining about?");
    std::ostringstream msg;
    msg << function << ": wrong type " << type << "; expected " << expected;
    throw std::domain_error{msg.str()};
}

auto error_index_out_of_bounds(char const* function, size_t const i,
                               size_t const max) -> void
{
    TCM_ASSERT(i >= max, "What are you complaining about?");
    throw std::invalid_argument{std::string{function} + ": index out of bounds "
                                + std::to_string(i) + "; expected <"
                                + std::to_string(max)};
}

auto error_float_not_isfinite(char const* function, float const x) -> void
{
    TCM_ASSERT(!std::isfinite(x), "What are you complaining about?");
    throw std::runtime_error{std::string{function}
                             + ": expected a finite float (i.e. either "
                               "normal, subnormal or zero); got "
                             + std::to_string(x)};
}

auto error_float_not_isfinite(char const* function, std::complex<float> const x)
    -> void
{
    TCM_ASSERT(!std::isfinite(x.real()) || !std::isfinite(x.imag()),
               "What are you complaining about?");
    throw std::runtime_error{
        std::string{function}
        + ": expected a finite complex number (i.e. either "
          "normal, subnormal or zero); got "
        + std::to_string(x.real()) + " + " + std::to_string(x.imag()) + "i"};
}

auto error_float_not_isfinite(char const*                function,
                              std::complex<double> const x) -> void
{
    TCM_ASSERT(!std::isfinite(x.real()) || !std::isfinite(x.imag()),
               "What are you complaining about?");
    throw std::runtime_error{
        std::string{function}
        + ": expected a finite complex number (i.e. either "
          "normal, subnormal or zero); got "
        + std::to_string(x.real()) + " + " + std::to_string(x.imag()) + "i"};
}
// [errors.implementation] }}}
} // namespace detail



namespace detail {
auto make_what_message(char const* file, size_t const line,
                       char const* function, std::string const& description)
    -> std::string
{
    return fmt::format("{}:{}: {}: {}", file, line, function, description);
}

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
} // namespace detail

TCM_NAMESPACE_END

#if defined(TCM_GCC)
#pragma GCC diagnostic pop
#endif

namespace py = pybind11;

PYBIND11_MODULE(_C_nqs, m)
{
    m.doc() = R"EOF()EOF";

    using ::TCM_NAMESPACE::Heisenberg;
    using ::TCM_NAMESPACE::Polynomial;
    using ::TCM_NAMESPACE::PolynomialState;
    using ::TCM_NAMESPACE::SpinVector;
    // using ::TCM_NAMESPACE::Machine;
    // using ::TCM_NAMESPACE::TargetStateImpl;

    using ::TCM_NAMESPACE::real_type;
    using ::TCM_NAMESPACE::complex_type;

    m.def("say_hi", [](torch::optional<int> const& x) {
        if (x.has_value()) { py::print("Has value:", x); }
        else {
            py::print("No value!");
        }
    });

    m.def("do_sort", [](Polynomial const& p) {
        using MicroSecondsT =
            std::chrono::duration<double, std::chrono::microseconds::period>;
        std::vector<std::pair<SpinVector, std::complex<double>>> array;
        array.reserve(p.size());
        std::transform(
            std::begin(p.vectors()), std::end(p.vectors()),
            std::back_inserter(array), [](auto const s) {
                return std::pair<SpinVector, std::complex<double>>{s, 0.0};
            });
        auto time_start = std::chrono::steady_clock::now();
        // __gnu_parallel::sort(std::begin(array), std::end(array),
        //     [](auto const& a, auto const& b) { return a.first.ska_key() < b.first.ska_key(); });
        ska_sort(std::begin(array), std::end(array),
                 [](auto const& x) { return x.first.ska_key(); });
        // boost::sort::spreadsort::integer_sort(std::begin(array), std::end(array),
        //     [](auto const& a, unsigned const offset) { return a.first.ska_key() >> offset; },
        //     [](auto const& a, auto const& b) { return a.first.ska_key() < b.first.ska_key(); });
        auto time_interval =
            MicroSecondsT(std::chrono::steady_clock::now() - time_start);
        return time_interval.count();
    });

    // m.def("foo", [](torch::nn::ModuleHolder& x) { auto y = torch::randn({10}, torch::kFloat32); return x.forward(y); });

    py::class_<SpinVector>(m, "CompactSpin",
        R"EOF( Hello world! )EOF")
        .def(py::init<torch::Tensor const&>())
        .def(py::init<py::str>())
        .def(py::init<py::array_t<float, py::array::c_style>>())
        .def("__copy__", [](SpinVector const& x) { return SpinVector{x}; },
             R"EOF(Copies the current spin configuration.)EOF")
        .def("__deepcopy__", [](SpinVector const& x) { return SpinVector{x}; },
             R"EOF(Same as ``self.__copy__()``.)EOF")
        .def("__len__", &SpinVector::size, R"EOF(
            Returns the number of spins in the vector.
        )EOF")
        .def("__int__", &SpinVector::operator std::size_t)
        .def("__str__", &SpinVector::operator std::string)
        .def("__getitem__",
             [](SpinVector const& x, unsigned const i) {
                 return x.at(i) == ::TCM_NAMESPACE::Spin::up ? 1.0f : -1.0f;
             })
        .def("__setitem__",
             [](SpinVector& x, unsigned const i, float const spin) {
                 auto const float2spin = [](auto const s) {
                     if (s == -1.0f) { return ::TCM_NAMESPACE::Spin::down; }
                     if (s == 1.0f) { return ::TCM_NAMESPACE::Spin::up; }
                     throw std::invalid_argument{
                         "Invalid spin: expected either -1 or +1, but got "
                         + std::to_string(s) + "."};
                 };
                 x.at(i) = float2spin(spin);
             })
        .def("__eq__",
             [](SpinVector const& x, SpinVector const& y) { return x == y; },
             py::is_operator())
        .def("__ne__",
             [](SpinVector const& x, SpinVector const& y) { return x != y; },
             py::is_operator())
        .def_property_readonly("size", &SpinVector::size,
                               R"EOF(Same as ``self.__len__()``.)EOF")
        .def_property_readonly("magnetisation", &SpinVector::magnetisation)
        .def("numpy", [](SpinVector const& x) { return x.numpy(); },
             py::return_value_policy::move)
        .def("numpy",
             [](SpinVector const&                      x,
                py::array_t<float, py::array::c_style> out) {
                 return x.numpy(std::move(out));
             },
             py::return_value_policy::move)
        .def("__hash__", &SpinVector::hash, R"EOF(
            Returns the hash of the spin configuration.
        )EOF");

    py::class_<Heisenberg, std::shared_ptr<Heisenberg>>(m, "Heisenberg")
        .def(py::init<std::vector<std::pair<unsigned, unsigned>>, real_type>(),
             py::arg("edges"), py::arg("coupling"),
             R"EOF(Constructs Heisenberg Hamiltonian from a list of edges.)EOF")
        .def("__len__", &Heisenberg::size,
             R"EOF(Returns the number of edges.)EOF")
        .def_property("coupling",
                      [](Heisenberg const& h) { return h.coupling(); },
                      [](Heisenberg& h, real_type const coupling) {
                          h.coupling(coupling);
                      })
        .def("edges", &Heisenberg::edges,
             R"EOF(
                 Returns graph edges

                 .. warning:: This function copies the edges
             )EOF");

    py::class_<Polynomial>(m, "Polynomial",
                           R"EOF(
            Represents polynomials in H.
        )EOF")
        .def(py::init(
                 [](std::shared_ptr<Heisenberg>                           h,
                    std::vector<std::pair<
                        complex_type, torch::optional<real_type>>> const& ts) {
                     std::vector<Polynomial::Term> terms;
                     terms.reserve(ts.size());
                     std::transform(ts.cbegin(), ts.cend(),
                                    std::back_inserter(terms),
                                    [](auto const& t) -> Polynomial::Term {
                                        return {t.first, t.second};
                                    });
                     return std::make_unique<Polynomial>(std::move(h),
                                                         std::move(terms));
                 }),
             py::arg("hamiltonian"), py::arg("terms"),
             R"EOF(
                Given a Hamiltonian H and coefficients cᵢ (i ∈ {0, 1, ..., n-1})
                constructs the following polynomial

                (H - c₀)(H - c₁)(H - c₂)...(H - cₙ₋₁)

                Even though each cᵢ is complex, after expanding the brackets all
                coefficients should be real.
             )EOF")
        .def_property_readonly("size", &Polynomial::size,
                               R"EOF(
                Returns the number of different spin configurations in the current
                state.
            )EOF")
        .def("__len__", &Polynomial::size, R"EOF(Same as ``size``.)EOF")
        // .def("print", &Polynomial::print)
        .def("__call__", &Polynomial::operator(), py::arg("coeff"),
             py::arg("spin"))
        .def("keys",
             [](Polynomial const& p) {
                 if (p.size() == 0) {
                     return ::TCM_NAMESPACE::detail::make_f32_tensor(0);
                 }
                 auto const number_spins = p.vectors().front().size();
                 auto       output = ::TCM_NAMESPACE::detail::make_f32_tensor(
                     p.size(), number_spins);
                 auto* data = p.vectors().data();
                 ::TCM_NAMESPACE::detail::unpack_to_tensor(
                     /*first=*/data, /*last=*/data + p.size(),
                     /*destination=*/output);
                 return output;
             },
             py::return_value_policy::move)
        .def("values",
             [](Polynomial const& p) { return torch::Tensor{p.coefficients()}; })
#if 0
        .def("keys",
             [](Polynomial const&                      p,
                py::array_t<float, py::array::c_style> out) {
                 p.keys(std::move(out));
             })
        .def("values",
             [](Polynomial const&                                     p,
                py::array_t<std::complex<float>, py::array::c_style>& out) {
                 p.values(out);
             })
        .def("values", [](Polynomial const&                       p,
                          py::array_t<float, py::array::c_style>& out) {
            p.values(out);
        })
#endif
        ;

    /*
    using Fn = std::function<torch::Tensor(torch::Tensor const&)>;
    py::class_<TargetState<Fn>>(m, "TargetState")
        .def(py::init<Polynomial, Fn>())
        .def("forward", &TargetState<Fn>::forward);
    */

    py::class_<PolynomialState>(m, "TargetState")
        .def(py::init([](std::string const& filename, Polynomial const& poly,
                         size_t batch_size) {
            return std::make_unique<PolynomialState>(
                ::TCM_NAMESPACE::detail::load_forward_fn(filename), poly,
                batch_size);
        }))
        .def("__call__", [](PolynomialState& state,
                            SpinVector const input) { return state(input); })
        .def("__call__",
             [](PolynomialState& state, torch::Tensor const& input) {
                 return state(SpinVector{input});
             })
        .def_property_readonly("time_poly", &PolynomialState::time_poly)
        .def_property_readonly("time_psi", &PolynomialState::time_psi)
        ;

#if 0
    torch::python::bind_module<TargetStateImpl>(m, "TargetState")
        .def(py::init<std::shared_ptr<Machine>, std::shared_ptr<Polynomial>, size_t>(),
             py::arg("machine"), py::arg("poly"), py::arg("batch_size") = 512)
        .def("forward", &TargetStateImpl::forward);

    py::class_<Machine, std::shared_ptr<Machine>>(m, "Machine")
        .def(py::init([](std::string const& filename) {
            return std::make_shared<Machine>(
                ::TCM_NAMESPACE::detail::load_forward_fn(filename));
        }))
        .def(py::init<std::function<torch::Tensor(torch::Tensor const&)>>())
        // .def("psi", &MachineImpl<torch::jit::script::Module>::psi)
        .def("forward", [](Machine& machine, torch::Tensor const& input) {
            return machine.forward(input);
        });
#endif
}
