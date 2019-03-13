#include "nqs.hpp"

// #include <boost/sort/spreadsort/integer_sort.hpp>
// #include <parallel/algorithm>
// #include <pybind11/numpy.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
#include <pybind11/iostream.h>

// #if defined(TCM_PYTORCH_EXTENSION)
// #include <torch/extension.h>
// #include <torch/script.h>
// #endif

namespace py = ::pybind11;
// namespace tcm = ::TCM_NAMESPACE;

inline auto bind_spin_vector(py::module m) -> void
{
    using namespace tcm;

    py::class_<SpinVector>(m, "CompactSpin", R"EOF(
        Compact representation of spin configurations
    )EOF")
        .def(py::init<torch::Tensor const&>(), py::arg("x"),
             R"EOF(
                 Creates a compact spin configuration from a tensor.

                 :param x: a one-dimensional tensor of ``float``. ``-1.0`` means
                           spin down and ``1.0`` means spin up.
             )EOF")
        .def(py::init<py::str>(), py::arg("x"),
             R"EOF(
                 Creates a compact spin configuration from a string.

                 :param x: a string consisting of '0's and '1's. '0' means spin
                           down and '1' means spin up.
             )EOF")
        .def(py::init<py::array_t<float, py::array::c_style> const&>(),
             py::arg("x"),
             R"EOF(
                 Creates a compact spin configuration from a numpy array.

                 :param x: a one-dimensional contiguous array of ``float``. ``-1.0``
                           means spin down and ``1.0`` means spin up.
             )EOF")
        .def(
            "__copy__", [](SpinVector const& x) { return SpinVector{x}; },
            R"EOF(Copies the current spin configuration.)EOF")
        .def(
            "__deepcopy__", [](SpinVector const& x) { return SpinVector{x}; },
            R"EOF(Same as ``self.__copy__()``.)EOF")
        .def("__len__", &SpinVector::size,
             R"EOF(
                 Returns the number of spins in the spin configuration.
             )EOF")
        .def("__int__", &SpinVector::operator std::size_t,
             R"EOF(
                 Implements ``int(self)``, i.e. conversion to ``int``.

                 .. warning::

                    This function does not work with spin configurations longer than 64.
             )EOF")
        .def("__str__", &SpinVector::operator std::string,
             R"EOF(
                 Implements ``str(self)``, i.e. conversion to ``str``.
             )EOF")
        .def(
            "__getitem__",
            [](SpinVector const& x, unsigned const i) {
                return x.at(i) == ::TCM_NAMESPACE::Spin::up ? 1.0f : -1.0f;
            },
            py::arg("i"),
            R"EOF(
                 Returns ``self[i]``.
             )EOF")
        .def(
            "__setitem__",
            [](SpinVector& x, unsigned const i, float const spin) {
                auto const float2spin = [](auto const s) {
                    if (s == -1.0f) { return tcm::Spin::down; }
                    if (s == 1.0f) { return tcm::Spin::up; }
                    throw std::invalid_argument{fmt::format(
                        "Invalid spin {}; expected either -1 or +1", s)};
                };
                x.at(i) = float2spin(spin);
            },
            py::arg("i"), py::arg("spin"),
            R"EOF(
                 Performs ``self[i] = spin``.

                 ``spin`` must be either ``-1.0`` or ``1.0``.
             )EOF")
        .def(
            "__eq__",
            [](SpinVector const& x, SpinVector const& y) { return x == y; },
            py::is_operator())
        .def(
            "__ne__",
            [](SpinVector const& x, SpinVector const& y) { return x != y; },
            py::is_operator())
        .def_property_readonly("size", &SpinVector::size,
                               R"EOF(Same as ``self.__len__()``.)EOF")
        .def_property_readonly("magnetisation", &SpinVector::magnetisation,
                               R"EOF(Returns the magnetisation)EOF")
        .def(
            "numpy", [](SpinVector const& x) { return x.numpy(); },
            py::return_value_policy::move,
            R"EOF(Converts the spin configuration to a numpy.ndarray)EOF")
        .def(
            "tensor", [](SpinVector const& x) { return x.tensor(); },
            py::return_value_policy::move,
            R"EOF(Converts the spin configuration to a torch.Tensor)EOF")
        .def("__hash__", &SpinVector::hash, R"EOF(
                Returns the hash of the spin configuration.
            )EOF");
}

inline auto bind_heisenberg(py::module m) -> void
{
    using namespace tcm;
    py::class_<Heisenberg, std::shared_ptr<Heisenberg>>(m, "Heisenberg")
        .def(py::init<std::vector<std::pair<unsigned, unsigned>>, real_type>(),
             py::arg("edges"), py::arg("coupling"),
             R"EOF(
                 Creates an isotropic Heisenberg Hamiltonian from a list of edges.
             )EOF")
        .def("__len__", &Heisenberg::size,
             R"EOF(
                 Returns the number of edges in the graph.
             )EOF")
        .def_property(
            "coupling", [](Heisenberg const& h) { return h.coupling(); },
            [](Heisenberg& h, real_type const coupling) {
                h.coupling(coupling);
            })
        .def("edges", &Heisenberg::edges,
             R"EOF(
                 Returns graph edges

                 .. warning:: This function copies the edges
             )EOF");
}

inline auto bind_polynomial(py::module m) -> void
{
    using namespace tcm;
    py::class_<Polynomial>(m, "Polynomial",
                           R"EOF(
            Represents polynomials in H.
        )EOF")
        .def(
            py::init([](std::shared_ptr<Heisenberg>                        h,
                        std::vector<std::pair<complex_type,
                                              optional<real_type>>> const& ts) {
                // TODO(twesterhout): This function performs an ugly
                // copy... I know it doesn't matter from performance point
                // of view, but it still bugs me.
                std::vector<Polynomial::Term> terms;
                terms.reserve(ts.size());
                std::transform(std::begin(ts), std::end(ts),
                               std::back_inserter(terms),
                               [](auto const& t) -> Polynomial::Term {
                                   return {t.first, t.second};
                               });
                return std::make_unique<Polynomial>(std::move(h),
                                                    std::move(terms));
            }),
            py::arg("hamiltonian"), py::arg("terms"),
            R"EOF(
                 Given a Hamiltonian H and terms (cᵢ, εᵢ) (i ∈ {0, 1, ..., n-1})
                 constructs the following polynomial

                     P[εₙ₋₁](H - cₙ₋₁)...P[ε₂](H - c₂)P[ε₁](H - c₁)P[ε₀](H - c₀)

                 Even though each cᵢ is complex, after expanding the brackets
                 __all coefficients should be real__.

                 P[ε] means a projection from a state ∑aᵢ|σᵢ⟩ to a state ∑bᵢ|σᵢ⟩
                 where bᵢ = aᵢ if aᵢ > ε and bᵢ = 0 otherwise.
             )EOF")
        .def_property_readonly("size", &Polynomial::size,
                               R"EOF(
                Returns the number of different spin configurations in the current
                state. This attribute is only useful after applying the polynomial
                to a state |σ⟩.
            )EOF")
        .def("__len__", &Polynomial::size, R"EOF(Same as ``size``.)EOF")
        .def("__call__", &Polynomial::operator(), py::arg("coeff"),
             py::arg("spin"),
             R"EOF(

             )EOF")
        .def(
            "vectors",
            [](Polynomial const& p) {
                auto const& spins = p.vectors();
                return ::tcm::detail::unpack_to_tensor(std::begin(spins),
                                                       std::end(spins));
            },
            py::return_value_policy::move)
        .def("coefficients", [](Polynomial const& p) {
            return torch::Tensor{p.coefficients()};
        });
}

struct Net : torch::nn::Module {
    Net()
        : _fc1{register_module("_fc1", torch::nn::Linear(48, 291))}
        , _fc2{register_module("_fc2", torch::nn::Linear(291, 1))}
    {}

    auto forward(torch::Tensor x) -> torch::Tensor
    {
        x = torch::relu(_fc1->forward(x));
        x = torch::sigmoid(_fc2->forward(x));
        return x;
    }

    torch::nn::Linear _fc1{nullptr};
    torch::nn::Linear _fc2{nullptr};
};

PYBIND11_MODULE(_C_nqs, m)
{
    m.doc() = R"EOF()EOF";

    using namespace tcm;

    using ::TCM_NAMESPACE::Heisenberg;
    using ::TCM_NAMESPACE::Polynomial;
    using ::TCM_NAMESPACE::PolynomialState;
    using ::TCM_NAMESPACE::SpinVector;
    // using ::TCM_NAMESPACE::Machine;
    // using ::TCM_NAMESPACE::TargetStateImpl;

    using ::TCM_NAMESPACE::complex_type;
    using ::TCM_NAMESPACE::real_type;

    bind_spin_vector(m);
    bind_heisenberg(m);
    bind_polynomial(m);

    m.def("say_hi", []() {
        return std::make_tuple(sizeof(tcm::Polynomial),
                               sizeof(tcm::PolynomialState));
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
        ska_sort(std::begin(array), std::end(array), [](auto const& x) {
            return x.first.key(::TCM_NAMESPACE::detail::unsafe_tag);
        });
        // boost::sort::spreadsort::integer_sort(std::begin(array), std::end(array),
        //     [](auto const& a, unsigned const offset) { return a.first.ska_key() >> offset; },
        //     [](auto const& a, auto const& b) { return a.first.ska_key() < b.first.ska_key(); });
        auto time_interval =
            MicroSecondsT(std::chrono::steady_clock::now() - time_start);
        return time_interval.count();
    });

    // m.def("foo", [](torch::nn::ModuleHolder& x) { auto y = torch::randn({10}, torch::kFloat32); return x.forward(y); });

    m.def(
        "random_spin",
        [](size_t const size, optional<int> magnetisation) {
            auto& generator = global_random_generator();
            if (magnetisation.has_value()) {
                return SpinVector::random(size, *magnetisation, generator);
            }
            else {
                py::print("No magnetisation");
                return SpinVector::random(size, generator);
            }
        },
        py::arg("n"), py::arg("magnetisation") = py::none(),
        R"EOF(
              Generates a random spin configuration.
          )EOF");

    /*
    using Fn = std::function<torch::Tensor(torch::Tensor const&)>;
    py::class_<TargetState<Fn>>(m, "TargetState")
        .def(py::init<Polynomial, Fn>())
        .def("forward", &TargetState<Fn>::forward);
    */

    py::class_<PolynomialState>(m, "TargetState")
        .def(py::init([](std::string const& filename, Polynomial const& poly,
                         std::tuple<size_t, size_t> dim) {
#if 0
            return std::make_unique<PolynomialState>(
                ::TCM_NAMESPACE::detail::load_forward_fn(filename), poly,
                batch_size, filename);
#else
            return std::make_unique<PolynomialState>(
                tcm::detail::load_forward_fn(filename, omp_get_max_threads()),
                Polynomial{poly, SplitTag{}}, dim);
#endif
        }))
        .def("__call__",
             [](PolynomialState& state, SpinVector const input) {
                 // mkl_set_num_threads(1);
                 torch::NoGradGuard no_grad;
                 return state(input);
             })
        .def("__call__",
             [](PolynomialState& state, torch::Tensor const& input) {
                 auto const dim = input.dim();
                 TCM_CHECK_DIM(dim, 1, 2);
                 if (dim == 1) {
                     auto out = tcm::detail::make_tensor<float>(1);
                     out[0]   = state(SpinVector{input});
                     return out;
                 }
                 auto const size = input.size(0);
                 auto       out =
                     tcm::detail::make_tensor<float>(static_cast<size_t>(size));
                 auto out_accessor = out.accessor<float, 1>();
                 for (auto i = int64_t{0}; i < size; ++i) {
                     out_accessor[i] = state(SpinVector{input[i]});
                 }
                 return out;
             })
        .def_property_readonly(
            "time_poly",
            [](PolynomialState const& self) { return self.time_poly(); })
        .def_property_readonly("time_psi", [](PolynomialState const& self) {
            return self.time_psi();
        });

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

    m.def("sample_some",
          [](std::string const& filename, Polynomial const& polynomial,
             Options const& options, int num_threads) {
              return tcm::sample_some(
                         filename, Polynomial{polynomial, SplitTag{}}, options,
                         num_threads > 0 ? num_threads : omp_get_max_threads())
                  .to_tensors();
          },
          py::arg("filename"), py::arg("polynomial"), py::arg("options"),
          py::arg("num_threads") =
              -1 /*, py::call_guard<py::gil_scoped_release>()*/);

#if 1
    m.def(
        "parallel_sample_some",
        [](std::string const& filename, Polynomial const& polynomial,
           Options const& options, std::tuple<unsigned, unsigned> num_threads) {
            py::scoped_ostream_redirect stream(
                    std::cout,                               // std::ostream&
                    py::module::import("sys").attr("stdout") // Python output
                );
            return tcm::parallel_sample_some(filename, polynomial, options,
                                             num_threads)
                .to_tensors();
        },
        py::arg{"filename"}, py::arg{"polynomial"}, py::arg{"options"},
        py::arg{"num_threads"} // , py::call_guard<py::gil_scoped_release>()
    );
#endif

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
