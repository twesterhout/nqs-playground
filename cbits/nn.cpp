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

#include "nn.hpp"

TCM_NAMESPACE_BEGIN

// -------------------------------- [Gemm] --------------------------------- {{{
namespace detail {
inline auto make_gemm(MKL_LAYOUT const layout, MKL_TRANSPOSE const transa,
                      MKL_TRANSPOSE const transb, MKL_INT const m,
                      MKL_INT const n, MKL_INT const k, float const alpha,
                      MKL_INT const lda, MKL_INT const ldb, float const beta,
                      MKL_INT const ldc) -> std::shared_ptr<Gemm>
{
    struct Deleter {
        auto operator()(Gemm* gemm) const noexcept -> void
        {
            TCM_ASSERT(gemm != nullptr, "Trying to delete a nullptr");
            auto const status = mkl_jit_destroy(gemm->jitter);
            if (TCM_UNLIKELY(status != MKL_JIT_SUCCESS)) {
                std::fprintf(stderr,
                             "Bug! Invalid pointer passed to mkl_jit_destroy!");
                std::terminate();
            }
        }
    };

    auto       p = std::unique_ptr<Gemm>{new Gemm};
    auto const status =
        mkl_jit_create_sgemm(&p->jitter, layout, transa, transb, m, n, k, alpha,
                             lda, ldb, beta, ldc);
    TCM_CHECK(status == MKL_JIT_SUCCESS || status == MKL_NO_JIT,
              std::runtime_error, "JIT handle cannot be created (no memory)");
    p->func = mkl_jit_get_sgemm_ptr(p->jitter);
    TCM_CHECK(p->func != nullptr, std::runtime_error, "");
    return std::shared_ptr<Gemm>{p.release(), Deleter{}};
}
} // namespace detail
// -------------------------------- [Gemm] --------------------------------- }}}

// ------------------------- [Bias & Activation] --------------------------- {{{
namespace detail {
template <class Activation>
auto bias_and_activation(size_t batch_size, size_t out_features, float* out,
                         size_t ldim, float const* bias, Activation&& func)
    -> void
{
    if (bias != nullptr) {
        for (auto i = size_t{0}; i < batch_size; ++i) {
            for (auto j = size_t{0}; j < out_features; ++j) {
                out[i * ldim + j] = func(out[i * ldim + j] + bias[j]);
            }
        }
    }
    else {
        for (auto i = size_t{0}; i < batch_size; ++i) {
            for (auto j = size_t{0}; j < out_features; ++j) {
                out[i * ldim + j] = func(out[i * ldim + j]);
            }
        }
    }
}

#define TCM_SPECIALISE(type)                                                   \
    template auto bias_and_activation<type>(                                   \
        size_t batch_size, size_t out_features, float* out, size_t ldim,       \
        float const* bias, type func)                                          \
        ->void
TCM_SPECIALISE(ReLU const&);
TCM_SPECIALISE(ReLU&&);
TCM_SPECIALISE(Softplus const&);
TCM_SPECIALISE(Softplus&&);
TCM_SPECIALISE(Tanh const&);
TCM_SPECIALISE(Tanh&&);
TCM_SPECIALISE(Identity const&);
TCM_SPECIALISE(Identity&&);
#undef TCM_SPECIALISE
} // namespace detail
// ------------------------- [Bias & Activation] --------------------------- }}}

// ------------------------- [DisableConversions] -------------------------- {{{
void DisableConversions::to(torch::Device, torch::Dtype, bool)
{
    TCM_ERROR(std::runtime_error,
              "Module::to() family of functions is not supported");
}

void DisableConversions::to(torch::Dtype, bool)
{
    TCM_ERROR(std::runtime_error,
              "Module::to() family of functions is not supported");
}

void DisableConversions::to(torch::Device, bool)
{
    TCM_ERROR(std::runtime_error,
              "Module::to() family of functions is not supported");
}
// ------------------------- [DisableConversions] -------------------------- }}}

// --------------------------- [DenseLayerImpl] ---------------------------- {{{
DenseLayerImpl::DenseLayerImpl(size_t in_features, size_t out_features,
                               size_t batch_size, bool with_bias)
    : DisableConversions{}
    , _weight{}
    , _bias{}
    , _dst{}
    , _in_features{static_cast<int64_t>(in_features)}
    , _out_features{static_cast<int64_t>(out_features)}
    , _batch_size{static_cast<int64_t>(batch_size)}
    , _with_bias{with_bias}
    , _gemm{}
    , _weight_tensor{}
    , _bias_tensor{}
    , _dst_tensor{}
{
    init_buffers();
    init_parameters();
    init_gemm();
    reset();
}

DenseLayerImpl::DenseLayerImpl(DenseLayerImpl const& other)
    : DisableConversions{}
    , _weight{other._weight}
    , _bias{other._bias}
    , _dst(other._dst.size())
    , _in_features{other._in_features}
    , _out_features{other._out_features}
    , _batch_size{other._batch_size}
    , _with_bias{other._with_bias}
    , _gemm{other._gemm}
    , _weight_tensor{}
    , _bias_tensor{}
    , _dst_tensor{}
{
    init_parameters();
}

auto DenseLayerImpl::reset() -> void
{
    torch::NoGradGuard no_grad;
    const auto         stdv = 1.0 / std::sqrt(_in_features);
    _weight_tensor.uniform_(-stdv, stdv);
    if (_with_bias) { _bias_tensor.uniform_(-stdv, stdv); }
}

auto DenseLayerImpl::init_buffers() -> void
{
    _weight.resize(static_cast<size_t>(_out_features * _in_features));
    if (_with_bias) { _bias.resize(static_cast<size_t>(_out_features)); }
    _dst.resize(static_cast<size_t>(_batch_size * _out_features));
}

auto DenseLayerImpl::init_parameters() -> void
{
    _weight_tensor = register_parameter(
        "weight", torch::from_blob(/*data=*/_weight.data(),
                                   /*sizes=*/{_out_features, _in_features},
                                   /*strides=*/{_in_features, 1},
                                   torch::TensorOptions{torch::kFloat32}));
    if (_with_bias) {
        _bias_tensor = register_parameter(
            "bias", torch::from_blob(_bias.data(), {_out_features},
                                     torch::TensorOptions{torch::kFloat32}));
    }
    _dst_tensor = torch::from_blob(_dst.data(), {_batch_size, _out_features},
                                   torch::TensorOptions{torch::kFloat32});
}

auto DenseLayerImpl::init_gemm() -> void
{
    _gemm = detail::make_gemm(MKL_ROW_MAJOR, MKL_NOTRANS, MKL_TRANS,
                              _batch_size, _out_features, _in_features, 1.0f,
                              _in_features, _in_features, 0.0f, _out_features);
}

auto DenseLayerImpl::check_input(torch::Tensor const& input) const -> void
{
    TCM_CHECK_TYPE(input.scalar_type(), torch::kFloat32);
    TCM_CHECK(input.is_contiguous(), std::invalid_argument,
              "input tensor should be contiguous");
    auto const dim = input.dim();
    switch (input.dim()) {
    case 1: TCM_CHECK_SHAPE(input.size(0), _in_features); break;
    case 2:
        if (torch::nn::Module::is_training()) {
            TCM_CHECK_SHAPE(input.size(1), _in_features);
        }
        else {
            TCM_CHECK_SHAPE((std::make_tuple(input.size(0), input.size(1))),
                            (std::make_tuple(_batch_size, _in_features)));
        }
        break;
    default:
        TCM_ERROR(
            std::domain_error,
            fmt::format("wrong dimension {}; expected either 1 or 2", dim));
    }
}

auto DenseLayerImpl::batch_size(size_t const batch_size) -> void
{
    _batch_size = static_cast<int64_t>(batch_size);
    _dst.resize(static_cast<size_t>(_batch_size * _out_features));
    _dst_tensor = torch::from_blob(_dst.data(), {_batch_size, _out_features},
                                   torch::TensorOptions{torch::kFloat32});
    _gemm       = detail::make_gemm(MKL_ROW_MAJOR, MKL_NOTRANS, MKL_TRANS,
                              _batch_size, _out_features, _in_features, 1.0f,
                              _in_features, _in_features, 0.0f, _out_features);
}
// --------------------------- [DenseLayerImpl] ---------------------------- }}}

// ---------------------------- [AmplitudeNet] ----------------------------- {{{
AmplitudeNet::AmplitudeNet(std::array<size_t, 3> sizes, size_t batch_size)
    : DisableConversions{}
    , _layer_1{std::make_shared<DenseLayer<ReLU>>(sizes[0], sizes[1],
                                                  batch_size)}
    , _layer_2{std::make_shared<DenseLayer<ReLU>>(sizes[1], sizes[2],
                                                  batch_size)}
    , _layer_3{std::make_shared<tcm::DenseLayer<tcm::Softplus>>(
          sizes[2], 1, batch_size, false)}
{
    register_module("dense1", _layer_1);
    register_module("dense2", _layer_2);
    register_module("dense3", _layer_3);
}

AmplitudeNet::AmplitudeNet(AmplitudeNet const& other)
    : DisableConversions{}
    , _layer_1{std::make_shared<DenseLayer<ReLU>>(*other._layer_1)}
    , _layer_2{std::make_shared<DenseLayer<ReLU>>(*other._layer_2)}
    , _layer_3{std::make_shared<DenseLayer<Softplus>>(*other._layer_3)}
{
    register_module("dense1", _layer_1);
    register_module("dense2", _layer_2);
    register_module("dense3", _layer_3);
}

auto AmplitudeNet::operator()(torch::Tensor const& input) const -> torch::Tensor
{
    _layer_1->check_input(input);
    return _layer_3->forward(_layer_2->forward(_layer_1->forward(input)));
}
// ---------------------------- [AmplitudeNet] ----------------------------- }}}

// ------------------------------ [PhaseNet] ------------------------------- {{{
PhaseNet::PhaseNet(std::array<size_t, 2> sizes, size_t batch_size)
    : DisableConversions{}
    , _layer_1{std::make_shared<DenseLayer<Tanh>>(sizes[0], sizes[1],
                                                  batch_size)}
    , _layer_2{std::make_shared<DenseLayer<Identity>>(sizes[1], 2, batch_size,
                                                      false)}
{
    register_module("dense1", _layer_1);
    register_module("dense2", _layer_2);
}

PhaseNet::PhaseNet(PhaseNet const& other)
    : DisableConversions{}
    , _layer_1{std::make_shared<DenseLayer<Tanh>>(*other._layer_1)}
    , _layer_2{std::make_shared<DenseLayer<Identity>>(*other._layer_2)}
{
    register_module("dense1", _layer_1);
    register_module("dense2", _layer_2);
}

auto PhaseNet::operator()(torch::Tensor input) const -> torch::Tensor
{
    _layer_1->check_input(input);
    return _layer_2->forward(_layer_1->forward(input));
}
// ------------------------------ [PhaseNet] ------------------------------- }}}

// ------------------------------- [Python] -------------------------------- {{{
namespace {
template <class M>
auto bind_module(pybind11::module m, char const* name)
    -> py::class_<M, std::shared_ptr<M>>
{
    namespace py = pybind11;
    return py::class_<M, std::shared_ptr<M>>(m, name)
        .def("train", [](M& module) { module.train(); })
        .def("eval", [](M& module) { module.eval(); })
        .def_property_readonly(
            "training", [](M const& module) { return module.is_training(); })
        .def("zero_grad", [](M& module) { module.zero_grad(); })
        .def("parameters", [](M& module) { return module.parameters(); })
        .def("named_parameters",
             [](M& module) { return module.named_parameters(); })
        .def("buffers", [](M& module) { return module.buffers(); })
        .def("named_buffers", [](M& module) { return module.named_buffers(); })
        .def("forward",
             [](M& module, torch::Tensor const& x) { return module(x); })
        .def("__call__",
             [](M& module, torch::Tensor const& x) { return module(x); });
}
} // namespace

auto bind_networks(pybind11::module m) -> void
{
    namespace py = pybind11;

    bind_module<AmplitudeNet>(m, "AmplitudeNet")
        .def(py::init<std::array<size_t, 3>, size_t>(), py::arg{"layer_sizes"},
             py::arg{"batch_size"})
        .def("__deepcopy__",
             [](AmplitudeNet const& self, py::dict /*unused*/) {
                 return std::make_shared<AmplitudeNet>(self);
             })
        .def_property_readonly("batch_size", &AmplitudeNet::batch_size);

    bind_module<PhaseNet>(m, "PhaseNet")
        .def(py::init<std::array<size_t, 2>, size_t>(), py::arg{"layer_sizes"},
             py::arg{"batch_size"})
        .def("__deepcopy__",
             [](PhaseNet const& self, py::dict /*unused*/) {
                 return std::make_shared<PhaseNet>(self);
             })
        .def_property_readonly("batch_size", &PhaseNet::batch_size);

    py::class_<CombiningState>(m, "CombiningState")
        .def(py::init<std::shared_ptr<AmplitudeNet>,
                      std::shared_ptr<PhaseNet>>(),
             py::arg("amplitude"), py::arg("phase"))
        .def_property_readonly(
            "amplitude",
            [](CombiningState const& self) { return self.amplitude(); })
        .def_property_readonly(
            "phase", [](CombiningState const& self) { return self.phase(); })
        .def("forward", [](CombiningState&      self,
                           torch::Tensor const& x) { return self(x); })
        .def("__call__", [](CombiningState& self, torch::Tensor const& x) {
            return self(x);
        });
}
// ------------------------------- [Python] -------------------------------- }}}

TCM_NAMESPACE_END
