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

#pragma once

#include "config.hpp"
#include "errors.hpp"
#include <boost/align/aligned_allocator.hpp>
#include <mkl.h>
// #include <sleef.h>
#include <torch/extension.h>
#include <immintrin.h>

#if defined(TCM_GCC)
#    pragma GCC diagnostic push
// #    pragma GCC diagnostic ignored "-Wsign-conversion"
// #    pragma GCC diagnostic ignored "-Wsign-promo"
// #    pragma GCC diagnostic ignored "-Wswitch-default"
// #    pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
// #    pragma GCC diagnostic ignored "-Wstrict-overflow"
#    pragma GCC diagnostic ignored "-Wold-style-cast"
#    pragma GCC diagnostic ignored "-Wshadow"
#elif defined(TCM_CLANG)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-noreturn"
#    pragma clang diagnostic ignored "-Wsign-conversion"
#    pragma clang diagnostic ignored "-Wswitch-enum"
#    pragma clang diagnostic ignored "-Wundefined-func-template"
#endif
#include <pybind11/pybind11.h>
#if defined(TCM_GCC)
#    pragma GCC diagnostic pop
#elif defined(TCM_CLANG)
#    pragma clang diagnostic pop
#endif

TCM_NAMESPACE_BEGIN

// ----------------------------- [Activations] ----------------------------- {{{
/// ReLU activation function
struct ReLU {
    auto operator()(float const x) const noexcept -> float
    {
        return std::max(x, 0.0f);
    }

    auto operator()(__m128 const x) const noexcept -> __m128
    {
        return _mm_max_ps(x, _mm_set1_ps(0.0f));
    }

    auto operator()(__m256 const x) const noexcept -> __m256
    {
        return _mm256_max_ps(x, _mm256_set1_ps(0.0f));
    }

    auto operator()(torch::Tensor const& input) const -> torch::Tensor
    {
        return torch::relu(input);
    }
};

/// Softplus activation function
struct Softplus {
    auto operator()(float const x) const noexcept -> float
    {
        return std::log(1.0f + std::exp(x));
    }

    auto operator()(torch::Tensor const& input) const -> torch::Tensor
    {
        return torch::softplus(input);
    }
};

/// Tanh activation function
struct Tanh {
    auto operator()(float const x) const noexcept -> float
    {
        // return Sleef_tanhf_u10(x);
        return std::tanh(x);
    }

#if 0
    auto operator()(__m128 const x) const noexcept -> __m128
    {
        return Sleef_tanhf4_u10sse4(x);
    }

    auto operator()(__m256 const x) const noexcept -> __m256
    {
        return Sleef_tanhf8_u10avx(x);
    }
#endif

    auto operator()(torch::Tensor const& input) const -> torch::Tensor
    {
        return torch::tanh(input);
    }
};

/// Identity activation function
struct Identity {
    template <class T> constexpr decltype(auto) operator()(T&& x) const noexcept
    {
        return std::forward<T>(x);
    }
};
// ----------------------------- [Activations] ----------------------------- }}}

namespace detail {
struct Gemm {
    sgemm_jit_kernel_t func;
    void*              jitter;

    auto operator()(float const* a, float const* b, float* c) const noexcept
        -> void
    {
        (*func)(jitter, const_cast<float*>(a), const_cast<float*>(b), c);
    }
};

template <class Activation>
auto bias_and_activation(size_t batch_size, size_t out_features, float* out,
                         size_t ldim, float const* bias, Activation&& func)
    -> void;
} // namespace detail

// ------------------------- [DisableConversions] -------------------------- {{{
class DisableConversions : public torch::nn::Module {
  public:
    using torch::nn::Module::Module;

    DisableConversions(DisableConversions const&)     = default;
    DisableConversions(DisableConversions&&) noexcept = default;
    DisableConversions& operator=(DisableConversions const&) = default;
    DisableConversions& operator=(DisableConversions&&) noexcept = default;

    // to() functions just throw. We override them to "fail fast".
    virtual void to(torch::Device, torch::Dtype,
                    bool non_blocking = false) override;
    virtual void to(torch::Dtype, bool non_blocking = false) override;
    virtual void to(torch::Device, bool non_blocking = false) override;
};
// ------------------------- [DisableConversions] -------------------------- }}}

// --------------------------- [DenseLayerImpl] ---------------------------- {{{
class DenseLayerImpl : public DisableConversions {
  public:
    using buffer_type =
        std::vector<float, boost::alignment::aligned_allocator<float, 64>>;

  protected:
    buffer_type                   _weight;
    buffer_type                   _bias;
    buffer_type                   _dst;
    int64_t                       _in_features;
    int64_t                       _out_features;
    int64_t                       _batch_size;
    bool                          _with_bias;
    std::shared_ptr<detail::Gemm> _gemm;
    torch::Tensor                 _weight_tensor;
    torch::Tensor                 _bias_tensor;
    torch::Tensor                 _dst_tensor;

  private:
    auto init_buffers() -> void;
    auto init_parameters() -> void;
    auto init_gemm() -> void;

  public:
    DenseLayerImpl(size_t in_features, size_t out_features, size_t batch_size,
                   bool with_bias = true);

    DenseLayerImpl(DenseLayerImpl const& other);
    DenseLayerImpl(DenseLayerImpl&&) = delete;
    DenseLayerImpl& operator=(DenseLayerImpl const&) = delete;
    DenseLayerImpl& operator=(DenseLayerImpl&&) = delete;

    auto           reset() -> void;
    auto           check_input(torch::Tensor const& input) const -> void;
    auto           batch_size(size_t batch_size) -> void;
    constexpr auto batch_size() const noexcept -> size_t;
    constexpr auto weight_tensor() const noexcept -> torch::Tensor const&;
    constexpr auto bias_tensor() const noexcept -> torch::Tensor const&;
};

constexpr auto DenseLayerImpl::weight_tensor() const noexcept
    -> torch::Tensor const&
{
    return _weight_tensor;
}

constexpr auto DenseLayerImpl::bias_tensor() const noexcept
    -> torch::Tensor const&
{
    return _bias_tensor;
}

constexpr auto DenseLayerImpl::batch_size() const noexcept -> size_t
{
    return static_cast<size_t>(_batch_size);
}
// --------------------------- [DenseLayerImpl] ---------------------------- }}}

// ----------------------------- [DenseLayer] ------------------------------ {{{
template <class Activation>
class DenseLayer
    : private Activation
    , public DenseLayerImpl {
  public:
    using DenseLayerImpl::DenseLayerImpl;

  private:
    inline auto apply_bias_and_activation() -> void;
    inline auto forward_fast(torch::Tensor const& input) -> torch::Tensor;
    inline auto forward_slow(torch::Tensor const& input) -> torch::Tensor;

  public:
    inline auto forward(torch::Tensor const& input) -> torch::Tensor;
    inline auto operator()(torch::Tensor const& input) -> torch::Tensor;
};

template <class Activation>
auto DenseLayer<Activation>::apply_bias_and_activation() -> void
{
    TCM_ASSERT(_with_bias || _bias.data() == nullptr, "");
    detail::bias_and_activation(
        static_cast<size_t>(_batch_size), static_cast<size_t>(_out_features),
        _dst.data(), static_cast<size_t>(_out_features), _bias.data(),
        static_cast<Activation const&>(*this));
}

template <class Activation>
auto DenseLayer<Activation>::forward_slow(torch::Tensor const& input)
    -> torch::Tensor
{
    TCM_ASSERT(!_with_bias || _bias_tensor.defined(), "");
    return static_cast<Activation const&>(*this)(
        torch::linear(input, _weight_tensor, _bias_tensor));
}

template <class Activation>
auto DenseLayer<Activation>::forward_fast(torch::Tensor const& input)
    -> torch::Tensor
{
    if (input.dim() == 2) {
        // cblas_sgemm(
        //     /*layout=*/CblasRowMajor, /*transa=*/CblasNoTrans,
        //     /*transb=*/CblasTrans, /*m=*/_batch_size, /*n=*/_out_features,
        //     /*k=*/_in_features,
        //     /*alpha=*/1.0f, /*a=*/input.data<float>(), /*lda=*/_in_features,
        //     /*b=*/_weight.data(), /*ldb=*/_in_features, /*beta=*/0.0f,
        //     /*c=*/_dst.data(), /*ldc=*/_out_features);
        (*_gemm)(static_cast<float const*>(input.data_ptr()), _weight.data(),
                 _dst.data());
        apply_bias_and_activation();
        return _dst_tensor;
    }
    else {
        TCM_ASSERT(input.dim() == 1, "");
        cblas_sgemv(/*layout=*/CblasRowMajor, /*trans=*/CblasNoTrans,
                    /*m=*/_out_features, /*n=*/_in_features,
                    /*alpha=*/1.0f, /*a=*/_weight.data(), /*lda=*/_in_features,
                    /*x=*/input.data<float>(), /*incx=*/1, /*beta*/ 0.0f,
                    /*y=*/_dst.data(), /*incy=*/1);
        detail::bias_and_activation(
            1, static_cast<size_t>(_out_features), _dst.data(),
            static_cast<size_t>(_out_features), _bias.data(),
            static_cast<Activation const&>(*this));
        return torch::from_blob(_dst.data(), {_out_features},
                                torch::TensorOptions{torch::kFloat32});
    }
}

template <class Activation>
auto DenseLayer<Activation>::forward(torch::Tensor const& input)
    -> torch::Tensor
{
    return torch::nn::Module::is_training() ? forward_slow(input)
                                            : forward_fast(input);
}

template <class Activation>
auto DenseLayer<Activation>::operator()(torch::Tensor const& input)
    -> torch::Tensor
{
    return forward(input);
}
// ----------------------------- [DenseLayer] ------------------------------ }}}

// ---------------------------- [AmplitudeNet] ----------------------------- {{{
class AmplitudeNet : public DisableConversions {
    std::shared_ptr<DenseLayer<ReLU>>     _layer_1;
    std::shared_ptr<DenseLayer<ReLU>>     _layer_2;
    std::shared_ptr<DenseLayer<Softplus>> _layer_3;

  public:
    AmplitudeNet(std::array<size_t, 3> sizes, size_t batch_size);
    AmplitudeNet(AmplitudeNet const&);
    AmplitudeNet(AmplitudeNet&&) noexcept = default;
    AmplitudeNet& operator=(AmplitudeNet const&) = delete;
    AmplitudeNet& operator=(AmplitudeNet&&) = delete;

    inline auto batch_size() const noexcept -> size_t;
    auto        operator()(torch::Tensor const& input) const -> torch::Tensor;
};

auto AmplitudeNet::batch_size() const noexcept -> size_t
{
    return _layer_1->batch_size();
}
// ---------------------------- [AmplitudeNet] ----------------------------- }}}

// ------------------------------ [PhaseNet] ------------------------------- {{{
class PhaseNet : public DisableConversions {
    std::shared_ptr<DenseLayer<Tanh>>     _layer_1;
    std::shared_ptr<DenseLayer<Identity>> _layer_2;

  public:
    PhaseNet(std::array<size_t, 2> sizes, size_t batch_size);
    PhaseNet(PhaseNet const& other);
    PhaseNet(PhaseNet&&) noexcept = default;
    PhaseNet& operator=(PhaseNet const& other) = delete;
    PhaseNet& operator=(PhaseNet&&) noexcept = delete;

    inline auto batch_size() const noexcept -> size_t;
    auto        operator()(torch::Tensor input) const -> torch::Tensor;
};

inline auto PhaseNet::batch_size() const noexcept -> size_t
{
    return _layer_1->batch_size();
}
// ------------------------------ [PhaseNet] ------------------------------- }}}

// --------------------------- [CombiningState] ---------------------------- {{{
inline auto combine_amplitude_and_phase(torch::Tensor&       amplitude,
                                        torch::Tensor const& phase)
{
    if (amplitude.dim() == 1) {
        auto amplitude_accessor = amplitude.accessor<float, 1>();
        auto phase_accessor     = phase.accessor<float, 1>();
        TCM_CHECK_SHAPE(amplitude_accessor.size(0), 1);
        TCM_CHECK_SHAPE(phase_accessor.size(0), 2);
        auto const flag = phase_accessor[0] < phase_accessor[1];
        if (flag) { amplitude_accessor[0] *= -1.0f; }
        return;
    }
    auto amplitude_accessor = amplitude.accessor<float, 2>();
    auto phase_accessor     = phase.accessor<float, 2>();
    TCM_CHECK_SHAPE(amplitude_accessor.size(0), phase_accessor.size(0));
    TCM_CHECK_SHAPE(phase_accessor.size(1), 2);
    TCM_CHECK_SHAPE(amplitude_accessor.size(1), 1);
    for (auto i = int64_t{0}; i < amplitude_accessor.size(0); ++i) {
        auto const flag = phase_accessor[i][0] < phase_accessor[i][1];
        if (flag) { amplitude_accessor[i][0] *= -1.0f; }
    }
}

class CombiningState {
    std::shared_ptr<AmplitudeNet> _amplitude;
    std::shared_ptr<PhaseNet>     _phase;

  public:
    CombiningState(std::shared_ptr<AmplitudeNet> amplitude,
                   std::shared_ptr<PhaseNet>     phase)
        : _amplitude{std::move(amplitude)}, _phase{std::move(phase)}
    {}

    CombiningState(CombiningState const& other)
        : _amplitude{std::make_shared<AmplitudeNet>(*other._amplitude)}
        , _phase{std::make_shared<PhaseNet>(*other._phase)}
    {}
    CombiningState(CombiningState&&) noexcept = default;
    CombiningState& operator=(CombiningState const&) = delete;
    CombiningState& operator=(CombiningState&&) noexcept = default;

    auto amplitude() const noexcept -> std::shared_ptr<AmplitudeNet>
    {
        return _amplitude;
    }

    auto phase() const noexcept -> std::shared_ptr<PhaseNet> { return _phase; }

    auto operator()(torch::Tensor const& input) const -> torch::Tensor
    {
        auto       amplitude = (*_amplitude)(input);
        auto const phase     = (*_phase)(input);
        combine_amplitude_and_phase(amplitude, phase);
        return amplitude;
    }
};
// --------------------------- [CombiningState] ---------------------------- }}}

auto bind_networks(py::module m) -> void;

TCM_NAMESPACE_END

namespace pybind11 {
namespace detail {
    template <class Key, class Value>
    struct type_caster<::torch::OrderedDict<Key, Value>> {
      public:
        using type       = ::torch::OrderedDict<Key, Value>;
        using key_conv   = make_caster<Key>;
        using value_conv = make_caster<Value>;

        PYBIND11_TYPE_CASTER(type, _("Dict[") + key_conv::name + _(", ")
                                       + value_conv::name + _("]"));

        auto load(handle src, bool convert) -> bool
        {
            if (!isinstance<dict>(src)) { return false; }
            auto d = reinterpret_borrow<dict>(src);
            value.clear();
            for (auto it : d) {
                key_conv   kconv;
                value_conv vconv;
                if (!kconv.load(it.first.ptr(), convert)
                    || !vconv.load(it.second.ptr(), convert)) {
                    return false;
                }
                value.insert(cast_op<Key&&>(std::move(kconv)),
                             cast_op<Value&&>(std::move(vconv)));
            }
            return true;
        }

        template <typename T>
        static handle cast(T&& src, return_value_policy policy, handle parent)
        {
            dict d;
            auto policy_key   = policy;
            auto policy_value = policy;
            if (!std::is_lvalue_reference<T>::value) {
                policy_key =
                    return_value_policy_override<Key>::policy(policy_key);
                policy_value =
                    return_value_policy_override<Value>::policy(policy_value);
            }
            for (auto&& kv : src) {
                auto key   = reinterpret_steal<object>(key_conv::cast(
                    forward_like<T>(kv.key()), policy_key, parent));
                auto value = reinterpret_steal<object>(value_conv::cast(
                    forward_like<T>(kv.value()), policy_value, parent));
                if (!key || !value) { return handle(); }
                d[key] = value;
            }
            return d.release();
        }
    };
} // namespace detail
} // namespace pybind11
