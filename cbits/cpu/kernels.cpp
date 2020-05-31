#include "kernels.hpp"
#include "../errors.hpp"
#include "../tensor_info.hpp"

#include <torch/types.h>
#include <vectorclass/version2/vectorclass.h>
#include <vectorclass/version2/vectormath_exp.h>

// Define name of entry function depending on which instruction set we compile for
#if INSTRSET >= 10 // AVX512VL
#    define zanella_jump_rates_simd zanella_jump_rates_avx512
#elif INSTRSET >= 8 // AVX2
#    define zanella_jump_rates_simd zanella_jump_rates_avx2
#elif INSTRSET >= 7 // AVX
#    define zanella_jump_rates_simd zanella_jump_rates_avx
#elif INSTRSET >= 2 // SSE2
#    define INCLUDE_DISPATCH_CODE
#    define zanella_jump_rates_simd zanella_jump_rates_sse2
#else
#    error Unsupported instruction set
#endif

#if defined(INCLUDE_DISPATCH_CODE)
#    include <caffe2/utils/cpuid.h>
#endif

TCM_NAMESPACE_BEGIN

// Figure out what the underlying element type of a SIMD vector is
template <class V> struct element_type;
template <> struct element_type<vcl::Vec16f> {
    using type = float;
};
template <> struct element_type<vcl::Vec8d> {
    using type = double;
};
template <class V> using element_t = typename element_type<V>::type;

// Figure out, what is the SIMD vector type corresponding to the scalar T
template <class T> struct vector_type;
template <> struct vector_type<float> {
    using type = vcl::Vec8f;
};
template <> struct vector_type<double> {
    using type = vcl::Vec4d;
};
template <class T> using vector_t = typename vector_type<T>::type;

namespace detail {
namespace {
    template <class T>
    TCM_FORCEINLINE auto loadu(T const* src, int64_t const stride) noexcept
    {
        using V = vector_t<T>;
        V a;
        if (stride == 1) { a.load(src); }
        else {
            alignas(64) T buffer[V::size()];
            for (auto n = 0; n < V::size(); ++n) {
                buffer[n] = src[n * stride];
            }
            a.load_a(buffer);
        }
        return a;
    }

    template <class T>
    TCM_FORCEINLINE auto storeu(T* dst, int64_t const stride,
                                vector_t<T> a) noexcept -> void
    {
        using V = vector_t<T>;
        if (stride == 1) { a.store(dst); }
        else {
            alignas(64) T buffer[V::size()];
            a.store_a(buffer);
            for (auto n = 0; n < V::size(); ++n) {
                dst[n * stride] = buffer[n];
            }
        }
    }

    template <class T>
    TCM_FORCEINLINE auto load_partial(T const* src, int64_t const stride,
                                      int const count) noexcept
    {
        using V = vector_t<T>;
        V a;
        if (stride == 1) { a.load_partial(src); }
        else {
            alignas(64) T buffer[V::size()];
            for (auto n = 0; n < count; ++n) {
                buffer[n] = src[n * stride];
            }
            for (auto n = count; n < V::size(); ++n) {
                buffer[n] = T{0};
            }
            a.load_a(buffer);
        }
        return a;
    }

    template <class T>
    TCM_FORCEINLINE auto store_partial(T* dst, int64_t const stride,
                                       vector_t<T> a, int const count) noexcept
    {
        using V = vector_t<T>;
        if (stride == 1) { a.store_partial(count, dst); }
        else {
            alignas(64) T buffer[V::size()];
            a.store_a(buffer);
            for (auto n = 0; n < count; ++n) {
                dst[n * stride] = buffer[n];
            }
        }
        return a;
    }

    template <class T>
    auto jump_rates_one(TensorInfo<T> out, TensorInfo<T const> log_prob,
                        T scale) -> T
    {
        using V                 = vector_t<T>;
        auto const count        = V::size() * (log_prob.size() / V::size());
        auto const rest         = log_prob.size() % V::size();
        auto const vector_scale = V{scale};
        auto       vector_sum   = V{T{0}};
        auto       i            = int64_t{0};
        for (; i < count; i += V::size()) {
            // Loading strided data into a SIMD vector
            auto v = detail::loadu(log_prob.data + i * log_prob.stride(),
                                   log_prob.stride());
            // The actual computation
            v = vcl::exp(v - vector_scale);
            vector_sum += v;
            // Storing into strided tensor
            detail::storeu(out.data + i * out.stride(), out.stride(), v);
        }
        auto sum = vcl::horizontal_add(vector_sum);
        for (; i < count + rest; ++i) {
            auto const r =
                std::exp(log_prob.data[i * log_prob.stride()] - scale);
            out.data[i * out.stride()] = r;
            sum += r;
        }
        return sum;
    }
} // namespace
} // namespace detail

TCM_EXPORT auto zanella_jump_rates_simd(torch::Tensor current_log_prob,
                                        torch::Tensor proposed_log_prob,
                                        std::vector<int64_t> const& counts)
    -> std::tuple<torch::Tensor, torch::Tensor>
{
    torch::NoGradGuard no_grad;
    TCM_CHECK(current_log_prob.device().is_cpu(), std::invalid_argument,
              fmt::format("current_log_prob must reside on the CPU"));
    TCM_CHECK(proposed_log_prob.device().is_cpu(), std::invalid_argument,
              fmt::format("proposed_log_prob must reside on the CPU"));
    TCM_CHECK(
        current_log_prob.dim() == 1, std::invalid_argument,
        fmt::format("current_log_prob has wrong shape: [{}]; expected a vector",
                    fmt::join(current_log_prob.sizes(), ", ")));
    TCM_CHECK(proposed_log_prob.dim() == 1, std::invalid_argument,
              fmt::format(
                  "proposed_log_prob has wrong shape: [{}]; expected a vector",
                  fmt::join(proposed_log_prob.sizes(), ", ")));
    TCM_CHECK(current_log_prob.scalar_type() == proposed_log_prob.scalar_type(),
              std::invalid_argument,
              fmt::format("current_log_prob and proposed_log_prob have "
                          "different types: {} != {}",
                          current_log_prob.scalar_type(),
                          proposed_log_prob.scalar_type()));

    auto rates     = torch::empty_like(proposed_log_prob);
    auto rates_sum = torch::empty_like(current_log_prob);

    AT_DISPATCH_FLOATING_TYPES(
        current_log_prob.scalar_type(), "zanella_jump_rates", [&] {
            auto       out_data        = rates.data_ptr<scalar_t>();
            auto       log_prob_data   = proposed_log_prob.data_ptr<scalar_t>();
            auto const scale_data      = current_log_prob.data_ptr<scalar_t>();
            auto const sum_data        = rates_sum.data_ptr<scalar_t>();
            auto const out_stride      = rates.stride(0);
            auto const log_prob_stride = proposed_log_prob.stride(0);
            auto const scale_stride    = current_log_prob.stride(0);
            auto const sum_stride      = rates_sum.stride(0);

            for (auto i = int64_t{0}; i < static_cast<int64_t>(counts.size());
                 ++i) {
                auto const n = counts[static_cast<size_t>(i)];
                TCM_CHECK(n >= 0, std::runtime_error, "negative count");
                auto const s = detail::jump_rates_one(
                    TensorInfo<scalar_t>{out_data, &n, &out_stride},
                    TensorInfo<scalar_t const>{log_prob_data, &n,
                                               &log_prob_stride},
                    scale_data[i * scale_stride]);
                sum_data[i * sum_stride] = s;

                out_data += n * out_stride;
                log_prob_data += n * log_prob_stride;
            }
        });
    return std::make_tuple(std::move(rates), std::move(rates_sum));
}

#if defined(INCLUDE_DISPATCH_CODE)
TCM_EXPORT auto zanella_jump_rates(torch::Tensor current_log_prob,
                                   torch::Tensor proposed_log_prob,
                                   std::vector<int64_t> const& counts)
    -> std::tuple<torch::Tensor, torch::Tensor>
{
    auto& cpuid = caffe2::GetCpuId();
    if (cpuid.avx2()) {
        return zanella_jump_rates_avx2(std::move(current_log_prob),
                                       std::move(proposed_log_prob), counts);
    }
    else if (cpuid.avx()) {
        return zanella_jump_rates_avx(std::move(current_log_prob),
                                      std::move(proposed_log_prob), counts);
    }
    else {
        return zanella_jump_rates_sse2(std::move(current_log_prob),
                                       std::move(proposed_log_prob), counts);
    }
}
#endif

TCM_NAMESPACE_END
