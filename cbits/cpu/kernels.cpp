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

/// Load a simd vector at index i
template <class T>
TCM_FORCEINLINE auto vload(TensorInfo<T const> src, int64_t const i) noexcept
    -> vector_t<T>
{
    using V = vector_t<T>;
    TCM_ASSERT((0 <= i) && (i + V::size() - 1 < src.size()), "index out of bounds");
    auto const* p = src.data + i * src.stride();

    V a;
    if (src.stride() == 1) { a.load(p); }
    else {
        // TODO: Use AVX2 vgather instructions when available
        alignas(64) T buffer[V::size()];
        for (auto n = 0; n < V::size(); ++n) {
            buffer[n] = p[n * src.stride()];
        }
        a.load_a(buffer);
    }
    return a;
}

template <class T>
TCM_FORCEINLINE auto vstore(TensorInfo<T> dst, int64_t i,
                            vector_t<T> a) noexcept -> void
{
    using V = vector_t<T>;
    TCM_ASSERT((0 <= i) && (i + V::size() - 1 < dst.size()), "index out of bounds");
    auto* p = dst.data + i * dst.stride();

    if (dst.stride() == 1) { a.store(p); }
    else {
        // TODO: Use AVX2 vscatter instructions when available
        alignas(64) T buffer[V::size()];
        a.store_a(buffer);
        for (auto n = 0; n < V::size(); ++n) {
            p[n * dst.stride()] = buffer[n];
        }
    }
}

namespace {
template <class T>
auto jump_rates_one(TensorInfo<T> out, TensorInfo<T const> log_prob, T scale)
    -> T
{
    using V           = vector_t<T>;
    auto const count  = V::size() * (log_prob.size() / V::size());
    auto const rest   = log_prob.size() % V::size();
    auto const vscale = V{scale};
    auto       vsum   = V{T{0}};
    auto       i      = int64_t{0};
    for (; i < count; i += V::size()) {
        auto v = vload(log_prob, i);
        v      = vcl::min(vcl::exp(v - vscale), V{T{1}});
        vsum += v;
        vstore(out, i, v);
    }
    auto sum = vcl::horizontal_add(vsum);
    for (; i < count + rest; ++i) {
        auto const r = std::min(
            std::exp(log_prob.data[i * log_prob.stride()] - scale), T{1});
        out.data[i * out.stride()] = r;
        sum += r;
    }
    return sum;
}
} // namespace

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
            auto out_info = obtain_tensor_info<scalar_t>(rates);
            auto sum_info = obtain_tensor_info<scalar_t>(rates_sum);
            auto log_prob_info =
                obtain_tensor_info<scalar_t const>(proposed_log_prob);
            auto scale_info =
                obtain_tensor_info<scalar_t const>(current_log_prob);

            auto offset = int64_t{0};
            for (auto i = int64_t{0}; i < static_cast<int64_t>(counts.size());
                 ++i) {
                auto const n = counts[static_cast<size_t>(i)];
                TCM_CHECK(n >= 0, std::runtime_error, "negative count");
                TCM_CHECK(offset + n <= out_info.size(), std::runtime_error,
                          "sum of counts exceeds the size of current_log_prob");
                sum_info[i] = jump_rates_one(
                    slice(out_info, offset, offset + n),
                    slice(log_prob_info, offset, offset + n), scale_info[i]);
                offset += n;
            }
            TCM_CHECK(
                offset == out_info.size(), std::runtime_error,
                "sum of counts is smaller than the size of current_log_prob");
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
