#include "kernels.hpp"
#include "../errors.hpp"

#include <torch/types.h>
#include <vectorclass/version2/vectorclass.h>
#include <vectorclass/version2/vectormath_exp.h>

// Define name of entry function depending on which instruction set we compile for
#if INSTRSET >= 8 // AVX2
// #    define zanella_jump_rates_simd zanella_jump_rates_avx2
#    define jump_rates_one_simd jump_rates_one_avx2
#    define unpack_one_simd unpack_one_avx2
#    define bfly_simd bfly_avx2
#    define ibfly_simd ibfly_avx2
#elif INSTRSET >= 7 // AVX
// #    define zanella_jump_rates_simd zanella_jump_rates_avx
#    define jump_rates_one_simd jump_rates_one_avx
#    define unpack_one_simd unpack_one_avx
#    define bfly_simd bfly_avx
#    define ibfly_simd ibfly_avx
#elif INSTRSET >= 2 // SSE2
#    define INCLUDE_DISPATCH_CODE
// #    define zanella_jump_rates_simd zanella_jump_rates_sse2
#    define jump_rates_one_simd jump_rates_one_sse2
#    define unpack_one_simd unpack_one_sse2
#    define bfly_simd bfly_sse2
#    define ibfly_simd ibfly_sse2
#else
#    error Unsupported instruction set
#endif

#if defined(INCLUDE_DISPATCH_CODE)
#    include <ATen/Config.h>
#    include <caffe2/utils/cpuid.h>

#    define MKL_INT int
extern "C" void cblas_cdotu_sub(const MKL_INT n, const void* x,
                                const MKL_INT incx, const void* y,
                                const MKL_INT incy, void* dotu);
#endif

TCM_NAMESPACE_BEGIN

// element_t {{{
// Figure out what the underlying element type of a SIMD vector is
template <class V> struct element_type;
template <> struct element_type<vcl::Vec16f> {
    using type = float;
};
template <> struct element_type<vcl::Vec8d> {
    using type = double;
};
template <class V> using element_t = typename element_type<V>::type;
// }}}

// vector_t {{{
// Figure out, what is the SIMD vector type corresponding to the scalar T
template <class T> struct vector_type;
template <> struct vector_type<float> {
    using type = vcl::Vec8f;
};
template <> struct vector_type<double> {
    using type = vcl::Vec4d;
};
template <class T> using vector_t = typename vector_type<T>::type;
// }}}

namespace {
/// Load a simd vector at index i
template <class T>
TCM_FORCEINLINE auto vload(TensorInfo<T const> src, int64_t const i) noexcept
    -> vector_t<T>
{ // {{{
    using V = vector_t<T>;
    TCM_ASSERT((0 <= i) && (i + V::size() - 1 < src.size()),
               "index out of bounds");
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
} // }}}

template <class T>
TCM_FORCEINLINE auto vstore(TensorInfo<T> dst, int64_t i,
                            vector_t<T> a) noexcept -> void
{ // {{{
    using V = vector_t<T>;
    TCM_ASSERT((0 <= i) && (i + V::size() - 1 < dst.size()),
               "index out of bounds");
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
} // }}}
} // namespace

#if 0
template <class T>
auto jump_rates_one_simd(TensorInfo<T> const&       out,
                         TensorInfo<T const> const& log_prob, T scale) noexcept
    -> T
{ // {{{
    using V          = vector_t<T>;
    auto const count = static_cast<int64_t>(
        V::size() * (static_cast<uint64_t>(log_prob.size()) / V::size()));
    auto const rest = static_cast<int64_t>(
        static_cast<uint64_t>(log_prob.size()) % V::size());
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
} // }}}

template TCM_EXPORT auto jump_rates_one_simd(TensorInfo<float> const&,
                                             TensorInfo<float const> const&,
                                             float) noexcept -> float;
template TCM_EXPORT auto jump_rates_one_simd(TensorInfo<double> const&,
                                             TensorInfo<double const> const&,
                                             double) noexcept -> double;

#    if defined(INCLUDE_DISPATCH_CODE)
template <class T>
using jump_rates_one_for_t = auto (*)(TensorInfo<T> const&,
                                      TensorInfo<T const> const&, T) noexcept
                             -> T;

TCM_EXPORT auto zanella_jump_rates(torch::Tensor current_log_prob,
                                   torch::Tensor proposed_log_prob,
                                   std::vector<int64_t> const& counts)
    -> std::tuple<torch::Tensor, torch::Tensor>
{ // {{{
    torch::NoGradGuard no_grad;
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
    if (!current_log_prob.device().is_cpu()) {
        current_log_prob = current_log_prob.cpu();
    }
    if (!proposed_log_prob.device().is_cpu()) {
        proposed_log_prob = proposed_log_prob.cpu();
    }

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
            auto const jump_rates_one_ptr =
                []() -> jump_rates_one_for_t<scalar_t> {
                auto& cpuid = caffe2::GetCpuId();
                if (cpuid.avx2()) { return &jump_rates_one_avx2; }
                if (cpuid.avx()) { return &jump_rates_one_avx; }
                return &jump_rates_one_sse2;
            }();

            auto offset = int64_t{0};
            for (auto i = int64_t{0}; i < static_cast<int64_t>(counts.size());
                 ++i) {
                auto const n = counts[static_cast<size_t>(i)];
                TCM_CHECK(n >= 0, std::runtime_error, "negative count");
                TCM_CHECK(offset + n <= out_info.size(), std::runtime_error,
                          "sum of counts exceeds the size of current_log_prob");
                sum_info[i] = (*jump_rates_one_ptr)(
                    slice(out_info, offset, offset + n),
                    slice(log_prob_info, offset, offset + n), scale_info[i]);
                offset += n;
            }
            TCM_CHECK(
                offset == out_info.size(), std::runtime_error,
                "sum of counts is smaller than the size of current_log_prob");
        });
    return std::make_tuple(std::move(rates), std::move(rates_sum));
} // }}}

template <class scalar_t>
auto tabu_jump_rates(gsl::span<scalar_t const> proposed_log_prob,
                     scalar_t const current_log_prob) -> std::vector<scalar_t>
{ // {{{
    auto       rates = std::vector<scalar_t>(proposed_log_prob.size());
    auto const jump_rates_one_ptr = []() -> jump_rates_one_for_t<scalar_t> {
        auto& cpuid = caffe2::GetCpuId();
        if (cpuid.avx2()) { return &jump_rates_one_avx2; }
        if (cpuid.avx()) { return &jump_rates_one_avx; }
        return &jump_rates_one_sse2;
    }();
    (*jump_rates_one_ptr)(TensorInfo<scalar_t>{rates},
                          TensorInfo<scalar_t const>{proposed_log_prob},
                          current_log_prob);
    return rates;
} // }}}

template TCM_EXPORT auto
tabu_jump_rates(gsl::span<float const> proposed_log_prob,
                float                  current_log_prob) -> std::vector<float>;

template TCM_EXPORT auto
tabu_jump_rates(gsl::span<double const> proposed_log_prob,
                double current_log_prob) -> std::vector<double>;
#    endif
#endif

#if defined(INCLUDE_DISPATCH_CODE)
#    if !AT_MKL_ENABLED()

TCM_EXPORT auto dotu_cpu(TensorInfo<std::complex<float> const> const& x,
                         TensorInfo<std::complex<float> const> const& y)
    -> std::complex<double>
{
    TCM_ERROR(std::runtime_error, "PyTorch is compiled without MKL support");
}

#    else // AT_MKL_ENABLED

TCM_EXPORT auto dotu_cpu(TensorInfo<std::complex<float> const> const& x,
                         TensorInfo<std::complex<float> const> const& y)
    -> std::complex<double>
{
    std::complex<float> result;
    cblas_cdotu_sub(x.size(), x.data, x.stride(), y.data, y.stride(), &result);
    return static_cast<std::complex<double>>(result);
}

#    endif // AT_MKL_ENABLED
#endif     // INCLUDE_DISPATCH_CODE

namespace {
auto unpack_byte(uint8_t const bits) noexcept -> vcl::Vec8f
{
    auto const  up   = vcl::Vec8f{1.0f};
    auto const  down = vcl::Vec8f{-1.0f};
    vcl::Vec8fb mask;
    mask.load_bits(bits);
    return vcl::select(mask, up, down);
}

auto unpack_word(uint64_t x, float* out) noexcept -> void
{
    for (auto i = 0; i < 8; ++i, out += 8, x >>= 8) {
        unpack_byte(static_cast<uint8_t>(x & 0xFF)).store(out);
    }
}
} // namespace

TCM_EXPORT auto unpack_one_simd(uint64_t const x[], unsigned const number_spins,
                                float* out) noexcept -> void
{
    constexpr auto block = 64U;
    auto const     words = number_spins / block;
    for (auto i = 0U; i < words; ++i, out += block) {
        unpack_word(x[i], out);
    }
    auto const rest_words = number_spins % block;
    if (rest_words != 0) {
        auto const bytes      = rest_words / 8U;
        auto const rest_bytes = rest_words % 8U;
        auto       y          = x[words];
        for (auto i = 0U; i < bytes; ++i, out += 8, y >>= 8U) {
            unpack_byte(static_cast<uint8_t>(y & 0xFF)).store(out);
        }
        if (rest_bytes != 0) {
            auto const t = unpack_byte(static_cast<uint8_t>(y & 0xFF));
            t.store_partial(static_cast<int>(rest_bytes), out);
        }
    }
}

#if defined(INCLUDE_DISPATCH_CODE)
TCM_EXPORT auto unpack_cpu(TensorInfo<uint64_t const, 2> const& src_info,
                           TensorInfo<float, 2> const& dst_info) -> void
{
    if (src_info.size<0>() == 0) { return; }
    TCM_CHECK(
        dst_info.strides[0] == dst_info.size<1>(), std::invalid_argument,
        fmt::format("unpack_cpu does not support strided output tensors"));

    using unpack_one_fn_t =
        auto (*)(uint64_t const[], unsigned, float*) noexcept->void;
    auto unpack_ptr = []() -> unpack_one_fn_t {
        if (__builtin_cpu_supports("avx2")) { return &unpack_one_avx2; }
        if (__builtin_cpu_supports("avx")) { return &unpack_one_avx; }
        return &unpack_one_sse2;
    }();

    auto const number_spins = static_cast<unsigned>(dst_info.sizes[1]);
    for (auto i = int64_t{0}; i < src_info.size<0>(); ++i) {
        (*unpack_ptr)(src_info.data + i * src_info.stride<0>(), number_spins,
                      dst_info.data + i * dst_info.stride<0>());
    }
}
#endif

#if 0
namespace {
/// Performs one step of the Butterfly network. It exchanges bits with distance
/// \p d between them if the corresponding bits in the mask \p m are set.
TCM_FORCEINLINE auto bit_permute_step(__m128i& x0, __m128i& x1, __m128i& x2,
                                      __m128i& x3, __m128i m0, __m128i m1,
                                      __m128i m2, __m128i m3,
                                      __m128i d) noexcept -> void
{
    __m128i y0, y1, y2, y3;
    y0 = _mm_srl_epi64(x0, d);
    y1 = _mm_srl_epi64(x1, d);
    y2 = _mm_srl_epi64(x2, d);
    y3 = _mm_srl_epi64(x3, d);
    y0 = _mm_xor_si128(x0, y0);
    y1 = _mm_xor_si128(x1, y1);
    y2 = _mm_xor_si128(x2, y2);
    y3 = _mm_xor_si128(x3, y3);
    y0 = _mm_and_si128(y0, m0);
    y1 = _mm_and_si128(y1, m1);
    y2 = _mm_and_si128(y2, m2);
    y3 = _mm_and_si128(y3, m3);
    x0 = _mm_xor_si128(x0, y0);
    x1 = _mm_xor_si128(x1, y1);
    x2 = _mm_xor_si128(x2, y2);
    x3 = _mm_xor_si128(x3, y3);
    y0 = _mm_sll_epi64(y0, d);
    y1 = _mm_sll_epi64(y1, d);
    y2 = _mm_sll_epi64(y2, d);
    y3 = _mm_sll_epi64(y3, d);
    x0 = _mm_xor_si128(x0, y0);
    x1 = _mm_xor_si128(x1, y1);
    x2 = _mm_xor_si128(x2, y2);
    x3 = _mm_xor_si128(x3, y3);
}
} // namespace

TCM_EXPORT auto bfly_simd(uint64_t x[8], uint64_t const (*masks)[8]) noexcept
    -> void
{
    __m128i x0, x1, x2, x3;
    __m128i m0, m1, m2, m3;
    x0 = _mm_load_si128(reinterpret_cast<__m128i const*>(x));
    x1 = _mm_load_si128(reinterpret_cast<__m128i const*>(x) + 1);
    x2 = _mm_load_si128(reinterpret_cast<__m128i const*>(x) + 2);
    x3 = _mm_load_si128(reinterpret_cast<__m128i const*>(x) + 3);
    for (auto i = 0; i < 6; ++i) {
        m0 = _mm_load_si128(reinterpret_cast<__m128i const*>(masks[i]));
        m1 = _mm_load_si128(reinterpret_cast<__m128i const*>(masks[i]) + 1);
        m2 = _mm_load_si128(reinterpret_cast<__m128i const*>(masks[i]) + 2);
        m3 = _mm_load_si128(reinterpret_cast<__m128i const*>(masks[i]) + 3);
        bit_permute_step(x0, x1, x2, x3, m0, m1, m2, m3,
                         _mm_cvtsi32_si128(1 << i));
    }
    _mm_store_si128(reinterpret_cast<__m128i*>(x), x0);
    _mm_store_si128(reinterpret_cast<__m128i*>(x) + 1, x1);
    _mm_store_si128(reinterpret_cast<__m128i*>(x) + 2, x2);
    _mm_store_si128(reinterpret_cast<__m128i*>(x) + 3, x3);
}

TCM_EXPORT auto bfly_simd(uint64_t const x, uint64_t out[8],
                          uint64_t const (*masks)[8]) noexcept -> void
{
    __m128i x0, x1, x2, x3;
    __m128i m0, m1, m2, m3;
    x0 = _mm_set1_epi64x(static_cast<int64_t>(x));
    x1 = x0;
    x2 = x0;
    x3 = x0;
    for (auto i = 0; i < 6; ++i) {
        m0 = _mm_load_si128(reinterpret_cast<__m128i const*>(masks[i]));
        m1 = _mm_load_si128(reinterpret_cast<__m128i const*>(masks[i]) + 1);
        m2 = _mm_load_si128(reinterpret_cast<__m128i const*>(masks[i]) + 2);
        m3 = _mm_load_si128(reinterpret_cast<__m128i const*>(masks[i]) + 3);
        bit_permute_step(x0, x1, x2, x3, m0, m1, m2, m3,
                         _mm_cvtsi32_si128(1 << i));
    }
    _mm_store_si128(reinterpret_cast<__m128i*>(out), x0);
    _mm_store_si128(reinterpret_cast<__m128i*>(out) + 1, x1);
    _mm_store_si128(reinterpret_cast<__m128i*>(out) + 2, x2);
    _mm_store_si128(reinterpret_cast<__m128i*>(out) + 3, x3);
}

TCM_EXPORT auto ibfly_simd(uint64_t x[8], uint64_t const (*masks)[8]) noexcept
    -> void
{
    __m128i x0, x1, x2, x3;
    __m128i m0, m1, m2, m3;
    x0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(x));
    x1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(x) + 1);
    x2 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(x) + 2);
    x3 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(x) + 3);
    for (auto i = 6; i-- > 0;) {
        m0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(masks[i]));
        m1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(masks[i]) + 1);
        m2 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(masks[i]) + 2);
        m3 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(masks[i]) + 3);
        bit_permute_step(x0, x1, x2, x3, m0, m1, m2, m3,
                         _mm_cvtsi32_si128(1 << i));
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(x), x0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(x) + 1, x1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(x) + 2, x2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(x) + 3, x3);
}

#    if defined(INCLUDE_DISPATCH_CODE)
TCM_EXPORT auto bfly(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void
{
    auto& cpuid = caffe2::GetCpuId();
    if (cpuid.avx()) { bfly_avx(x, masks); }
    else {
        bfly_sse2(x, masks);
    }
}

TCM_EXPORT auto bfly(uint64_t const x, uint64_t out[8],
                     uint64_t const (*masks)[8]) noexcept -> void
{
    auto& cpuid = caffe2::GetCpuId();
    if (cpuid.avx()) { bfly_avx(x, out, masks); }
    else {
        bfly_sse2(x, out, masks);
    }
}

TCM_EXPORT auto ibfly(uint64_t x[8], uint64_t const (*masks)[8]) noexcept
    -> void
{
    auto& cpuid = caffe2::GetCpuId();
    if (cpuid.avx()) { ibfly_avx(x, masks); }
    else {
        ibfly_sse2(x, masks);
    }
}
#    endif
#endif

TCM_NAMESPACE_END
