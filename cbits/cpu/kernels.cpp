#include "kernels.hpp"
#include "../errors.hpp"

#include <torch/types.h>
#include <vectorclass/version2/vectorclass.h>
#include <vectorclass/version2/vectormath_exp.h>

// Define name of entry function depending on which instruction set we compile for
#if INSTRSET >= 8 // AVX2
#    define zanella_jump_rates_simd zanella_jump_rates_avx2
#    define unpack_cpu_simd unpack_cpu_avx2
#elif INSTRSET >= 7 // AVX
#    define zanella_jump_rates_simd zanella_jump_rates_avx
#    define unpack_cpu_simd unpack_cpu_avx
#elif INSTRSET >= 2 // SSE2
#    define INCLUDE_DISPATCH_CODE
#    define zanella_jump_rates_simd zanella_jump_rates_sse2
#    define unpack_cpu_simd unpack_cpu_sse2
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

namespace {
template <class T>
auto jump_rates_one(TensorInfo<T> out, TensorInfo<T const> log_prob, T scale)
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
} // namespace

TCM_EXPORT auto zanella_jump_rates_simd(torch::Tensor current_log_prob,
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
} // }}}

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

#endif // INCLUDE_DISPATCH_CODE

// unpack {{{
namespace {
TCM_FORCEINLINE auto unpack_byte(uint8_t const bits) noexcept -> vcl::Vec8f
{
    auto const one = vcl::Vec8f{1.0f}; // 1.0f == 0x3f800000
    auto const two = vcl::Vec8f{2.0f};
    // Adding 0x3f800000 to select ensures that we're working with valid
    // floats rather than denormals
    auto const select = vcl::Vec8f{vcl::reinterpret_f(vcl::Vec8i{
        0x3f800000 + (1 << 0), 0x3f800000 + (1 << 1), 0x3f800000 + (1 << 2),
        0x3f800000 + (1 << 3), 0x3f800000 + (1 << 4), 0x3f800000 + (1 << 5),
        0x3f800000 + (1 << 6), 0x3f800000 + (1 << 7)})};
    auto       broadcasted =
        vcl::Vec8f{vcl::reinterpret_f(vcl::Vec8i{static_cast<int>(bits)})};
    broadcasted |= one;
    broadcasted &= select;
    broadcasted = broadcasted == select;
    broadcasted &= two;
    broadcasted -= one;
    return broadcasted;
}

TCM_NOINLINE auto unpack_word(uint64_t x, float* out) noexcept -> void
{
    for (auto i = 0; i < 8; ++i, out += 8, x >>= 8) {
        unpack_byte(static_cast<uint8_t>(x & 0xFF)).store(out);
    }
}

template <bool Unsafe>
auto unpack_one(uint64_t x, unsigned const number_spins, float* out) noexcept
    -> void
{
    auto const chunks = number_spins / 8U;
    auto const rest   = number_spins % 8U;
    for (auto i = 0U; i < chunks; ++i, out += 8, x >>= 8U) {
        unpack_byte(static_cast<uint8_t>(x & 0xFF)).store(out);
    }
    if (rest != 0) {
        auto const t = unpack_byte(static_cast<uint8_t>(x & 0xFF));
        if constexpr (Unsafe) { t.store(out); }
        else {
            t.store_partial(static_cast<int>(rest), out);
        }
    }
}

template <bool Unsafe>
auto unpack_one(bits512 const& bits, unsigned const count, float* out) noexcept
    -> void
{
    constexpr auto block  = 64U;
    auto const     chunks = count / block;
    auto const     rest   = count % block;

    auto i = 0U;
    for (; i < chunks; ++i, out += block) {
        unpack_word(bits.words[i], out);
    }
    if (rest != 0) { unpack_one<Unsafe>(bits.words[i], rest, out); }
}

struct _IdentityProjection {
    template <class T> constexpr decltype(auto) operator()(T&& x) const noexcept
    {
        return std::forward<T>(x);
    }
};
} // namespace

template <class Bits, class Projection = _IdentityProjection>
auto unpack_impl(TensorInfo<Bits const> const& src_info,
                 TensorInfo<float, 2> const&   dst_info,
                 Projection                    proj = Projection{}) -> void
{
    if (TCM_UNLIKELY(src_info.size() == 0)) { return; }
    TCM_CHECK(
        dst_info.strides[0] == dst_info.sizes[1], std::invalid_argument,
        fmt::format("unpack_cpu_avx does not support strided output tensors"));
    auto const number_spins = static_cast<unsigned>(dst_info.sizes[1]);
    auto const rest         = number_spins % 8;
    auto const tail         = std::min<int64_t>(
        ((8U - rest) + number_spins - 1U) / number_spins, src_info.size());
    auto* src = src_info.data;
    auto* dst = dst_info.data;
    auto  i   = int64_t{0};
    for (; i < src_info.size() - tail;
         ++i, src += src_info.stride(), dst += dst_info.strides[0]) {
        unpack_one</*Unsafe=*/true>(proj(*src), number_spins, dst);
    }
    for (; i < src_info.size();
         ++i, src += src_info.stride(), dst += dst_info.strides[0]) {
        unpack_one</*Unsafe=*/false>(proj(*src), number_spins, dst);
    }
}

template <class Bits>
auto unpack_cpu_simd(TensorInfo<Bits const> const& src_info,
                     TensorInfo<float, 2> const&   dst_info) -> void
{
    unpack_impl(src_info, dst_info);
}

template TCM_EXPORT auto unpack_cpu_simd(TensorInfo<uint64_t const> const&,
                                         TensorInfo<float, 2> const&) -> void;
template TCM_EXPORT auto unpack_cpu_simd(TensorInfo<bits512 const> const&,
                                         TensorInfo<float, 2> const&) -> void;

#if defined(INCLUDE_DISPATCH_CODE)
template <class Bits>
auto unpack_cpu(TensorInfo<Bits const> const& spins,
                TensorInfo<float, 2> const&   out) -> void
{
    auto& cpuid = caffe2::GetCpuId();
    if (cpuid.avx2()) { unpack_cpu_avx2(spins, out); }
    else if (cpuid.avx()) {
        unpack_cpu_avx(spins, out);
    }
    else {
        unpack_cpu_sse2(spins, out);
    }
}

template TCM_EXPORT auto unpack_cpu(TensorInfo<uint64_t const> const&,
                                    TensorInfo<float, 2> const&) -> void;
template TCM_EXPORT auto unpack_cpu(TensorInfo<bits512 const> const&,
                                    TensorInfo<float, 2> const&) -> void;
#endif

TCM_NAMESPACE_END
