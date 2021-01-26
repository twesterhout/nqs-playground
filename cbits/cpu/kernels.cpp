#include "kernels.hpp"
#include "../common/config.hpp"
#include "../common/errors.hpp"

#include <torch/types.h>
#include <vectorclass/version2/vectorclass.h>

#if defined(TCM_COMPILE_avx2)
#    define unpack_one_simd unpack_one_avx2
#elif defined(TCM_COMPILE_avx)
#    define unpack_one_simd unpack_one_avx
#elif defined(TCM_COMPILE_sse2)
#    define unpack_one_simd unpack_one_sse2
#else
#    error "One of TCM_COMPILE_avx2, TCM_COMPILE_avx, or TCM_COMPILE_sse2 must be defined"
#endif

TCM_NAMESPACE_BEGIN

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

#if defined(TCM_ADD_DISPATCH_CODE)
TCM_EXPORT auto unpack_cpu(TensorInfo<uint64_t const, 2> const& src_info,
                           TensorInfo<float, 2> const&          dst_info) -> void
{
    if (src_info.size<0>() == 0) { return; }
    TCM_CHECK(dst_info.strides[0] == dst_info.size<1>(), std::invalid_argument,
              fmt::format("unpack_cpu does not support strided output tensors"));

    using unpack_one_fn_t = auto (*)(uint64_t const[], unsigned, float*) noexcept->void;
    auto unpack_ptr       = []() -> unpack_one_fn_t {
#if TCM_HAS_AVX2()
        return &unpack_one_avx2;
#elif TCM_HAS_AVX()
        if (__builtin_cpu_supports("avx2")) { return &unpack_one_avx2; }
        return &unpack_one_avx;
#else
        if (__builtin_cpu_supports("avx2")) { return &unpack_one_avx2; }
        if (__builtin_cpu_supports("avx")) { return &unpack_one_avx; }
        return &unpack_one_sse2;
#endif
    }();

    auto const number_spins = static_cast<unsigned>(dst_info.sizes[1]);
    for (auto i = int64_t{0}; i < src_info.size<0>(); ++i) {
        (*unpack_ptr)(src_info.data + i * src_info.stride<0>(), number_spins,
                      dst_info.data + i * dst_info.stride<0>());
    }
}
#endif

TCM_NAMESPACE_END
