#include "unpack.hpp"
#include "../bits512.hpp"
#include "../errors.hpp"
#include "../tensor_info.hpp"
#include <caffe2/utils/cpuid.h>

TCM_NAMESPACE_BEGIN
namespace cpu {

namespace detail {
    struct _IdentityProjection {
        template <class T>
        constexpr decltype(auto) operator()(T&& x) const noexcept
        {
            return std::forward<T>(x);
        }
    };

    TCM_FORCEINLINE auto unpack_word(uint64_t bits, float* const out) noexcept
        -> void
    {
        for (auto i = 0; i < 64; ++i, bits >>= 1) {
            out[i] = 2.0f * static_cast<float>(bits & 0x01) - 1.0f;
        }
    }

    TCM_FORCEINLINE auto unpack_rest(uint64_t bits, uint32_t const count,
                                     float* const out) noexcept -> void
    {
        for (auto i = 0U; i < count; ++i, bits >>= 1) {
            out[i] = 2.0f * static_cast<float>(bits & 0x01) - 1.0f;
        }
    }

} // namespace detail

TCM_FORCEINLINE auto unpack_one(uint64_t const bits, uint32_t const count,
                                float* const out) noexcept -> void
{
    return detail::unpack_rest(bits, count, out);
}

TCM_FORCEINLINE auto unpack_one(bits512 const& bits, uint32_t const count,
                                float* out) noexcept -> void
{
    constexpr auto block = 64U;

    auto i = 0U;
    for (; i < count / block; ++i) {
        detail::unpack_word(bits.words[i], out);
        out += block;
    }
    if (auto const rest = count % block; rest != 0) {
        detail::unpack_rest(bits.words[i], rest, out);
    }
}

template <class Bits, class Projection = detail::_IdentityProjection>
auto unpack_cpu_generic(TensorInfo<Bits const> const& src_info,
                        TensorInfo<float, 2> const&   dst_info,
                        Projection proj = Projection{}) -> void
{
    if (TCM_UNLIKELY(src_info.size() == 0)) { return; }
    auto const* src          = src_info.data;
    auto*       dst          = dst_info.data;
    auto const  number_spins = static_cast<unsigned>(dst_info.sizes[1]);
    for (auto i = int64_t{0}; i < src_info.size();
         ++i, src += src_info.stride(), dst += dst_info.strides[0]) {
        unpack_one(proj(*src), number_spins, dst);
    }
}

#if 0
template <class Iterator, class Sentinel,
          class Projection = detail::_IdentityProjection>
auto unpack_cpu_generic(Iterator first, Sentinel last,
                        unsigned const number_spins, torch::Tensor& dst,
                        Projection proj = Projection{}) -> void
{
    if (first == last) { return; }
    auto const size = static_cast<size_t>(std::distance(first, last));
    auto*      data = dst.data_ptr<float>();
    for (auto i = size_t{0}; i < size; ++i, ++first, data += number_spins) {
        detail::unpack_one(proj(*first), number_spins, data);
    }
}

TCM_EXPORT auto unpack_cpu_generic(torch::Tensor spins,
                                   int64_t const number_spins,
                                   torch::Tensor out) -> void
{
    auto const first = static_cast<uint64_t const*>(spins.data_ptr());
    auto const last  = first + spins.size(0);
    unpack_cpu_generic(first, last, static_cast<unsigned>(number_spins), out);
}
#endif

#define DEFINE_INTERFACE_FN(Type)                                              \
    TCM_EXPORT auto unpack_cpu(TensorInfo<Type const> const& spins,            \
                               TensorInfo<float, 2> const&   out)              \
        ->void                                                                 \
    {                                                                          \
        if (caffe2::GetCpuId().avx()) { unpack_cpu_avx(spins, out); }          \
        else {                                                                 \
            unpack_cpu_generic(spins, out);                                    \
        }                                                                      \
    }

DEFINE_INTERFACE_FN(uint64_t)
DEFINE_INTERFACE_FN(bits512)

#if 0
TCM_EXPORT auto unpack_cpu(torch::Tensor spins, torch::Tensor out) -> void
{
    TCM_CHECK_CONTIGUOUS("spins", spins);
    TCM_CHECK_CONTIGUOUS("out", out);
    auto const number_spins = out.size(1);
    if (caffe2::GetCpuId().avx()) {
        unpack_cpu_avx(std::move(spins), number_spins, std::move(out));
    }
    else {
        unpack_cpu_generic(std::move(spins), number_spins, std::move(out));
    }
}
#endif

#if 0
TCM_EXPORT auto unpack(torch::Tensor x, torch::Tensor indices,
                       int64_t const number_spins) -> torch::Tensor
{
    TCM_CHECK(
        number_spins > 0, std::invalid_argument,
        fmt::format("invalid number_spins: {}; expected a positive integer",
                    number_spins));
    TCM_CHECK_TYPE("x", x, torch::kInt64);
    TCM_CHECK_TYPE("indices", indices, torch::kInt64);
    TCM_CHECK_CONTIGUOUS("x", x);

    auto x_shape = x.sizes();
    TCM_CHECK(x_shape.size() == 1 || (x_shape.size() == 2 && x_shape[1] == 1),
              std::domain_error,
              fmt::format("x has wrong shape: [{}]; expected a vector",
                          fmt::join(x_shape, ", ")));
    auto indices_shape = indices.sizes();
    TCM_CHECK(indices_shape.size() == 1
                  || (indices_shape.size() == 2 && indices_shape[1] == 1),
              std::domain_error,
              fmt::format("indices has wrong shape: [{}]; expected a vector",
                          fmt::join(indices_shape, ", ")));

    auto out = torch::empty(
        std::initializer_list<int64_t>{indices_shape[0], number_spins},
        torch::TensorOptions{}.dtype(torch::kFloat32));
    auto const first = static_cast<int64_t const*>(indices.data_ptr());
    auto const last  = first + indices_shape[0];
    unpack(first, last, static_cast<unsigned>(number_spins), out,
           [data = static_cast<uint64_t const*>(x.data_ptr()),
            n    = x_shape[0]](auto const i) {
               TCM_CHECK(
                   0 <= i && i < n, std::out_of_range,
                   fmt::format("indices contains an index which is out of "
                               "range: {}; expected an index in [{}, {})",
                               i, 0, n));
               return data[i];
           });
    return out;
}
#endif

} // namespace cpu
TCM_NAMESPACE_END
