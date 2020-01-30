#include "unpack.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>

TCM_NAMESPACE_BEGIN
namespace gpu {

namespace detail {

    struct SpinsInfo {
        uint64_t const* const data;
        int32_t const         stride;
    };

    struct OutInfo {
        float* const  data;
        int32_t const shape[2];
        int32_t const stride[2];
    };

    __device__ inline auto unpack_one(uint64_t bits, int32_t const count,
                                      float* const  out,
                                      int32_t const stride) noexcept -> void
    {
        for (auto i = 0; i < count; ++i, bits >>= 1) {
            out[i * stride] = 2.0f * static_cast<float>(bits & 0x01) - 1.0f;
        }
    }

    __global__ auto unpack_kernel_cuda(SpinsInfo spins, OutInfo out) -> void
    {
        auto const idx    = blockIdx.x * blockDim.x + threadIdx.x;
        auto const stride = blockDim.x * gridDim.x;
        for (auto i = idx; i < out.shape[0]; i += stride) {
            unpack_one(spins.data[i * spins.stride], out.shape[1],
                       out.data + i * out.stride[0], out.stride[1]);
        }
    }
} // namespace detail

auto unpack_cuda(torch::Tensor spins, int64_t const number_spins,
                 torch::Tensor out) -> void
{
    // clang-format off
    cudaSetDevice(spins.get_device());
    auto spins_info = detail::SpinsInfo{
        static_cast<uint64_t const*>(spins.data_ptr()),
        static_cast<int32_t>(spins.stride(0))};
    auto out_info = detail::OutInfo{
        static_cast<float*>(out.data_ptr()),
        {static_cast<int32_t>(out.size(0)), static_cast<int32_t>(out.size(1))},
        {static_cast<int32_t>(out.stride(0)), static_cast<int32_t>(out.stride(1))}};
    auto stream = at::cuda::getCurrentCUDAStream();
    detail::unpack_kernel_cuda<<<at::cuda::detail::GET_BLOCKS(out_info.shape[0]),
        at::cuda::detail::CUDA_NUM_THREADS, 0, stream>>>(spins_info, out_info);
    // clang-format on
}

} // namespace gpu
TCM_NAMESPACE_END
