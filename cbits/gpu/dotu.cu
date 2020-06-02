#include "dotu.hpp"
#include <ATen/cuda/CUDAContext.h>

TCM_NAMESPACE_BEGIN
namespace gpu {

TCM_EXPORT auto dotu_gpu(TensorInfo<std::complex<float> const> const& x,
                         TensorInfo<std::complex<float> const> const& y,
                         c10::Device const device) -> std::complex<double>
{
    std::complex<float> result;
    cudaSetDevice(device.index());
    auto const handle = at::cuda::getCurrentCUDABlasHandle();
    auto const status = cublasCdotu(handle, x.size(),
                                    reinterpret_cast<cuComplex const*>(x.data), x.stride(),
                                    reinterpret_cast<cuComplex const*>(y.data), y.stride(),
                                    reinterpret_cast<cuComplex*>(&result));
    TCM_CHECK(status == CUBLAS_STATUS_SUCCESS, std::runtime_error, "CUBLAS Error");
    return result;
}

} // namespace gpu
TCM_NAMESPACE_END

