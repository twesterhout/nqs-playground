#include "../tensor_info.hpp"
#include <torch/types.h>

TCM_NAMESPACE_BEGIN
namespace gpu {

TCM_EXPORT auto dotu_gpu(TensorInfo<std::complex<float> const> const& x,
                         TensorInfo<std::complex<float> const> const& y,
                         c10::Device device) -> std::complex<double>;

} // namespace gpu
TCM_NAMESPACE_END
