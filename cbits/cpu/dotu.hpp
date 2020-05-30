#include "../tensor_info.hpp"
#include <torch/types.h>

TCM_NAMESPACE_BEGIN
namespace cpu {

TCM_EXPORT auto dotu_cpu(TensorInfo<std::complex<float> const> const& x,
                         TensorInfo<std::complex<float> const> const& y)
    -> std::complex<double>;

} // namespace cpu
TCM_NAMESPACE_END
