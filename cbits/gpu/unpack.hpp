#include "../bits512.hpp"
#include "../tensor_info.hpp"
#include <torch/types.h>

TCM_NAMESPACE_BEGIN
namespace gpu {

// auto unpack_cuda(torch::Tensor spins, torch::Tensor out) -> void;
auto unpack_cuda(TensorInfo<uint64_t const> const& spins,
                 TensorInfo<float, 2> const& out,
                 c10::Device device) -> void;
auto unpack_cuda(TensorInfo<bits512 const> const& spins,
                 TensorInfo<float, 2> const& out,
                 c10::Device device) -> void;

} // namespace gpu
TCM_NAMESPACE_END
