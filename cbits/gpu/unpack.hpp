#include "../comon/bits512.hpp"
#include "../comon/tensor_info.hpp"
#include <torch/types.h>

TCM_NAMESPACE_BEGIN

// auto unpack_cuda(torch::Tensor spins, torch::Tensor out) -> void;
auto unpack_cuda(TensorInfo<uint64_t const> const& spins, TensorInfo<float, 2> const& out,
                 c10::Device device) -> void;
auto unpack_cuda(TensorInfo<bits512 const> const& spins, TensorInfo<float, 2> const& out,
                 c10::Device device) -> void;

TCM_NAMESPACE_END
