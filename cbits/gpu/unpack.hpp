#include "../common/bits512.hpp"
#include "../common/tensor_info.hpp"
#include <torch/types.h>

TCM_NAMESPACE_BEGIN

auto unpack_cuda(TensorInfo<uint64_t const, 2> const& spins, TensorInfo<float, 2> const& out,
                 c10::Device device) -> void;

TCM_NAMESPACE_END
