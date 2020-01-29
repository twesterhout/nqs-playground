#include "../config.hpp"
#include <torch/types.h>

TCM_NAMESPACE_BEGIN
namespace gpu {

auto unpack_cuda(torch::Tensor spins, int64_t number_spins, torch::Tensor out) -> void;

} // namespace gpu
TCM_NAMESPACE_END
