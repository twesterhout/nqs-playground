#include "../config.hpp"
#include <torch/types.h>

TCM_NAMESPACE_BEGIN
namespace cpu {

auto unpack_cpu(torch::Tensor spins, int64_t number_spins, torch::Tensor out) -> void;

} // namespace cpu
TCM_NAMESPACE_END
