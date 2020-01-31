#include "../config.hpp"
#include <torch/types.h>

TCM_NAMESPACE_BEGIN
namespace cpu {

TCM_IMPORT auto unpack_cpu_generic(torch::Tensor spins, int64_t number_spins,
                                   torch::Tensor out) -> void;
TCM_IMPORT auto unpack_cpu_avx(torch::Tensor spins, int64_t number_spins,
                               torch::Tensor out) -> void;

TCM_IMPORT auto unpack_cpu(torch::Tensor spins, torch::Tensor out) -> void;

} // namespace cpu
TCM_NAMESPACE_END
