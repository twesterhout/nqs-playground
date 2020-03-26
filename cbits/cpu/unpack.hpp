#include "../bits512.hpp"
#include "../tensor_info.hpp"
#include <torch/types.h>

TCM_NAMESPACE_BEGIN
namespace cpu {

template <class Bits>
TCM_EXPORT auto unpack_cpu_avx(TensorInfo<Bits const> const& spins,
                               TensorInfo<float, 2> const&   out) -> void;

TCM_EXPORT auto unpack_cpu(TensorInfo<uint64_t const> const& spins,
                           TensorInfo<float, 2> const&       out) -> void;

TCM_EXPORT auto unpack_cpu(TensorInfo<bits512 const> const& spins,
                           TensorInfo<float, 2> const&      out) -> void;

} // namespace cpu
TCM_NAMESPACE_END
