#include "unpack.hpp"
#include "errors.hpp"

#include "cpu/unpack.hpp"
#if defined(TCM_USE_CUDA)
#    include "gpu/unpack.hpp"
#endif

#include <torch/script.h>

TCM_NAMESPACE_BEGIN

TCM_EXPORT auto unpack(torch::Tensor spins, int64_t const number_spins)
    -> torch::Tensor
{
    TCM_CHECK(
        number_spins > 0, std::domain_error,
        fmt::format("invalid number_spins: {}; expected a positive integer",
                    number_spins));
    TCM_CHECK_TYPE("spins", spins, torch::kInt64);
    auto shape = spins.sizes();
    TCM_CHECK(shape.size() == 1 || (shape.size() == 2 && shape[1] == 1),
              std::domain_error,
              fmt::format("spins has wrong shape: [{}]; expected a vector",
                          fmt::join(shape, ", ")));
	TCM_CHECK(shape[0] * spins.stride(0) < std::numeric_limits<int32_t>::max(),
	    std::domain_error, fmt::format("spins tensor is too big"));
	auto device = spins.device();
    auto out =
        torch::empty(std::initializer_list<int64_t>{shape[0], number_spins},
                     torch::TensorOptions{}.device(device).dtype(torch::kFloat32));
	switch (device.type()) {
	case c10::DeviceType::CPU:
		cpu::unpack_cpu(spins, number_spins, out);
	    break;
#if defined(TCM_USE_CUDA)
	case c10::DeviceType::CUDA:
		gpu::unpack_cuda(spins, number_spins, out);
	    break;
#endif
	default: {
#if defined(TCM_USE_CUDA)
		auto const error_msg = "spins resides on an unsupported device: {}; expected either CPU or CUDA";
#else
		auto const error_msg = "spins resides on an unsupported device: {}; expected CPU";
#endif
		TCM_ERROR(std::domain_error, fmt::format(error_msg, c10::DeviceTypeName(device.type())));
	}
	} // end switch
    return out;
}

static auto torch_script_operators =
    torch::RegisterOperators{}
        .op("tcm::unpack",
            torch::RegisterOperators::options()
                .catchAllKernel<auto(torch::Tensor, int64_t)->torch::Tensor,
                        static_cast<auto (*)(torch::Tensor, int64_t)
                                        ->torch::Tensor>(&unpack)>());

TCM_NAMESPACE_END
