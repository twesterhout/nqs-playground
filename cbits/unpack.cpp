#include "unpack.hpp"
#include "errors.hpp"

#include "cpu/kernels.hpp"
#if defined(TCM_USE_CUDA)
#    include "gpu/unpack.hpp"
#endif

#include <torch/script.h>

TCM_NAMESPACE_BEGIN

template <class Bits>
auto unpack_impl(TensorInfo<Bits const> const& spins,
                 int64_t const number_spins, c10::Device const device)
    -> torch::Tensor
{
    TCM_CHECK(
        number_spins > 0, std::domain_error,
        fmt::format("invalid number_spins: {}; expected a positive integer",
                    number_spins));
    auto out = torch::empty(
        std::initializer_list<int64_t>{spins.size(), number_spins},
        torch::TensorOptions{}.device(device).dtype(torch::kFloat32));
    auto out_info =
        TensorInfo<float, 2>{static_cast<float*>(out.data_ptr()),
                             out.sizes().data(), out.strides().data()};
    switch (device.type()) {
    case c10::DeviceType::CPU: unpack_cpu(spins, out_info); break;
#if defined(TCM_USE_CUDA)
    case c10::DeviceType::CUDA:
        gpu::unpack_cuda(spins, out_info, device);
        break;
#endif
    default: {
#if defined(TCM_USE_CUDA)
        auto const error_msg = "spins resides on an unsupported device: {}; "
                               "expected either CPU or CUDA";
#else
        auto const error_msg =
            "spins resides on an unsupported device: {}; expected CPU";
#endif
        TCM_ERROR(std::domain_error,
                  fmt::format(error_msg, c10::DeviceTypeName(device.type())));
    }
    } // end switch
    return out;
}

TCM_EXPORT auto unpack(torch::Tensor spins, int64_t const number_spins)
    -> torch::Tensor
{
    auto const shape  = spins.sizes();
    auto const device = spins.device();
    switch (shape.size()) {
    case 1:
        return unpack_impl(obtain_tensor_info<uint64_t const>(spins, "spins"),
                           number_spins, device);
    case 2:
        return unpack_impl(obtain_tensor_info<bits512 const>(spins, "spins"),
                           number_spins, device);
    default:
        TCM_ERROR(std::domain_error,
                  fmt::format("spins has wrong shape: [{}]; expected either a "
                              "one- or two-dimensional tensor",
                              fmt::join(shape, ", ")));
    }
}

static auto torch_script_operators = torch::RegisterOperators{}.op(
    "tcm::unpack",
    torch::RegisterOperators::options()
        .catchAllKernel<auto(torch::Tensor, int64_t)->torch::Tensor,
                        static_cast<auto (*)(torch::Tensor, int64_t)
                                        ->torch::Tensor>(&unpack)>());

TCM_NAMESPACE_END
