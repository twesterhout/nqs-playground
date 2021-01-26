#include "unpack.hpp"
#include "errors.hpp"

#include "../cpu/kernels.hpp"
#if defined(TCM_USE_CUDA)
#    include "../gpu/unpack.hpp"
#endif

#include <torch/script.h>

TCM_NAMESPACE_BEGIN

namespace {
auto unpack_impl(TensorInfo<uint64_t const, 2> const& spins, int64_t const number_spins,
                 c10::Device const device) -> torch::Tensor
{
    TCM_CHECK(number_spins > 0, std::domain_error,
              fmt::format("invalid number_spins: {}; expected a positive integer", number_spins));
    TCM_CHECK(number_spins <= 64 * spins.size<1>(), std::domain_error,
              fmt::format("number_spins too big: {}; expected <= {}", number_spins,
                          64 * spins.size<1>()));
    auto out      = torch::empty(std::initializer_list<int64_t>{spins.size<0>(), number_spins},
                            torch::TensorOptions{}.device(device).dtype(torch::kFloat32));
    auto out_info = tensor_info<float, 2>(out);
    switch (device.type()) {
    case c10::DeviceType::CPU: unpack_cpu(spins, out_info); break;
#if defined(TCM_USE_CUDA)
    case c10::DeviceType::CUDA: unpack_cuda(spins, out_info, device); break;
#endif
    default: {
#if defined(TCM_USE_CUDA)
        auto const error_msg = "'spins' tensor resides on an unsupported "
                               "device: {}; expected either CPU or CUDA";
#else
        auto const error_msg = "'spins' tensor resides on an unsupported device: {}; expected CPU";
#endif
        TCM_ERROR(std::domain_error, fmt::format(error_msg, c10::DeviceTypeName(device.type())));
    }
    } // end switch
    return out;
}

auto hamming_weight_impl(TensorInfo<uint64_t const, 2> const& spins, int64_t const number_spins,
                         c10::Device const device) -> torch::Tensor
{
    TCM_CHECK(number_spins > 0, std::domain_error,
              fmt::format("invalid number_spins: {}; expected a positive integer", number_spins));
    TCM_CHECK(number_spins <= 64 * spins.size<1>(), std::domain_error,
              fmt::format("number_spins too big: {}; expected <= {}", number_spins,
                          64 * spins.size<1>()));
    auto out      = torch::empty(std::initializer_list<int64_t>{spins.size<0>()},
                            torch::TensorOptions{}.device(device).dtype(torch::kFloat32));
    auto out_info = tensor_info<float>(out);
    switch (device.type()) {
    case c10::DeviceType::CPU: hamming_weight_cpu(spins, out_info); break;
#if defined(TCM_USE_CUDA)
    case c10::DeviceType::CUDA:
        TCM_ERROR(std::domain_error, "hamming_weight() is not yet implemented for CUDA");
#endif
    default: {
#if defined(TCM_USE_CUDA)
        auto const error_msg = "'spins' tensor resides on an unsupported "
                               "device: {}; expected either CPU or CUDA";
#else
        auto const error_msg = "'spins' tensor resides on an unsupported device: {}; expected CPU";
#endif
        TCM_ERROR(std::domain_error, fmt::format(error_msg, c10::DeviceTypeName(device.type())));
    }
    } // end switch
    return out;
}

} // namespace

TCM_EXPORT auto unpack(torch::Tensor spins, int64_t const number_spins) -> torch::Tensor
{
    auto const shape  = spins.sizes();
    auto const device = spins.device();
    switch (shape.size()) {
    case 1: spins = torch::unsqueeze(spins, /*dim=*/1); TCM_FALLTHROUGH;
    case 2:
        return unpack_impl(tensor_info<uint64_t const, 2>(spins, "spins"), number_spins, device);
    default:
        TCM_ERROR(std::domain_error, fmt::format("spins has wrong shape: [{}]; expected either a "
                                                 "one- or two-dimensional tensor",
                                                 fmt::join(shape, ", ")));
    } // end switch
}

TCM_EXPORT auto hamming_weight(torch::Tensor spins, int64_t const number_spins) -> torch::Tensor
{
    auto const shape  = spins.sizes();
    auto const device = spins.device();
    switch (shape.size()) {
    case 1: spins = torch::unsqueeze(spins, /*dim=*/1); TCM_FALLTHROUGH;
    case 2:
        return hamming_weight_impl(tensor_info<uint64_t const, 2>(spins, "spins"), number_spins,
                                   device);
    default:
        TCM_ERROR(std::domain_error, fmt::format("spins has wrong shape: [{}]; expected either a "
                                                 "one- or two-dimensional tensor",
                                                 fmt::join(shape, ", ")));
    } // end switch
}

static auto torch_script_operators =
    torch::RegisterOperators{}
        .op("tcm::unpack",
            torch::RegisterOperators::options()
                .catchAllKernel<auto(torch::Tensor, int64_t)->torch::Tensor, &unpack>())
        .op("tcm::hamming_weight",
            torch::RegisterOperators::options()
                .catchAllKernel<auto(torch::Tensor, int64_t)->torch::Tensor, &hamming_weight>());

TCM_NAMESPACE_END
