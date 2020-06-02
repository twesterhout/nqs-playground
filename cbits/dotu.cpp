#include "dotu.hpp"
#include "common.hpp"
#include "errors.hpp"

#include "cpu/dotu.hpp"
#if defined(TCM_USE_CUDA)
#    include "gpu/dotu.hpp"
#endif

TCM_NAMESPACE_BEGIN

inline auto dotu_impl(TensorInfo<std::complex<float> const> const& x,
                      TensorInfo<std::complex<float> const> const& y,
                      c10::Device const device) -> std::complex<double>
{
    if (x.size() == 0) { return std::complex<double>{0.0, 0.0}; }
    switch (device.type()) {
    case c10::DeviceType::CPU: return cpu::dotu_cpu(x, y);
#if defined(TCM_USE_CUDA)
    case c10::DeviceType::CUDA: return gpu::dotu_gpu(x, y, device);
#endif
    default: {
#if defined(TCM_USE_CUDA)
        auto const error_msg = "x & y reside on an unsupported device: {}; "
                               "expected either CPU or CUDA";
#else
        auto const error_msg =
            "x & y reside on an unsupported device: {}; expected CPU";
#endif
        TCM_ERROR(std::domain_error,
                  fmt::format(error_msg, c10::DeviceTypeName(device.type())));
    }
    } // end switch
}

TCM_EXPORT auto dotu(torch::Tensor x, torch::Tensor y) -> std::complex<double>
{
    auto const device = x.device();
    TCM_CHECK(device == y.device(), std::invalid_argument,
              fmt::format("x and y reside on different devices: {} and {}",
                          c10::DeviceTypeName(device.type()),
                          c10::DeviceTypeName(y.device().type())));
    auto const x_info = obtain_tensor_info<std::complex<float> const>(x, "x");
    auto const y_info = obtain_tensor_info<std::complex<float> const>(y, "y");
    TCM_CHECK(x_info.size() == y_info.size(), std::invalid_argument,
        fmt::format("x and y have different shapes: [{}] != [{}]", x_info.size(), y_info.size()));
    constexpr auto max_size = std::numeric_limits<int>::max();
    TCM_CHECK(x_info.size() <= max_size, std::invalid_argument,
        fmt::format("x is too long: {}; expected <={}", x_info.size(), max_size));
    TCM_CHECK(x_info.stride() <= max_size, std::invalid_argument,
        fmt::format("stride of x is too big: {}; expected <={}", x_info.stride(), max_size));
    TCM_CHECK(y_info.stride() <= max_size, std::invalid_argument,
        fmt::format("stride of y is too big: {}; expected <={}", y_info.stride(), max_size));
    return dotu_impl(x_info, y_info, device);
}

TCM_NAMESPACE_END
