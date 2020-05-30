#include "dotu.hpp"
#include <ATen/Config.h>

#define MKL_INT int
extern "C" void cblas_cdotu_sub(const MKL_INT n, const void* x,
                                const MKL_INT incx, const void* y,
                                const MKL_INT incy, void* dotu);

TCM_NAMESPACE_BEGIN
namespace cpu {
#if !AT_MKL_ENABLED()

TCM_EXPORT auto dotu_cpu(TensorInfo<std::complex<float> const> const& x,
                         TensorInfo<std::complex<float> const> const& y)
    -> std::complex<double>
{
    TCM_ERROR(std::runtime_error, "PyTorch is compiled without MKL support");
}

#else // AT_MKL_ENABLED

TCM_EXPORT auto dotu_cpu(TensorInfo<std::complex<float> const> const& x,
                         TensorInfo<std::complex<float> const> const& y)
    -> std::complex<double>
{
    std::complex<float> result;
    cblas_cdotu_sub(x.size(), x.data, x.stride(), y.data, y.stride(), &result);
    return static_cast<std::complex<double>>(result);
}

#endif
} // namespace cpu
TCM_NAMESPACE_END
