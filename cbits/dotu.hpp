#pragma once

#include "config.hpp"
#include <torch/types.h>

TCM_NAMESPACE_BEGIN

TCM_IMPORT auto dotu(torch::Tensor, torch::Tensor) -> std::complex<double>;

TCM_NAMESPACE_END
