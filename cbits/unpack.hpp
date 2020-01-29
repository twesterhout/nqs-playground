#pragma once

#include "config.hpp"
#include <torch/types.h>

TCM_NAMESPACE_BEGIN

TCM_IMPORT auto unpack(torch::Tensor, int64_t) -> torch::Tensor;
TCM_IMPORT auto unpack(torch::Tensor, torch::Tensor, int64_t) -> torch::Tensor;

TCM_NAMESPACE_END
