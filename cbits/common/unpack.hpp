#pragma once

#include "config.hpp"
#include <torch/types.h>

TCM_NAMESPACE_BEGIN

auto unpack(torch::Tensor, int64_t) -> torch::Tensor;
auto hamming_weight(torch::Tensor, int64_t) -> torch::Tensor;

TCM_NAMESPACE_END
