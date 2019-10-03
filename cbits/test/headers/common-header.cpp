#include "../../common.hpp"

template auto tcm::detail::make_tensor<float, size_t, size_t>(size_t, size_t) -> torch::Tensor;

auto main() -> int { return 0; }
