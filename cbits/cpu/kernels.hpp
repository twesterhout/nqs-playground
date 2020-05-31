#include "../config.hpp"
#include <torch/types.h>

TCM_NAMESPACE_BEGIN

auto zanella_jump_rates(torch::Tensor               current_log_prob,
                        torch::Tensor               proposed_log_prob,
                        std::vector<int64_t> const& counts)
    -> std::tuple<torch::Tensor, torch::Tensor>;

// Specific implementations
auto zanella_jump_rates_sse2(torch::Tensor               current_log_prob,
                             torch::Tensor               proposed_log_prob,
                             std::vector<int64_t> const& counts)
    -> std::tuple<torch::Tensor, torch::Tensor>;
auto zanella_jump_rates_avx(torch::Tensor               current_log_prob,
                            torch::Tensor               proposed_log_prob,
                            std::vector<int64_t> const& counts)
    -> std::tuple<torch::Tensor, torch::Tensor>;
auto zanella_jump_rates_avx2(torch::Tensor               current_log_prob,
                             torch::Tensor               proposed_log_prob,
                             std::vector<int64_t> const& counts)
    -> std::tuple<torch::Tensor, torch::Tensor>;
auto zanella_jump_rates_avx512(torch::Tensor               current_log_prob,
                               torch::Tensor               proposed_log_prob,
                               std::vector<int64_t> const& counts)
    -> std::tuple<torch::Tensor, torch::Tensor>;

TCM_NAMESPACE_END
