#include "../bits512.hpp"
#include "../tensor_info.hpp"
#include <torch/types.h>

TCM_NAMESPACE_BEGIN

auto dotu_cpu(TensorInfo<std::complex<float> const> const& x,
              TensorInfo<std::complex<float> const> const& y)
    -> std::complex<double>;

auto zanella_jump_rates(torch::Tensor               current_log_prob,
                        torch::Tensor               proposed_log_prob,
                        std::vector<int64_t> const& counts)
    -> std::tuple<torch::Tensor, torch::Tensor>;

auto unpack_cpu(TensorInfo<uint64_t const> const& spins,
                TensorInfo<float, 2> const&       out) -> void;

auto unpack_cpu(TensorInfo<bits512 const> const& spins,
                TensorInfo<float, 2> const&      out) -> void;

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

template <class Bits>
auto unpack_cpu_sse2(TensorInfo<Bits const> const& spins,
                     TensorInfo<float, 2> const&   out) -> void;

template <class Bits>
auto unpack_cpu_avx(TensorInfo<Bits const> const& spins,
                    TensorInfo<float, 2> const&   out) -> void;

template <class Bits>
auto unpack_cpu_avx2(TensorInfo<Bits const> const& spins,
                     TensorInfo<float, 2> const&   out) -> void;

TCM_NAMESPACE_END
