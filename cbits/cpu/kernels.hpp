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

template <class T>
auto jump_rates_one_avx2(TensorInfo<T> const&, TensorInfo<T const> const&,
                         T) noexcept -> T;
template <class T>
auto jump_rates_one_avx(TensorInfo<T> const&, TensorInfo<T const> const&,
                        T) noexcept -> T;
template <class T>
auto jump_rates_one_sse2(TensorInfo<T> const&, TensorInfo<T const> const&,
                         T) noexcept -> T;

template <class Bits>
auto unpack_cpu(TensorInfo<Bits const> const& spins,
                TensorInfo<float, 2> const&   out) -> void;

auto unpack_one_avx2(bits512 const&, unsigned, float*) noexcept -> void;
auto unpack_one_avx2(uint64_t, unsigned, float*) noexcept -> void;
auto unpack_one_avx(bits512 const&, unsigned, float*) noexcept -> void;
auto unpack_one_avx(uint64_t, unsigned, float*) noexcept -> void;
auto unpack_one_sse2(bits512 const&, unsigned, float*) noexcept -> void;
auto unpack_one_sse2(uint64_t, unsigned, float*) noexcept -> void;

TCM_NAMESPACE_END
