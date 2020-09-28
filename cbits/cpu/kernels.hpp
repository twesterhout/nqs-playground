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

template <class scalar_t>
auto tabu_jump_rates(gsl::span<scalar_t const> proposed_log_prob,
                     scalar_t current_log_prob) -> std::vector<scalar_t>;

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

auto bfly(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;
auto bfly_avx(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;
auto bfly_sse2(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;

auto ibfly(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;
auto ibfly_avx(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;
auto ibfly_sse2(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;

auto bfly(uint64_t x, uint64_t out[8], uint64_t const (*masks)[8]) noexcept
    -> void;
auto bfly_avx(uint64_t x, uint64_t out[8], uint64_t const (*masks)[8]) noexcept
    -> void;
auto bfly_sse2(uint64_t x, uint64_t out[8], uint64_t const (*masks)[8]) noexcept
    -> void;

// These are not really used anywhere
auto bfly_avx2(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;
auto bfly_avx2(uint64_t x, uint64_t out[8], uint64_t const (*masks)[8]) noexcept
    -> void;
auto ibfly_avx2(uint64_t x[8], uint64_t const (*masks)[8]) noexcept -> void;

TCM_NAMESPACE_END
