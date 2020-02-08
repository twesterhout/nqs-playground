#include "polynomial_state.hpp"

TCM_NAMESPACE_BEGIN

TCM_EXPORT auto apply(torch::Tensor spins, Heisenberg const& hamiltonian,
                      v2::ForwardT psi) -> torch::Tensor
{
    torch::NoGradGuard no_grad;
    constexpr auto     batch_size = 1024U;

    TCM_CHECK(spins.dim() == 1, std::domain_error,
              fmt::format("spins has wrong shape: [{}]; expected a 1D tensor",
                          fmt::join(spins.sizes(), ", ")));
    TCM_CHECK_TYPE("spins", spins, torch::kInt64);
    TCM_CHECK_CONTIGUOUS("spins", spins);
    auto const device = spins.device();
    spins = spins.to(spins.options().device(torch::DeviceType::CPU));
    auto states = gsl::span<uint64_t const>{
        static_cast<uint64_t const*>(spins.data_ptr()),
        static_cast<size_t>(spins.numel())};
    auto buffer = torch::empty({static_cast<int64_t>(states.size()), 2},
                               torch::TensorOptions{}.dtype(torch::kFloat32));
    auto out    = gsl::span<std::complex<float>>{
        static_cast<std::complex<float>*>(buffer.data_ptr()), states.size()};

    detail::Accumulator acc{std::move(psi), out, batch_size, device};
    for (auto const x : states) {
        acc([&hamiltonian, x](auto&& f) {
            hamiltonian(x, std::forward<decltype(f)>(f));
        });
    }
    acc.finalize();

    return buffer;
}

TCM_EXPORT auto diag(torch::Tensor spins, Heisenberg const& hamiltonian)
    -> torch::Tensor
{
    TCM_CHECK(spins.dim() == 1, std::domain_error,
              fmt::format("spins has wrong shape: [{}]; expected a 1D tensor",
                          fmt::join(spins.sizes(), ", ")));
    TCM_CHECK_TYPE("spins", spins, torch::kInt64);
    auto const device = spins.device();
    spins = spins.to(spins.options().device(torch::DeviceType::CPU));

    auto const spins_data = static_cast<uint64_t const*>(spins.data_ptr());
    auto const spins_size = spins.numel();
    auto const spins_stride = spins.stride(0);

    auto out = torch::empty({spins_size, 2},
        torch::TensorOptions{}
            .dtype(torch::kFloat32)
            .pinned_memory(device.type() == torch::DeviceType::CUDA));
    auto const out_data = static_cast<std::complex<float>*>(out.data_ptr());

    for (auto i = int64_t{0}; i < spins_size; ++i) {
        auto const spin = spins_data[i * spins_stride];
        out_data[i] = static_cast<std::complex<float>>(hamiltonian.diag(spin));
    }
    return out.to(out.options().device(device), /*non_blocking=*/true, /*copy=*/false);
}

TCM_NAMESPACE_END
