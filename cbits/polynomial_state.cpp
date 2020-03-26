#include "polynomial_state.hpp"
#include "tensor_info.hpp"

TCM_NAMESPACE_BEGIN

TCM_EXPORT auto apply(torch::Tensor spins, Heisenberg const& hamiltonian,
                      v2::ForwardT psi) -> torch::Tensor
{
    torch::NoGradGuard no_grad;
    constexpr auto     batch_size = 1024U;

    auto const device = spins.device();
    spins = spins.to(spins.options().device(torch::DeviceType::CPU));
    auto const spins_info = obtain_tensor_info<bits512 const>(spins);

    auto buffer =
        torch::empty(std::initializer_list<int64_t>{spins_info.size(), 2L},
                     torch::TensorOptions{}.dtype(torch::kFloat32));
    auto out = gsl::span<std::complex<float>>{
        static_cast<std::complex<float>*>(buffer.data_ptr()),
        static_cast<size_t>(spins_info.size())};

    detail::Accumulator acc{std::move(psi), out, batch_size, device};
    for (auto i = int64_t{0}; i < spins_info.size(); ++i) {
        auto const& x = spins_info.data[i * spins_info.stride()];
        acc([&hamiltonian, &x](auto&& f) {
            hamiltonian(x, std::forward<decltype(f)>(f));
        });
    }
    acc.finalize();

    return buffer;
}

TCM_EXPORT auto diag(torch::Tensor spins, Heisenberg const& hamiltonian)
    -> torch::Tensor
{
    torch::NoGradGuard no_grad;
    auto const         device = spins.device();
    spins = spins.to(spins.options().device(torch::DeviceType::CPU));
    auto const spins_info = obtain_tensor_info<bits512 const>(spins);
    auto const out        = torch::empty(
        {spins_info.size(), 2},
        torch::TensorOptions{}
            .dtype(torch::kFloat32)
            .pinned_memory(device.type() == torch::DeviceType::CUDA));
    auto const out_data = static_cast<std::complex<float>*>(out.data_ptr());
    for (auto i = int64_t{0}; i < spins_info.size(); ++i) {
        auto const& spin = spins_info.data[i * spins_info.stride()];
        out_data[i] = static_cast<std::complex<float>>(hamiltonian.diag(spin));
    }
    return out.to(out.options().device(device), /*non_blocking=*/true,
                  /*copy=*/false);
}

TCM_IMPORT auto apply(torch::Tensor spins, Polynomial<Heisenberg>& polynomial,
                      v2::ForwardT psi) -> torch::Tensor
{
    torch::NoGradGuard no_grad;
    constexpr auto     batch_size = 1024U;

    auto const device = spins.device();
    spins = spins.to(spins.options().device(torch::DeviceType::CPU));
    auto const spins_info = obtain_tensor_info<bits512 const>(spins);

    auto buffer =
        torch::empty(std::initializer_list<int64_t>{spins_info.size(), 2L},
                     torch::TensorOptions{}.dtype(torch::kFloat32));
    auto out = gsl::span<std::complex<float>>{
        static_cast<std::complex<float>*>(buffer.data_ptr()),
        static_cast<size_t>(spins_info.size())};

    detail::Accumulator acc{std::move(psi), out, batch_size, device};
    for (auto i = int64_t{0}; i < spins_info.size(); ++i) {
        auto const& state =
            polynomial(spins_info.data[i * spins_info.stride()]);
        acc([&state](auto&& f) {
            for (auto const& item : state) {
                f(item.first, item.second);
            }
        });
    }
    acc.finalize();

    return buffer;
}

TCM_NAMESPACE_END
