from loguru import logger
import lattice_symmetries as ls
import os
import numpy as np
import torch
import torch.nn.functional as F
import nqs_playground as nqs

np.random.seed(53)
torch.manual_seed(19)
if torch.cuda.is_available():
    torch.cuda.manual_seed(19)
nqs.manual_seed(24)


class Phase(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spins):
        return torch.zeros((spins.size(0), 1), dtype=torch.float32, device=spins.device)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=0
            ),
            torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x):
        x = torch.cat([x[:, :, -1:, :], x, x[:, :, :1, :]], dim=2)
        x = torch.cat([x[:, :, :, -1:], x, x[:, :, :, :1]], dim=3)
        return self.layer(x)


class LogAmplitude(torch.nn.Module):
    def __init__(self, shape, number_blocks=3, number_channels=8):
        super().__init__()
        layers = [ConvBlock(in_channels=1, out_channels=number_channels)]
        for i in range(number_blocks - 1):
            layers.append(ConvBlock(in_channels=number_channels, out_channels=number_channels))

        self.shape = shape
        self.layers = torch.nn.Sequential(*layers)
        self.tail = torch.nn.Linear(number_channels, 1, bias=False)

    def forward(self, x):
        number_spins = self.shape[0] * self.shape[1]
        x = nqs.unpack(x, number_spins)
        x = x.view(x.size(0), 1, *self.shape)
        x = self.layers(x)
        x = x.view(*x.size()[:2], -1).sum(dim=2)
        return self.tail(x)


def negative_log_overlap(log_ψ, φ, weights):
    ψ = torch.exp(log_ψ).squeeze()
    normalization = torch.sqrt(torch.dot(ψ * ψ, weights)) * torch.sqrt(torch.dot(φ * φ, weights))
    return 1 - torch.dot(ψ * φ, weights) / normalization


def pretrain_model(amplitude: torch.nn.Module):
    shape = (4, 6)
    number_spins = shape[0] * shape[1]
    log_dir = "runs/pre/{}x{}".format(*shape)
    os.makedirs(log_dir, exist_ok=True)
    hamiltonian = nqs.load_hamiltonian("no_symm/heisenberg_square_{}x{}.yaml".format(*shape))
    ground_state, energy, _representatives = nqs.load_ground_state(
        "data/no_symm/heisenberg_square_{}x{}.h5".format(*shape)
    )
    logger.info("Hilbert space dimension is: {}", ground_state.size(0))
    logger.info(
        "Amplitude network has {} parameters", sum(t.numel() for t in amplitude.parameters())
    )
    basis = hamiltonian.basis
    basis.build(representatives=_representatives)
    _representatives = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amplitude = amplitude.to(device)

    target_amplitude = ground_state.to(torch.float32).abs().to(device)
    states = torch.from_numpy(basis.states.view(np.int64)).to(device)

    @torch.no_grad()
    def overlap_error():
        log_ψ = nqs.forward_with_batches(amplitude, states, batch_size=16 * 1024)
        φ = target_amplitude
        return negative_log_overlap(log_ψ, φ, torch.ones_like(φ))

    logger.info("Initially: {}", overlap_error())
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(states, target_amplitude),
        batch_size=1024,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(amplitude.parameters(), lr=1e-3)
    for epoch in range(10):
        for x, φ in dataloader:
            optimizer.zero_grad()
            log_ψ = amplitude(x)
            log_ψ = log_ψ - torch.max(log_ψ)
            loss = negative_log_overlap(log_ψ, φ, torch.ones_like(φ))
            loss.backward()
            optimizer.step()
        torch.save(
            {"model": amplitude.state_dict(), "optimizer": optimizer.state_dict()},
            os.path.join(log_dir, "state_dict_{}.pt".format(epoch)),
        )
        logger.info("Epoch {}: {}", epoch, overlap_error())


def evaluate_model(amplitude: torch.nn.Module):
    shape = (6, 6)
    E0 = -97.75758959723598
    number_spins = shape[0] * shape[1]
    hamiltonian = nqs.load_hamiltonian("no_symm/heisenberg_square_{}x{}.yaml".format(*shape))
    logger.info(
        "Amplitude network has {} parameters", sum(t.numel() for t in amplitude.parameters())
    )
    basis = hamiltonian.basis

    for t in amplitude.parameters():
        device = t.device
        break
    sampling_options = nqs.SamplingOptions(
        number_samples=2000, number_chains=8, sweep_size=10, mode="zanella", device=device
    )
    states, log_probs, info = nqs.sample_some(amplitude, basis, sampling_options)
    logger.info("Additional info from the sampler: {}", info)

    local_energies = nqs.local_values(
        states,
        hamiltonian,
        nqs.combine_amplitude_and_phase(amplitude, Phase(), use_jit=False),
        batch_size=16 * 1024,
    )
    tau = nqs.integrated_autocorr_time(local_energies)
    logger.info("Autocorrelation time of Eₗ(σ): {}", tau)

    energy_per_chain = torch.mean(local_energies.real, dim=0, keepdim=True)
    variance_per_chain = torch.var(local_energies.real, dim=0, keepdim=True)
    energy_err, energy = torch.std_mean(energy_per_chain)
    variance_err, variance = torch.std_mean(variance_per_chain)
    logger.info("  Energy: {} ± {:.2e}", energy, energy_err)
    logger.info("Variance: {} ± {:.2e}", variance, variance_err)
    logger.info("   Error: {:e}", abs((energy - E0) / E0))


def do_dense():
    amplitude = torch.nn.Sequential(
        nqs.Unpack(16),
        torch.nn.Linear(16, 13),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(13, 1, bias=False),
    )
    pretrain_model(amplitude)


def do_cnn():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if True:
        amplitude = LogAmplitude((6, 6), number_blocks=10, number_channels=16).to(device)
        amplitude.load_state_dict(torch.load("runs/pre/4x6/state_dict_8.pt")["model"])
        evaluate_model(amplitude)
    else:
        amplitude = LogAmplitude((4, 6), number_blocks=10, number_channels=16).to(device)
        pretrain_model(amplitude)


def main():
    # amplitude = torch.nn.Sequential(
    #     nqs.Unpack(24),
    #     torch.nn.Linear(24, 13),
    #     torch.nn.ReLU(inplace=True),
    #     torch.nn.Linear(13, 1, bias=False)
    # )
    # amplitude = LogAmplitude(24, number_blocks=6, number_channels=32)
    # pretrain_model(amplitude)
    # optimize_with_sgd(40)
    # optimize_with_full_sr(10)
    # optimize_with_swo(40)
    # test_second_power()
    # do_dense()
    do_cnn()


if __name__ == "__main__":
    main()
