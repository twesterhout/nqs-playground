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


class SignModel(torch.nn.Module):
    def __init__(self, shape, number_blocks=3, number_channels=8):
        super().__init__()
        layers = [ConvBlock(in_channels=1, out_channels=number_channels)]
        for i in range(number_blocks - 1):
            layers.append(ConvBlock(in_channels=number_channels, out_channels=number_channels))

        self.shape = shape
        self.layers = torch.nn.Sequential(*layers)
        self.tail = torch.nn.Linear(number_channels, 2, bias=False)

    def forward(self, x):
        number_spins = self.shape[0] * self.shape[1]
        x = nqs.unpack(x, number_spins)
        x = x.view(x.size(0), 1, *self.shape)
        x = self.layers(x)
        x = x.view(*x.size()[:2], -1).sum(dim=2)
        return self.tail(x)


def _zeroth_power(matrix: np.ndarray, edges, coeff: float):
    matrix = coeff * np.array([[1, 0], [0, 1]])
    number_spins = 1 + max(max(t) for t in edges)
    edges = [(i,) for i in range(number_spins)]
    return [ls.Interaction(matrix, edges)]


def _first_power(matrix: np.ndarray, edges, coeff: float):
    return [ls.Interaction(coeff * matrix, edges)]


def _second_power(matrix: np.ndarray, edges, coeff: float):
    assert np.all(matrix[[0, 2, 1, 3], :][:, [0, 2, 1, 3]] == matrix)
    edges_0 = []
    edges_1 = []
    edges_2 = []
    for e1 in edges:
        for e2 in edges:
            if e1 == e2 or e1 == e2[::-1]:
                edges_2.append(e1)
            elif e1[0] == e2[0]:
                edges_1.append((e1[1], e1[0], e2[1]))
            elif e1[0] == e2[1]:
                edges_1.append((e1[1], e1[0], e2[0]))
            elif e1[1] == e2[0]:
                edges_1.append((e1[0], e1[1], e2[1]))
            elif e1[1] == e2[1]:
                edges_1.append((e1[0], e1[1], e2[0]))
            else:
                edges_0.append(e1 + e2)
    i_0 = ls.Interaction(coeff * np.kron(matrix, matrix), edges_0)
    # i_1 = ls.Interaction(coeff * (np.kron(matrix, np.eye(2)) @ np.kron(np.eye(2), matrix)), edges_1)
    i_1 = ls.Interaction(coeff * (np.kron(np.eye(2), matrix) @ np.kron(matrix, np.eye(2))), edges_1)
    i_2 = ls.Interaction(coeff * (matrix @ matrix), edges_2)
    return [i_0, i_1, i_2]


def heisenberg2d_evolution_operator(number_spins: int):
    assert number_spins > 0
    if number_spins % 2 != 0:
        raise NotImplementedError()
    basis = ls.SpinBasis(ls.Group([]), number_spins=number_spins, hamming_weight=number_spins // 2)
    # fmt: off
    matrix = np.array([[1,  0,  0, 0],
                       [0, -1, -2, 0],
                       [0, -2, -1, 0],
                       [0,  0,  0, 1]])
    # fmt: on
    edges = [(i, (i + 1) % number_spins) for i in range(number_spins)]
    Λ = float(2 * number_spins)
    return ls.Operator(
        basis,
        _second_power(matrix, edges, coeff=1.0)
        + _first_power(matrix, edges, coeff=-2 * Λ)
        + _zeroth_power(matrix, edges, coeff=Λ ** 2)
        # _second_power(matrix, edges, coeff=-1.0) + _zeroth_power(matrix, edges, coeff=400.0)
    )


def negative_log_overlap(log_ψ, φ, weights):
    ψ = torch.exp(log_ψ).squeeze()
    normalization = torch.sqrt(torch.dot(ψ * ψ, weights)) * torch.sqrt(torch.dot(φ * φ, weights))
    return 1 - torch.dot(ψ * φ, weights) / normalization


def pretrain_sign_model(model: torch.nn.Module):
    shape = (4, 6)
    number_spins = shape[0] * shape[1]
    log_dir = "runs/pre/{}x{}".format(*shape)
    os.makedirs(log_dir, exist_ok=True)
    hamiltonian = nqs.load_hamiltonian("no_symm/heisenberg_square_{}x{}.yaml".format(*shape))
    ground_state, energy, _representatives = nqs.load_ground_state(
        "data/no_symm/heisenberg_square_{}x{}.h5".format(*shape)
    )
    # hamiltonian = nqs.load_hamiltonian("no_symm/heisenberg_square_{}x{}.yaml".format(*shape))
    # ground_state, energy, _representatives = nqs.load_ground_state(
    #     "data/no_symm/heisenberg_square_{}x{}.h5".format(*shape)
    # )
    logger.info("Hilbert space dimension is: {}", ground_state.size(0))
    number_parameters = sum(t.numel() for t in model.parameters())
    logger.info("Sign network has {} parameters", number_parameters)
    basis = hamiltonian.basis
    basis.build(representatives=_representatives)
    _representatives = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    probabilities = (ground_state ** 2).to(dtype=torch.float32, device=device)
    target_labels = torch.where(
        ground_state > 0,
        torch.scalar_tensor(0, dtype=torch.int64),
        torch.scalar_tensor(1, dtype=torch.int64),
    ).to(device)
    states = torch.from_numpy(basis.states.view(np.int64)).to(device)

    @torch.no_grad()
    def overlap_error():
        predicted_labels = nqs.forward_with_batches(lambda x: torch.argmax(model(x), dim=1), states, batch_size=16 * 1024)
        signs = 2 * (target_labels == predicted_labels).to(torch.float32) - 1
        return torch.dot(signs, probabilities)

    logger.info("Initially: {}", overlap_error())
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(states, target_labels, probabilities),
        batch_size=4096,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        total_loss = 0
        total_count = 0
        for x, y, w in dataloader:
            optimizer.zero_grad()
            ŷ = model(x)
            loss = torch.dot(loss_fn(ŷ, y).view(-1), w / torch.sum(w))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_count += x.size(0)
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            os.path.join(log_dir, "sign_state_dict_{}.pt".format(epoch)),
        )
        error = overlap_error().item()
        logger.info("Epoch {}: {} {}", epoch, error, total_loss / total_count)
        if np.isclose(error, 1):
            break


def pretrain_model(amplitude: torch.nn.Module):
    shape = (4, 6)
    number_spins = shape[0] * shape[1]
    log_dir = "runs/pre/{}x{}".format(*shape)
    os.makedirs(log_dir, exist_ok=True)
    hamiltonian = nqs.load_hamiltonian("no_symm/j1j2_square_{}x{}.yaml".format(*shape))
    ground_state, energy, _representatives = nqs.load_ground_state(
        "data/no_symm/j1j2_square_{}x{}.h5".format(*shape)
    )
    # hamiltonian = nqs.load_hamiltonian("no_symm/heisenberg_square_{}x{}.yaml".format(*shape))
    # ground_state, energy, _representatives = nqs.load_ground_state(
    #     "data/no_symm/heisenberg_square_{}x{}.h5".format(*shape)
    # )
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
    basis = hamiltonian.basis
    logger.info(
        "Amplitude network has {} parameters", sum(t.numel() for t in amplitude.parameters())
    )

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


def _pad_states(states):
    """Pad states with zeros to get a Tensor of bits512 instead of int64."""
    padding = torch.zeros(states.size(0), 7, device=states.device, dtype=torch.int64)
    return torch.cat([states.unsqueeze(dim=1), padding], dim=1)


@torch.no_grad()
def evaluate_sign_overlap(model: torch.nn.Module):
    shape = (6, 6)
    number_spins = shape[0] * shape[1]
    # hamiltonian = nqs.load_hamiltonian("symm/j1j2_square_{}x{}.yaml".format(*shape))
    # ground_state, energy, _representatives = nqs.load_ground_state(
    #     "data/symm/j1j2_square_{}x{}.h5".format(*shape)
    # )
    hamiltonian = nqs.load_hamiltonian("symm/heisenberg_square_{}x{}.yaml".format(*shape))
    ground_state, energy, _representatives = nqs.load_ground_state(
        "data/symm/heisenberg_square_{}x{}.h5".format(*shape)
    )
    ground_state = - ground_state
    logger.info("Hilbert space dimension is: {}", ground_state.size(0))
    basis = hamiltonian.basis
    logger.info("Building the basis...")
    basis.build(representatives=_representatives)
    _representatives = None

    for t in model.parameters():
        device = t.device
        break
    states = torch.from_numpy(basis.states.view(np.int64))
    logger.info("Computing ψ...")
    predicted_labels = nqs.forward_with_batches(lambda x: torch.argmax(model(x.to(device)), dim=1), states, batch_size=16 * 1024)
    predicted_signs = 1 - 2 * predicted_labels.to(torch.float32)
    # log_ψ = nqs.forward_with_batches(
    #     lambda x: amplitude(x.to(device)), states, batch_size=32 * 1024
    # )
    # log_ψ -= torch.max(log_ψ)
    # ψ = torch.exp(log_ψ.to(torch.float64))
    # ψ.squeeze_()

    logger.info("Computing φ...")
    # _, _, norm = basis.batched_state_info(_pad_states(states).numpy().view(np.uint64))
    # ψ /= torch.from_numpy(norm).squeeze().to(device)
    # ψ /= torch.linalg.norm(ψ)
    # logger.info("{}", ground_state)
    φ = ground_state.to(device)
    ψ = ground_state.to(device).abs() * predicted_signs
    logger.info("{}, {}", ψ.size(), φ.size())

    logger.info("Overlap: {}", torch.dot(φ, ψ))

@torch.no_grad()
def evaluate_overlap(model: torch.nn.Module):
    shape = (6, 6)
    number_spins = shape[0] * shape[1]
    hamiltonian = nqs.load_hamiltonian("symm/heisenberg_square_{}x{}.yaml".format(*shape))
    ground_state, energy, _representatives = nqs.load_ground_state(
        "data/symm/heisenberg_square_{}x{}.h5".format(*shape)
    )
    # hamiltonian = nqs.load_hamiltonian("symm/heisenberg_square_{}x{}.yaml".format(*shape))
    # ground_state, energy, _representatives = nqs.load_ground_state(
    #     "data/symm/heisenberg_square_{}x{}.h5".format(*shape)
    # )
    logger.info("Hilbert space dimension is: {}", ground_state.size(0))
    logger.info(
        "Amplitude network has {} parameters", sum(t.numel() for t in amplitude.parameters())
    )
    basis = hamiltonian.basis
    logger.info("Building the basis...")
    basis.build(representatives=_representatives)
    _representatives = None

    for t in amplitude.parameters():
        device = t.device
        break
    states = torch.from_numpy(basis.states.view(np.int64))
    logger.info("Computing ψ...")
    log_ψ = nqs.forward_with_batches(
        lambda x: amplitude(x.to(device)), states, batch_size=32 * 1024
    )
    log_ψ -= torch.max(log_ψ)
    ψ = torch.exp(log_ψ.to(torch.float64))
    ψ.squeeze_()

    logger.info("Computing φ...")
    _, _, norm = basis.batched_state_info(_pad_states(states).numpy().view(np.uint64))
    ψ /= torch.from_numpy(norm).squeeze().to(device)
    ψ /= torch.linalg.norm(ψ)
    logger.info("{}", ground_state)
    φ = ground_state.abs().to(device)
    logger.info("{}, {}", ψ.size(), φ.size())

    logger.info("Overlap: {}", torch.dot(φ, ψ))


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
    if False:
        amplitude = LogAmplitude((6, 6), number_blocks=3, number_channels=32).to(device)
        amplitude.load_state_dict(torch.load("runs/pre/4x6/state_dict_9.pt")["model"])
        # evaluate_model(amplitude)
        evaluate_overlap(amplitude)
    else:
        amplitude = LogAmplitude((4, 6), number_blocks=3, number_channels=32).to(device)
        pretrain_model(amplitude)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SignModel((6, 6), number_blocks=3, number_channels=8).to(device)
    model.load_state_dict(torch.load("runs/pre/4x6/sign_state_dict_13.pt")["model"])
    evaluate_sign_overlap(model)
    # pretrain_sign_model(model)
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
    # do_cnn()


if __name__ == "__main__":
    main()
