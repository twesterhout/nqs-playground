import h5py
from loguru import logger
import lattice_symmetries as ls
import numpy as np
import torch
import torch.nn.functional as F
import nqs_playground as nqs
import nqs_playground.sgd
import nqs_playground.sr
import nqs_playground.swo

np.random.seed(52330877)
torch.manual_seed(9218223294)
nqs.manual_seed(2345398712)


class Phase(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spins):
        return torch.zeros((spins.size(0), 1), dtype=torch.float32, device=spins.device)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=0
            ),
            torch.nn.ReLU(inplace=True),
            # torch.nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x):
        x = torch.cat([x[:, :, -1:], x, x[:, :, :1]], dim=2)
        return self.layer(x)


class LogAmplitude(torch.nn.Module):
    def __init__(self, number_spins, number_blocks=3, number_channels=8):
        super().__init__()
        layers = [ConvBlock(in_channels=1, out_channels=number_channels)]
        for i in range(number_blocks - 1):
            layers.append(ConvBlock(in_channels=number_channels, out_channels=number_channels))

        self.number_spins = number_spins
        self.layers = torch.nn.Sequential(*layers)
        self.tail = torch.nn.Linear(number_channels, 1, bias=False)

    def forward(self, x):
        x = nqs.unpack(x, self.number_spins).unsqueeze(dim=1)
        x = self.layers(x).sum(dim=2)
        return self.tail(x)


def heisenberg1d_model(number_spins: int):
    assert number_spins > 0
    if number_spins % 2 != 0:
        raise NotImplementedError()
    sites = np.arange(number_spins)
    basis = ls.SpinBasis(
        # ls.Group([ls.Symmetry(sites[::-1], sector=0)]),
        ls.Group([]),
        number_spins=number_spins,
        hamming_weight=number_spins // 2,
        # spin_inversion=1,
    )
    # fmt: off
    matrix = np.array([[1,  0,  0, 0],
                       [0, -1, -2, 0],
                       [0, -2, -1, 0],
                       [0,  0,  0, 1]])
    # fmt: on
    edges = [(i, (i + 1) % number_spins) for i in range(number_spins)]
    hamiltonian = ls.Operator(basis, [ls.Interaction(matrix, edges)])
    return hamiltonian


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


def test_second_power(number_spins=4):
    basis = ls.SpinBasis(ls.Group([]), number_spins=number_spins, hamming_weight=number_spins // 2)
    # fmt: off
    matrix = np.array([[1,  0,  0, 0],
                       [0, -1, -2, 0],
                       [0, -2, -1, 0],
                       [0,  0,  0, 1]])
    # fmt: on
    edges = [(i, (i + 1) % number_spins) for i in range(number_spins)]
    hamiltonian = ls.Operator(basis, [ls.Interaction(matrix, edges)])
    hamiltonian_squared = ls.Operator(basis, _second_power(matrix, edges, coeff=1.0))

    basis.build()
    x = np.random.rand(basis.number_states) - 0.5
    y1 = hamiltonian_squared(x)
    y2 = hamiltonian(hamiltonian(x))
    print(np.allclose(y1, y2))


def heisenberg1d_evolution_operator(number_spins: int):
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


def pretrain_model(amplitude):
    number_spins = 24
    hamiltonian = heisenberg1d_model(number_spins)
    basis = hamiltonian.basis
    basis.build()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amplitude = amplitude.to(device)

    with h5py.File("data/heisenberg_chain_24.h5", "r") as input:
        logger.debug("Loading the ground state...")
        ground_state = input["/hamiltonian/eigenvectors"][:]
        ground_state = ground_state[:, 0]
        logger.debug("Hilbert space dimensionis {}", len(ground_state))
    assert basis.number_states == len(ground_state)

    target_amplitude = torch.from_numpy(ground_state).to(torch.float32).abs().squeeze().to(device)
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
    for epoch in range(3):
        for x, φ in dataloader:
            optimizer.zero_grad()
            log_ψ = amplitude(x)
            log_ψ = log_ψ - torch.max(log_ψ)
            loss = negative_log_overlap(log_ψ, φ, torch.ones_like(φ))
            loss.backward()
            optimizer.step()
        torch.save(
            {"model": amplitude.state_dict(), "optimizer": optimizer.state_dict()},
            "checkpoint_{}.pt".format(epoch),
        )
        logger.info("Epoch {}: {}", epoch, overlap_error())


def analyze_checkpoint(filename: str, number_spins: int = 40, device=None):
    hamiltonian = heisenberg1d_model(number_spins)
    basis = hamiltonian.basis
    if number_spins <= 16:
        basis.build()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    amplitude = LogAmplitude(number_spins, number_blocks=4, number_channels=16)
    # amplitude = torch.nn.Sequential(
    #     nqs.Unpack(number_spins),
    #     torch.nn.Linear(number_spins, 13),
    #     torch.nn.ReLU(inplace=True),
    #     torch.nn.Linear(13, 1, bias=False)
    # )
    amplitude.to(device)
    logger.info(
        "Amplitude network contains {} parameters", sum(t.numel() for t in amplitude.parameters())
    )
    amplitude.load_state_dict(torch.load(filename)["amplitude"])
    phase = Phase()
    phase.to(device)

    sampling_options = nqs.SamplingOptions(
        number_samples=1000, number_chains=4, sweep_size=1, number_discarded=10, device=device
    )
    states, _, info = nqs.sample_some(amplitude, basis, sampling_options, mode="zanella")
    logger.info("Info from the sampler: {}", info)

    combined_state = nqs.combine_amplitude_and_phase(amplitude, phase)
    local_energies = nqs.local_values(states, hamiltonian, combined_state, batch_size=8192,)
    logger.info("Energy: {}", local_energies.mean(dim=0).cpu())
    logger.info("Energy variance: {}", local_energies.var(dim=0).cpu())


def optimize_with_swo(number_spins: int = 10):
    hamiltonian = heisenberg1d_model(number_spins)
    basis = hamiltonian.basis
    # basis.build()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # amplitude = LogAmplitude(number_spins, number_blocks=5, number_channels=8).to(device)
    amplitude = LogAmplitude(number_spins, number_blocks=6, number_channels=32).to(device)
    amplitude.load_state_dict(torch.load("checkpoint_2.pt")["model"])
    # amplitude = torch.jit.script(
    #     torch.nn.Sequential(
    #         nqs.Unpack(number_spins),
    #         torch.nn.Linear(number_spins, 2 * number_spins),
    #         torch.nn.PReLU(),
    #         torch.nn.Linear(2 * number_spins, 1, bias=False),
    #     )
    # ).to(device)
    phase = Phase().to(device)
    # optimizer = torch.optim.SGD(amplitude.parameters(), lr=1e-2)
    # scheduler = None # torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda _: 0.997)
    evolution_operator = heisenberg1d_evolution_operator(number_spins)

    options = nqs.swo.Config(
        amplitude=amplitude,
        phase=phase,
        hamiltonian=hamiltonian,
        evolution_operator=evolution_operator,
        output="runs_1x{}/11".format(number_spins),
        epochs=100,
        sampling_options=nqs.SamplingOptions(
            number_samples=500, number_chains=8, sweep_size=10, mode="zanella"
        ),
        training_options=nqs.swo.TrainingOptions(
            epochs=100,
            batch_size=500 * 8,
            optimizer=torch.optim.Adam(amplitude.parameters(), lr=1e-3),
            scheduler=None,
        ),
        inference_batch_size=16 * 1024,
    )
    runner = nqs.swo.Runner(options)
    runner.run(number_inner=10)


def optimize_with_full_sr(number_spins: int = 10):
    hamiltonian = heisenberg1d_model(number_spins)
    basis = hamiltonian.basis
    basis.build()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amplitude = LogAmplitude(number_spins, number_blocks=5, number_channels=8).to(device)
    # amplitude = torch.jit.script(
    #     torch.nn.Sequential(
    #         nqs.Unpack(number_spins),
    #         torch.nn.Linear(number_spins, 2 * number_spins),
    #         torch.nn.PReLU(),
    #         torch.nn.Linear(2 * number_spins, 1, bias=False),
    #     )
    # )
    phase = Phase().to(device)
    optimizer = torch.optim.SGD(amplitude.parameters(), lr=1e-2)
    scheduler = (
        None  # torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda _: 0.997)
    )

    options = nqs.sr.Config(
        amplitude=amplitude,
        phase=phase,
        hamiltonian=hamiltonian,
        output="runs_1x{}/06".format(number_spins),
        epochs=100,
        sampling_options=nqs.SamplingOptions(number_samples=1, number_chains=1, mode="full",),
        exact=None,
        optimizer=optimizer,
        scheduler=scheduler,
        inference_batch_size=16 * 1024,
        linear_system_kwargs={"rcond": 1e-4},
    )
    runner = nqs.sr.Runner(options)
    runner.run(number_inner=10)
    pass


def optimize_with_sgd(number_spins: int = 40):
    hamiltonian = heisenberg1d_model(number_spins)
    basis = hamiltonian.basis
    if basis.number_spins <= 16:
        basis.build()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # amplitude = torch.jit.script(
    #     torch.nn.Sequential(
    #         nqs.Unpack(number_spins),
    #         torch.nn.Linear(number_spins, 13),
    #         torch.nn.ReLU(inplace=True),
    #         torch.nn.Linear(13, 1, bias=False),
    #     )
    # )
    amplitude = LogAmplitude(number_spins, number_blocks=6, number_channels=32).to(device)
    # amplitude = torch.nn.Sequential(
    #     nqs.Unpack(number_spins),
    #     torch.nn.Linear(number_spins, 13),
    #     torch.nn.ReLU(inplace=True),
    #     torch.nn.Linear(13, 1, bias=False)
    # )
    logger.info(
        "Amplitude network contains {} parameters", sum(t.numel() for t in amplitude.parameters())
    )
    # amplitude.load_state_dict(torch.load("runs_1x40/02/checkpoints/state_dict_00499.pt")["amplitude"])
    amplitude.load_state_dict(torch.load("checkpoint_2.32.pt")["model"])
    phase = Phase().to(device)
    optimizer = torch.optim.SGD(
        amplitude.parameters(),
        lr=1e-3,  # 5e-4 for 06
        momentum=0.9,
        # weight_decay=1e-4,
    )
    # optimizer = torch.optim.Adam(amplitude.parameters(), lr=1e-3)
    scheduler = (
        None  # torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda _: 0.997)
    )

    batch_size = 2048  # 1024 for 06
    options = nqs.sgd.Config(
        amplitude=amplitude,
        phase=phase,
        hamiltonian=hamiltonian,
        output="runs_1x{}/07".format(number_spins),
        epochs=1000,
        sampling_options=nqs.SamplingOptions(
            number_samples=batch_size,
            number_chains=8,
            sweep_size=10,
            number_discarded=10,
            other={"edges": [(i, (i + 1) % number_spins) for i in range(number_spins)]},
            mode="zanella",
        ),
        exact=None,
        constraints=dict(),  # {"hamming_weight": lambda i: 0.1},
        optimizer=optimizer,
        scheduler=scheduler,
        inference_batch_size=8192,
        checkpoint_every=1,
    )
    runner = nqs.sgd.Runner(options)
    runner.run(number_inner=50)
    # if runner._epoch % 50 == 0:
    #     batch_size *= 2
    #     runner.config = options._replace(
    #         sampling_options=nqs.SamplingOptions(number_samples=batch_size,
    #             number_chains=1, sweep_size=1, number_discarded=10),
    #     )


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
    optimize_with_swo(40)
    # test_second_power()


if __name__ == "__main__":
    main()
