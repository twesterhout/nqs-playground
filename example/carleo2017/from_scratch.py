from loguru import logger
import lattice_symmetries as ls
import numpy as np
import torch
import torch.nn.functional as F
import nqs_playground as nqs
import nqs_playground.sr

np.random.seed(77)
torch.manual_seed(18)
if torch.cuda.is_available():
    torch.cuda.manual_seed(18)
nqs.manual_seed(87)

class SpinRBM(torch.nn.Linear):
    def __init__(self, number_visible: int, number_hidden: int):
        super().__init__(number_visible, number_hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nqs.unpack(x, self.in_features).to(self.weight.dtype)
        y = F.linear(x, self.weight, self.bias)
        y = y - 2.0 * F.softplus(y, beta=-2.0)
        return y.sum(dim=1, keepdim=True)

class Phase(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spins):
        return torch.zeros((spins.size(0), 1), dtype=torch.float32, device=spins.device)


def heisenberg1d_model(number_spins: int):
    assert number_spins > 0
    if number_spins % 2 != 0:
        raise NotImplementedError()
    sites = np.arange(number_spins)
    basis = ls.SpinBasis(
        ls.Group([ls.Symmetry(sites[::-1], sector=0)]),
        number_spins=number_spins,
        hamming_weight=number_spins // 2,
        spin_inversion=1,
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


def optimize_with_sr(number_spins: int = 40):
    hamiltonian = heisenberg1d_model(number_spins)
    basis = hamiltonian.basis
    if basis.number_spins <= 16:
        basis.build()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amplitude = SpinRBM(number_spins, number_spins).to(device)
    # amplitude = torch.jit.script(
    #     torch.nn.Sequential(
    #         nqs.Unpack(number_spins),
    #         torch.nn.Linear(number_spins, number_spins),
    #         torch.nn.PReLU(),
    #         torch.nn.Linear(number_spins, number_spins),
    #         torch.nn.PReLU(),
    #         torch.nn.Linear(number_spins, 1, bias=False),
    #     )
    # ).to(device)
    # amplitude.load_state_dict(
    #     torch.load("runs_1x40/05/checkpoints/state_dict_0651_000.pt")["amplitude"]
    # )
    # amplitude = LogAmplitude(number_spins, number_blocks=2, number_channels=16).to(device)
    num_parameters = sum(t.numel() for t in amplitude.parameters())
    logger.info("Amplitude network contains {} parameters", num_parameters)
    phase = Phase().to(device)
    # optimizer = torch.optim.SGD(
    #     amplitude.parameters(),
    #     lr=1e-4,
    #     momentum=0.9,
    #     # weight_decay=1e-4,
    # )
    # optimizer = torch.optim.Adam(amplitude.parameters(), lr=1e-3)
    parts = [
        # (5000, 1, torch.optim.SGD(amplitude.parameters(), lr=1e-3)),
        (250, 1, torch.optim.SGD(amplitude.parameters(), lr=1e-2)),
        # (512 * number_spins, 1, torch.optim.SGD(amplitude.parameters(), lr=1e-2))
    ]
    options = nqs.sr.Config(
        amplitude=amplitude,
        phase=phase,
        hamiltonian=hamiltonian,
        output="runs_1x{}/09".format(number_spins),
        epochs=0,
        sampling_options=nqs.SamplingOptions(
            number_samples=1,
            number_chains=4,
            sweep_size=5,
            number_discarded=10,
            mode="zanella",
            other={"edges": [(i, (i + 1) % number_spins) for i in range(number_spins)]},
        ),
        exact=None,
        optimizer=None,
        scheduler=None,
        linear_system_kwargs={"rcond": 1e-4},
        inference_batch_size=16 * 1024,
    )
    runner = nqs.sr.Runner(options)

    for (batch_size, number_inner, optimizer) in parts:
        runner.config = runner.config._replace(
            sampling_options=runner.config.sampling_options._replace(number_samples=batch_size),
            optimizer=optimizer,
            epochs=runner.config.epochs + 1000,
        )
        runner.run(number_inner=number_inner)


if __name__ == "__main__":
    optimize_with_sr(40)
