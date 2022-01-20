from loguru import logger
import numpy as np
import torch
from torch.nn import functional as F
from typing import Tuple
import nqs_playground as nqs
from unpack_bits import unpack
import lattice_symmetries as ls

# np.random.seed(52)
# torch.manual_seed(92)


class SpinRBM(torch.nn.Linear):
    def __init__(self, number_visible: int, number_hidden: int):
        super().__init__(number_visible, number_hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = unpack(x, self.in_features).to(self.weight.dtype)
        y = F.linear(x, self.weight, self.bias)
        y = y - 2.0 * F.softplus(y, beta=-2.0)
        return y.sum(dim=1, keepdim=True)


class Phase(torch.nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.dtype = dtype

    def forward(self, spins):
        return torch.zeros((spins.size(0), 1), dtype=self.dtype, device=spins.device)


@torch.no_grad()
def _load_vector(stream, count: int):
    out = torch.empty(count, dtype=torch.float64)
    for i in range(count):
        out[i] = float(stream.readline().strip("()").split(",")[0])
    return out


@torch.no_grad()
def load_rbm_weights(filename: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with open(filename, "r") as input:
        number_visible = int(input.readline())
        number_hidden = int(input.readline())
        a = _load_vector(input, number_visible)
        b = _load_vector(input, number_hidden)
        w = _load_vector(input, number_visible * number_hidden).view(number_visible, number_hidden)
        return a, b, w


@torch.no_grad()
def load_rbm(filename: str) -> torch.nn.Module:
    a, b, w = load_rbm_weights(filename)
    assert torch.all(a == 0.0)
    model = SpinRBM(a.numel(), b.numel())
    model.to(torch.float64)
    model.weight.data.copy_(w.t())
    model.bias.data.copy_(b)
    return torch.jit.script(model)


def heisenberg2d_model():
    L = 10
    basis = ls.SpinBasis(ls.Group([]), number_spins=L * L, hamming_weight=L * L // 2)
    # fmt: off
    matrix = np.array([[1,  0,  0, 0],
                       [0, -1, -2, 0],
                       [0, -2, -1, 0],
                       [0,  0,  0, 1]])
    # fmt: on
    edges = []
    for y in range(L):
        for x in range(L):
            edges.append((x + L * y, (x + 1) % L + L * y))
            edges.append((x + L * y, x + L * ((y + 1) % L)))
    hamiltonian = ls.Operator(basis, [ls.Interaction(matrix, edges)])
    return hamiltonian


def heisenberg1d_model(number_spins: int = 40):
    basis = ls.SpinBasis(ls.Group([]), number_spins=number_spins, hamming_weight=number_spins // 2)
    # fmt: off
    matrix = np.array([[1,  0,  0, 0],
                       [0, -1, -2, 0],
                       [0, -2, -1, 0],
                       [0,  0,  0, 1]])
    # fmt: on
    edges = [(i, (i + 1) % number_spins) for i in range(number_spins)]
    hamiltonian = ls.Operator(basis, [ls.Interaction(matrix, edges)])
    return hamiltonian


def reproduce_results_of_askar(filename: str = "amplitude_weights_439.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug("Running the simulation on '{}'...", device)

    hamiltonian = heisenberg2d_model()
    basis = hamiltonian.basis
    log_ψ = torch.jit.script(
        torch.nn.Sequential(
            nqs.Unpack(basis.number_spins),
            torch.nn.Linear(basis.number_spins, 144),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(144, 1, bias=False),
        )
    )
    log_ψ.to(device)
    log_ψ.load_state_dict(torch.load(filename))

    sampling_options = nqs.SamplingOptions(
        number_samples=2000, number_chains=4, number_discarded=100, sweep_size=1, device=device
    )
    states, _, info = nqs.sample_some(log_ψ, basis, sampling_options, mode="zanella")
    logger.info("Info from the sampler: {}", info)

    combined_state = nqs.combine_amplitude_and_phase(log_ψ, Phase(dtype=torch.float32))
    local_energies = nqs.local_values(states, hamiltonian, combined_state,)
    logger.info("Energy: {}", local_energies.mean(dim=0).cpu())
    logger.info("Energy variance: {}", local_energies.var(dim=0).cpu())


def reproduce_results_of_carleo2017(filename: str):
    logger.debug("Loading RBM weights from '{}'...", filename)
    log_ψ = load_rbm(filename)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug("Running Monte Carlo sampling on '{}'...", device)
    log_ψ.to(device)

    if "Heisenberg2d" in filename:
        hamiltonian = heisenberg2d_model()
    elif "Heisenberg1d" in filename:
        hamiltonian = heisenberg1d_model(log_ψ.in_features)
    else:
        raise NotImplementedError()

    basis = hamiltonian.basis
    sampling_options = nqs.SamplingOptions(
        number_samples=2000,
        number_chains=4,
        number_discarded=100,
        sweep_size=1,
        mode="zanella",
        device=device,
    )
    states, _, _, info = nqs.sample_some(log_ψ, basis, sampling_options)
    logger.info("Info from the sampler: {}", info)

    combined_state = nqs.combine_amplitude_and_phase(log_ψ, Phase(dtype=torch.float64))
    local_energies = nqs.local_values(states, hamiltonian, combined_state, batch_size=8192)
    local_energies = local_energies.real
    logger.info("Energy: {}", local_energies.mean(dim=0).cpu())
    logger.info("Energy variance: {}", local_energies.var(dim=0).cpu())


if __name__ == "__main__":
    # reproduce_results_of_askar()
    reproduce_results_of_carleo2017("Nqs/Ground/Heisenberg1d_40_1_2.wf")
    # reproduce_results_of_carleo2017("Nqs/Ground/Heisenberg2d_100_1_32.wf")
