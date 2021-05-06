import lattice_symmetries as ls
from loguru import logger
import nqs_playground as nqs
import time
import torch
from torch import Tensor


def load_basis_and_hamiltonian(
    filename: str = "/vol/tcm28/westerhout_tom/papers/clever-sampling/exact_diagonalization/heisenberg_square_36.yaml",
):
    import yaml

    logger.info("Loading basis from '{}'...", filename)
    with open(filename, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    basis = ls.SpinBasis.load_from_yaml(config["basis"])
    logger.info("Loading Hamiltonian from '{}'...", filename)
    hamiltonian = ls.Operator.load_from_yaml(config["hamiltonian"], basis)
    return basis, hamiltonian


def main():
    number_spins = 36
    device = torch.device("cpu")
    amplitude = torch.nn.Sequential(
        nqs.Unpack(number_spins),
        torch.nn.Linear(number_spins, 144),
        torch.nn.ReLU(),
        torch.nn.Linear(144, 1, bias=False),
    ).to(device)
    phase = torch.nn.Sequential(
        nqs.Unpack(number_spins),
        torch.nn.Linear(number_spins, 144),
        torch.nn.ReLU(),
        torch.nn.Linear(144, 1, bias=False),
    ).to(device)
    combined_state = nqs.combine_amplitude_and_phase(amplitude, phase)
    basis, hamiltonian = load_basis_and_hamiltonian()
    basis.build()

    states, _, info = nqs.sample_some(
        amplitude,
        basis,
        nqs.SamplingOptions(number_samples=1, number_chains=1, device=device, mode="full")
    )

    for n in [2000000]:
        tick = time.time()
        local_energies = nqs.local_values(
            states[:n],
            hamiltonian,
            combined_state,
            batch_size=2048, # 2048
        )
        tock = time.time()
        logger.info("For n = {} local values were computed in {:.2f} seconds", n, tock - tick)


if __name__ == "__main__":
    main()
