from loguru import logger
import lattice_symmetries as ls
import numpy as np
import torch
import h5py
import yaml

np.random.seed(52339877)
torch.manual_seed(9218823294)

# try:
import nqs_playground
import nqs_playground.sgd
import nqs_playground.autoregressive

# except ImportError:
#     # For local development when we only compile the C++ extension, but don't
#     # actually install the package using pip
#     import os
#     import sys
#
#     sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
#     import nqs_playground


def make_amplitude(arch="dense"):
    if arch == "dense":
        return torch.nn.Sequential(
            nqs_playground.Unpack(36),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1, bias=False),
        )
    if arch == "nade":
        return nqs_playground.autoregressive.NADE(36, 100)
    raise ValueError("invalid arch: {}".format(arch))


def make_phase():
    class Phase(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, spins):
            return torch.zeros((spins.size(0), 1), dtype=torch.float32, device=spins.device)

    return Phase()


def load_ground_state(filename: str):
    logger.info("Loading ground state from '{}'...", filename)
    with h5py.File(filename, "r") as f:
        ground_state = f["/hamiltonian/eigenvectors"][:, 0]
        energy = f["/hamiltonian/eigenvalues"][0]
        basis_representatives = f["/basis/representatives"][:]
    logger.info("Ground state energy is {}", energy)
    return ground_state, energy, basis_representatives


def load_basis_and_hamiltonian(filename: str):
    import yaml

    logger.info("Loading basis from '{}'...", filename)
    with open(filename, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    basis = ls.SpinBasis.load_from_yaml(config["basis"])
    basis = ls.SpinBasis(ls.Group([]), number_spins=basis.number_spins, hamming_weight=None)
    logger.info("Loading Hamiltonian from '{}'...", filename)
    hamiltonian = ls.Operator.load_from_yaml(config["hamiltonian"], basis)
    return basis, hamiltonian


def main():
    use_autoregressive = True

    basis, hamiltonian = load_basis_and_hamiltonian("heisenberg_square_36_positive.yaml")
    ground_state, energy, representatives = load_ground_state("data/heisenberg_square_36.h5")
    # basis.build(representatives=representatives)
    del representatives

    amplitude = make_amplitude("nade" if use_autoregressive else "dense")
    phase = make_phase()
    optimizer = torch.optim.SGD(
        list(amplitude.parameters()) + list(phase.parameters()),
        lr=1e-3,
        momentum=0.9,
        weight_decay=1e-4,
    )
    options = nqs_playground.sgd.Config(
        amplitude=amplitude,
        phase=phase,
        hamiltonian=hamiltonian,
        output="data/6x6",
        epochs=500,
        sampling_options=nqs_playground.SamplingOptions(number_samples=256, number_chains=1),
        sampling_mode="autoregressive" if use_autoregressive else "exact",
        exact=None,  # ground_state[:, 0],
        constraints={"hamming_weight": lambda i: 0.2},
        optimizer=optimizer,
        inference_batch_size=8192,
    )
    runner = nqs_playground.sgd.Runner(options)
    for i in range(options.epochs):
        runner.step()
        # if runner._epoch % 200 == 0:
        #     for g in options.optimizer.param_groups:
        #         g["lr"] /= 2
        #         g["momentum"] /= 2
    # runner.config = options._replace(
    #     sampling_options=nqs_playground.SamplingOptions(number_samples=200000, number_chains=1),
    # )
    # runner.step()


if __name__ == "__main__":
    main()
