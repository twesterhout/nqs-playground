from loguru import logger
import lattice_symmetries as ls
import numpy as np
import torch

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
            nqs_playground.Unpack(10),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1, bias=False),
        )
    if arch == "nade":
        return nqs_playground.autoregressive.NADE(10, 100)
    raise ValueError("invalid arch: {}".format(arch))


def make_phase():
    class Phase(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, spins):
            return torch.zeros((spins.size(0), 1), dtype=torch.float32, device=spins.device)

    return Phase()


def main():
    use_autoregressive = True
    basis = ls.SpinBasis(ls.Group([]), number_spins=10, hamming_weight=None)
    basis.build()

    # fmt: off
    matrix = np.array([[1,  0,  0, 0],
                       [0, -1, -2, 0],
                       [0, -2, -1, 0],
                       [0,  0,  0, 1]])
    # fmt: on
    edges = [(i, (i + 1) % basis.number_spins) for i in range(basis.number_spins)]
    operator = ls.Operator(basis, [ls.Interaction(matrix, edges)])
    # E, ground_state = ls.diagonalize(operator)
    # logger.info("E = {}", E)
    # logger.info("v₀ = {}", ground_state[:, 0])

    amplitude = make_amplitude("nade" if use_autoregressive else "dense")
    phase = make_phase()
    optimizer = torch.optim.SGD(
        list(amplitude.parameters()) + list(phase.parameters()),
        lr=1e-1,
        momentum=0.9,
        weight_decay=1e-4,
    )
    # optimizer = torch.optim.Adam(list(amplitude.parameters()) + list(phase.parameters()), lr=1e-2)
    options = nqs_playground.sgd.Config(
        amplitude=amplitude,
        phase=phase,
        hamiltonian=operator,
        output="1x10.result",
        epochs=600,
        sampling_options=nqs_playground.SamplingOptions(number_samples=128, number_chains=1),
        sampling_mode="autoregressive" if use_autoregressive else "exact",
        exact=None,  # ground_state[:, 0],
        constraints={"hamming_weight": lambda i: 0.1},
        optimizer=optimizer,
        inference_batch_size=8192,
    )
    runner = nqs_playground.sgd.Runner(options)
    for i in range(options.epochs):
        runner.step()
        if runner._epoch % 200 == 0:
            for g in options.optimizer.param_groups:
                g["lr"] /= 2
                g["momentum"] /= 2
    runner.config = options._replace(
        sampling_options=nqs_playground.SamplingOptions(number_samples=200000, number_chains=1),
    )
    runner.step()


if __name__ == "__main__":
    main()
