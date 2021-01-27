from loguru import logger
import lattice_symmetries as ls
import numpy as np
import torch

np.random.seed(52339877)
torch.manual_seed(9218823294)

import nqs_playground as nqs
import nqs_playground.sr
import nqs_playground.autoregressive


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
    use_autoregressive = False
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

    amplitude = make_amplitude("dense")
    phase = make_phase()
    optimizer = torch.optim.SGD(list(amplitude.parameters()) + list(phase.parameters()), lr=1e-2)
    options = nqs_playground.sr.Config(
        amplitude=amplitude,
        phase=phase,
        hamiltonian=operator,
        optimizer=optimizer,
        epochs=100,
        output="1x10.result",
        exact=None,
        sampling_options=nqs.SamplingOptions(number_samples=2000, number_chains=1),
        sampling_mode="exact",
        linear_system_kwargs={"rcond": 1e-4},
        inference_batch_size=8192,
    )
    runner = nqs_playground.sr.Runner(options)
    for i in range(options.epochs):
        runner.step()


if __name__ == "__main__":
    main()
