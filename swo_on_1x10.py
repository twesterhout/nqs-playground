#!/usr/bin/env python3

import os
import numpy as np
import pickle
import torch
import nqs_playground
from nqs_playground.symmetry import Symmetry, make_group
from nqs_playground._C_nqs.v2 import SpinBasis, Heisenberg


NUMBER_SPINS = 24


def make_basis():
    assert NUMBER_SPINS % 2 == 0
    filename = "data/1x{}/basis.pickle".format(NUMBER_SPINS)
    if not os.path.exists(filename):
        extend = lambda p: p + list(range(NUMBER_SPINS, 64))
        symmetry_group = make_group(
            [
                # Translation
                Symmetry(
                    extend(list(range(1, NUMBER_SPINS)) + [0]), sector=0 # NUMBER_SPINS // 2
                ),
                # Parity
                Symmetry(extend(list(range(NUMBER_SPINS))[::-1]), sector=0),
            ]
        )
        basis = SpinBasis(
            symmetry_group, number_spins=NUMBER_SPINS, hamming_weight=NUMBER_SPINS // 2
        )
        # Construct the full list of representatives since the system is small
        basis.build()
        nqs_playground.core.with_file_like(
            filename, "wb", lambda f: pickle.dump(basis, f)
        )
    else:
        basis = nqs_playground.core.with_file_like(
            filename, "rb", lambda f: pickle.load(f)
        )

    return basis


def make_hamiltonian(basis):
    n = basis.number_spins
    return Heisenberg([(1.0, i, (i + 1) % n) for i in range(n)], basis)


def make_exact(hamiltonian):
    filename = "data/1x{}/ground_state.npy".format(NUMBER_SPINS)
    if not os.path.exists(filename):
        # assert False
        energy, ground_state = nqs_playground.symmetry.diagonalise(hamiltonian, k=1)
        ground_state = ground_state.squeeze()
        np.save(filename, ground_state, allow_pickle=False)
        print(energy)
    else:
        ground_state = np.load(filename)
    return ground_state


def make_amplitude_network():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Sequential(
                # a 1D convolutional layer
                torch.nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm1d(16),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2, stride=2),
                # another convolutional layer
                torch.nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm1d(16),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2, stride=2),
                # and another convolution layer
                torch.nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm1d(16),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2, stride=2),
                # and another convolution layer
                torch.nn.Conv1d(16, 16, kernel_size=5, stride=1, padding=2),
                torch.nn.BatchNorm1d(16),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2, stride=2),
            )
            self.linear = torch.nn.Sequential(torch.nn.Linear(16, 1))

        def forward(self, x):
            x = x.view(x.size(0), 1, -1)
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            return x  # torch.exp(x)

    return torch.jit.script(Net())


def make_sign_network():
    Net = lambda n: torch.nn.Sequential(
        torch.nn.Linear(n, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, 2),
    )
    return torch.jit.script(Net(NUMBER_SPINS))


def make_networks():
    amplitude = make_amplitude_network()
    sign = make_sign_network()
    # amplitude = torch.jit.script(
    #     torch.nn.Sequential(
    #         torch.nn.Linear(NUMBER_SPINS, 16),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(16, 16),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(16, 16),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(16, 1, bias=False),
    #     )
    # )
    # sign = torch.jit.script(
    #     torch.nn.Sequential(
    #         torch.nn.Linear(NUMBER_SPINS, 16),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(16, 16),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(16, 16),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(16, 2, bias=False),
    #     )
    # )
    return amplitude, sign


def main():
    basis = make_basis()
    hamiltonian = make_hamiltonian(basis)
    ground_state = make_exact(hamiltonian)
    return
    config = nqs_playground.swo.Config(
        model=make_networks(),
        hamiltonian=hamiltonian,
        epochs=50,
        roots=[35.0] * 2,
        number_samples=40000,
        number_chains=1,
        output="swo_on_1x{}_output/1".format(NUMBER_SPINS),
        exact=ground_state,
        amplitude=nqs_playground.swo.TrainingOptions(
            max_epochs=200,
            patience=20,
            train_fraction=0.8,
            train_batch_size=16,
            optimiser=lambda m: torch.optim.RMSprop(
                m.parameters(), lr=5e-4, weight_decay=1e-5
            ),
        ),
        sign=nqs_playground.swo.TrainingOptions(
            max_epochs=200,
            patience=20,
            train_fraction=0.8,
            train_batch_size=16,
            optimiser=lambda m: torch.optim.RMSprop(
                m.parameters(), lr=5e-4, weight_decay=3e-5
            ),
        ),
    )
    print("Starting SWO...")
    nqs_playground.swo.run(config)


if __name__ == "__main__":
    main()
