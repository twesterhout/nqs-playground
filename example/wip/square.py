#!/usr/bin/env python3

import pathlib
import pickle
from typing import Tuple

from loguru import logger
import numpy as np
import torch

from nqs_playground import *

# Fix random number generator seeds for reproducible runs
np.random.seed(1273982371)
torch.manual_seed(378549284)
manual_seed(5772842)

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()


def _site_positions(shape):
    L_x, L_y = shape
    sites = np.arange(L_x * L_y, dtype=np.int32)
    x = sites % L_x
    y = sites // L_x
    return sites, x, y


def _make_basis(shape: Tuple[int, int], use_symmetries=True):
    sites, x, y = _site_positions(shape)
    L_x, L_y = shape
    n = L_x * L_y

    symmetries = []
    # NOTE: We use translations only when number of sites in that direction is
    # even. No idea why but ground states of a samples with off number of sites
    # aren't momentum eigenstates...
    if L_x % 2 == 0 and L_x > 1:
        T_x = (x + 1) % L_x + L_x * y # translation along x-direction
        symmetries.append(Symmetry(T_x, sector=0))
    if L_y % 2 == 0 and L_y > 1:
        T_y = x + L_x * ((y + 1) % L_y) # translation along y-direction
        symmetries.append(Symmetry(T_y, sector=0))
    if L_x == L_y: # Rotations are valid only for square samples
        R = np.rot90(sites.reshape(L_y, L_x), k=-1).reshape(-1)
        symmetries.append(Symmetry(R, sector=0))
    if L_y > 1: # Always enforce parity
        P_x = x + L_x * (L_y - 1 - y) # reflection over x-axis
        symmetries.append(Symmetry(P_x, sector=0))
    if L_x > 1: # Always enforce parity
        P_y = (L_x - 1 - x) + L_x * y # reflection over y-axis
        symmetries.append(Symmetry(P_y, sector=0))
    symmetries = make_group(symmetries) if use_symmetries else []
    return SpinBasis(symmetries, number_spins=n, hamming_weight=n // 2)


def make_basis(shape, use_symmetries=True, build=True):
    logger.info("Creating basis...")
    if use_symmetries:
        prefix = CURRENT_DIR / "ED" / "with"
    else:
        prefix = CURRENT_DIR / "ED" / "without"
    filename = prefix / "basis_{}x{}.pickle".format(*shape)
    if not pathlib.Path(filename).exists():
        basis = _make_basis(shape)
        if build:
            logger.info("Building list of representatives...")
            basis.build()
        logger.info("Saving to file...")
        prefix.mkdir(parents=True, exist_ok=True)
        with_file_like(filename, "wb", lambda f: pickle.dump(basis, f))
    else:
        logger.info("Reading cached basis from file...")
        basis = with_file_like(filename, "rb", pickle.load)
    if build:
        logger.info("Hilbert space dimension is {}", basis.number_states)
    return basis


def make_hamiltonian(shape, basis):
    sites, x, y = _site_positions(shape)
    L_x, L_y = shape

    j1_edges = []
    if L_x > 1:
        j1_edges += list(zip(sites, (x + 1) % L_x + L_x * y))
    if L_y > 1:
        j1_edges += list(zip(sites, x + L_x * ((y + 1) % L_y)))
    return Heisenberg([(1, *e) for e in j1_edges], basis)


def make_exact(shape, hamiltonian, use_symmetries=True, check_state=True):
    if use_symmetries:
        prefix = CURRENT_DIR / "ED" / "with"
    else:
        prefix = CURRENT_DIR / "ED" / "without"
    filename = prefix / "state_{}x{}_{}.npy".format(*shape, 0)
    if not pathlib.Path(filename).exists():
        logger.info("Diagonalising...")
        energy, ground_state = diagonalise(hamiltonian, k=1)
        ground_state = ground_state.squeeze()
        prefix.mkdir(parents=True, exist_ok=True)
        np.save(filename, ground_state, allow_pickle=False)
        logger.info("Ground state energy: {}", float(energy[0]))
    else:
        ground_state = np.load(filename)
        if check_state:
            logger.info("Checking cached ground state...")
            Hy = hamiltonian(ground_state)
            E = np.dot(ground_state.conj(), Hy)
            close = np.allclose(E * ground_state, Hy)
            if close:
                logger.info("Ground state energy: {}", E)
            else:
                raise ValueError("'{}' contains an invalid eigenstate".format(filename))
    return ground_state


def _num_parameters(m):
    return sum(map(torch.numel, filter(lambda p: p.requires_grad, m.parameters())))


def make_amplitude_network(n, p):
    m = torch.nn.Sequential(
        Unpack(n),
        torch.nn.Linear(n, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(p),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(p),
        torch.nn.Linear(256, 1, bias=False),
    )
    logger.info("Constructed amplitude network with {} parameters", _num_parameters(m))
    return torch.jit.script(m)


def make_sign_network(n, p):
    m = torch.nn.Sequential(
        Unpack(n),
        torch.nn.Linear(n, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(p),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(p),
        torch.nn.Linear(256, 2, bias=False),
    )
    logger.info("Constructed sign network with {} parameters", _num_parameters(m))
    return torch.jit.script(m)


def make_networks(n):
    m1, m2 = make_amplitude_network(n), make_sign_network(n)
    if torch.cuda.is_available():
        m1 = m1.cuda()
        m2 = m2.cuda()
    return m1, m2


def config_with_poly_and_dropout(roots, dropout, initial_config):
    n = initial_config.hamiltonian.basis.number_spins
    amplitude = make_amplitude_network(n, dropout[0])
    sign = make_sign_network(n, dropout[1])
    if torch.cuda.is_available():
        amplitude = amplitude.cuda()
        sign = sign.cuda()
    amplitude.load_state_dict(initial_config.model[0].state_dict())
    sign.load_state_dict(initial_config.model[1].state_dict())
    return initial_config._replace(model=(amplitude, sign), roots=roots)


def do_4x6():
    import nqs_playground.swo

    shape = (4, 6)
    basis = make_basis(shape, use_symmetries=True, build=True)
    hamiltonian = make_hamiltonian(shape, basis)
    ground_state = make_exact(shape, hamiltonian, use_symmetries=True, check_state=True)
    n = basis.number_spins
    i = 1
    while pathlib.Path("SWO/{}x{}/{}".format(*shape, i)).exists():
        i += 1

    m1, m2 = make_amplitude_network(n, 0.2), make_sign_network(n, 0.3)
    if torch.cuda.is_available():
        m1 = m1.cuda()
        m2 = m2.cuda()

    initial_iteration = 0
    config = nqs_playground.swo.Config(
        model=(m1, m2),
        output="SWO/{}x{}/{}".format(*shape, i),
        hamiltonian=hamiltonian,
        roots=lambda _: [2 * n, 2 * n],
        epochs=50,
        sampling_method="zanella",
        number_chains=20 * n,
        number_samples=2000 // 20,
        sweep_size=10,
        amplitude=lambda _: nqs_playground.swo.TrainingOptions(
            train_batch_size=128,
            max_epochs=20,
            optimiser=lambda p: torch.optim.Adam(p, lr=1e-4),
        ),
        sign=lambda _: nqs_playground.swo.TrainingOptions(
            train_batch_size=128,
            max_epochs=5,
            optimiser=lambda p: torch.optim.RMSprop(p, lr=1e-4),
        ),
        exact=ground_state,
    )
    nqs_playground.swo.run(config)
    initial_iteration += config.epochs

    config = config_with_poly_and_dropout(lambda _: [-20, 30], (0.1, 0.3), config)
    nqs_playground.swo.run(config, initial_iteration)
    initial_iteration += config.epochs

    config = config_with_poly_and_dropout(lambda _: [-20, 30], (0.0, 0.1), config)
    config = config._replace(
        amplitude=lambda _: nqs_playground.swo.TrainingOptions(
            train_batch_size=1024,
            max_epochs=20,
            optimiser=lambda p: torch.optim.Adam(p, lr=1e-4),
        )
    )
    nqs_playground.swo.run(config, initial_iteration)
    initial_iteration += config.epochs


def do_6x6():
    import nqs_playground.swo

    shape = (6, 6)
    basis = make_basis(shape, use_symmetries=True, build=True)
    hamiltonian = make_hamiltonian(shape, basis)
    ground_state = make_exact(shape, hamiltonian, use_symmetries=True, check_state=False)
    n = basis.number_spins
    i = 1
    while pathlib.Path("SWO/{}x{}/{}".format(*shape, i)).exists():
        i += 1

    # m1, m2 = make_amplitude_network(n, 0.2), make_sign_network(n, 0.3)
    m1 = torch.jit.load("SWO/6x6/2/49/amplitude.pt")
    m2 = torch.jit.load("SWO/6x6/2/49/sign.pt")
    if torch.cuda.is_available():
        m1 = m1.cuda()
        m2 = m2.cuda()

    initial_iteration = 0
    config = nqs_playground.swo.Config(
        model=(m1, m2),
        output="SWO/{}x{}/{}".format(*shape, i),
        hamiltonian=hamiltonian,
        roots=lambda _: [2 * n, 2 * n],
        epochs=50,
        sampling_method="zanella",
        number_chains=20 * n,
        number_samples=2000 // 20,
        sweep_size=10,
        amplitude=lambda _: nqs_playground.swo.TrainingOptions(
            train_batch_size=128,
            max_epochs=20,
            optimiser=lambda p: torch.optim.Adam(p, lr=1e-4),
        ),
        sign=lambda _: nqs_playground.swo.TrainingOptions(
            train_batch_size=128,
            max_epochs=7,
            optimiser=lambda p: torch.optim.RMSprop(p, lr=1e-4),
        ),
        exact=ground_state,
    )
    # nqs_playground.swo.run(config)
    initial_iteration += config.epochs

    config = config_with_poly_and_dropout(lambda _: [-30, 50], (0.1, 0.3), config)
    nqs_playground.swo.run(config, initial_iteration)
    initial_iteration += config.epochs

    config = config_with_poly_and_dropout(lambda _: [-30, 50], (0.0, 0.1), config)
    config = config._replace(
        amplitude=lambda _: nqs_playground.swo.TrainingOptions(
            train_batch_size=1024,
            max_epochs=30,
            optimiser=lambda p: torch.optim.Adam(p, lr=1e-4),
        )
    )
    nqs_playground.swo.run(config, initial_iteration)
    initial_iteration += config.epochs


def do_swo():
    import nqs_playground.swo

    shape = (4, 4)
    basis = make_basis(shape, use_symmetries=True, build=True)
    hamiltonian = make_hamiltonian(shape, basis)
    ground_state = make_exact(shape, hamiltonian, use_symmetries=True, check_state=True)
    n = basis.number_spins
    config = nqs_playground.swo.Config(
        model=make_networks(n),
        output="SWO/{}x{}/1".format(*shape),
        hamiltonian=hamiltonian,
        roots=lambda i: [2 * n, 2 * n],
        epochs=100,
        sampling_method="zanella",
        number_chains=20 * n,
        number_samples=2000 // 20,
        sweep_size=10,
        amplitude=lambda i: nqs_playground.swo.TrainingOptions(
            train_batch_size=128,
            max_epochs=20,
            optimiser=lambda p: torch.optim.Adam(p, lr=1e-4),
        ),
        sign=lambda i: nqs_playground.swo.TrainingOptions(
            train_batch_size=128,
            max_epochs=5,
            optimiser=lambda p: torch.optim.RMSprop(p, lr=1e-4),
        ),
        exact=ground_state,
    )
    nqs_playground.swo.run(config)


def main():
    do_6x6()
    # do_swo()


if __name__ == "__main__":
    main()
