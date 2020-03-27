from math import sqrt
import numpy as np
import torch
from nqs_playground import *
from nqs_playground.core import Unpack


def analyze(spins, basis):
    counts = np.zeros(basis.number_states, dtype=np.float64)
    for s in spins:
        counts[basis.index(s)] += 1
    return counts


def experiment(n, basis, sample_exactly, sample_metropolis):
    options = SamplingOptions(number_chains=4, number_samples=(n + 4 - 1) // 4)
    q, _ = sample_exactly(options)
    q = analyze(q[:n], basis)
    p, _ = sample_metropolis(options)
    p = analyze(p[:n], basis)

    z = sum(((x - y) ** 2 - x - y) / (x + y) for x, y in zip(p, q) if x + y > 0)
    r = (z, z <= 1.0 * sqrt(n))
    print(r)
    return r


def closeness_testing_l1(n, basis, sample_exactly, sample_metropolis):
    results = [
        experiment(m, basis, sample_exactly, sample_metropolis)
        for m in np.random.poisson(n, size=10)
    ]
    return results


def test_simple():
    basis = SpinBasis([], number_spins=10, hamming_weight=5)
    basis.build()

    network = torch.nn.Sequential(
        Unpack(basis.number_spins),
        torch.nn.Linear(basis.number_spins, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
    log_psi = lambda x: network(x)

    def sample_exactly(options):
        states, log_prob = sample_some(log_psi, basis, options, mode="exact")
        return states[:, 0], log_prob

    def sample_metropolis(options):
        states, log_prob = sample_some(log_psi, basis, options, mode="monte_carlo")
        return states[:, 0], log_prob

    return closeness_testing_l1(10000, basis, sample_exactly, sample_metropolis)


def test_tricky():
    number_spins = 26
    extend = lambda p: p + list(range(number_spins, 64))
    symmetry_group = make_group(
        [
            # Translation
            Symmetry(
                extend(list(range(1, number_spins)) + [0]),
                sector=0,  # NUMBER_SPINS // 2
            ),
            # Parity
            Symmetry(extend(list(range(number_spins))[::-1]), sector=0),
        ]
    )
    basis = SpinBasis(
        symmetry_group, number_spins=number_spins, hamming_weight=number_spins // 2
    )
    basis.build()
    log_psi = torch.nn.Sequential(
        Unpack(basis.number_spins),
        torch.nn.Linear(basis.number_spins, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )

    def sample_exactly(options):
        states, log_prob = sample_some(log_psi, basis, options, mode="exact")
        return states[:, 0], log_prob

    def sample_metropolis(options):
        states, log_prob = sample_some(log_psi, basis, options, mode="monte_carlo")
        return states[:, 0], log_prob

    return closeness_testing_l1(50000, basis, sample_exactly, sample_metropolis)
