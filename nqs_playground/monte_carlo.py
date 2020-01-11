from itertools import islice
import math
from typing import Optional, Tuple, Callable

import numpy as np
import torch
from torch import Tensor

from ._C_nqs.v2 import unpack, MetropolisKernel


class Sampler:
    r"""Simple and generic sampler which uses Metropolis-Hastings algorithm to
    approximate the target distribution.
    """

    def __init__(
        self,
        transition_kernel: Callable[[Tensor, Tensor], Tensor],
        log_prob_fn: Callable[[Tensor], Tensor],
        batch_size: int = 32,
    ):
        r"""Constructs the sampler.

        :param transition_kernel: is a function which generates possible
            transitions. Given the current state ``s`` it should return a new
            state ``s'`` and a so-called norm (basically, just a probability
            correction; TODO: explain it better).
        :param log_prob_fn: is a function which when given a state returns its
            unnormalized log probability.
        :param batch_size: number of Markov chains to generate in parallel.
        """
        if batch_size <= 0:
            raise ValueError(
                "invalid batch_size: {}; expected a positive integer".format(batch_size)
            )
        self.kernel = transition_kernel
        self.basis = self.kernel.basis
        self.log_prob_fn = log_prob_fn
        self.batch_size = batch_size

    def bootstrap(self):
        state = _prepare_initial_state(self.basis, self.batch_size)
        log_prob = self.log_prob_fn(state)
        norm = torch.tensor(
            [self.basis.normalisation(x) for x in state], dtype=torch.float32
        )
        return state, norm, log_prob

    def __iter__(self):
        current = self.bootstrap()
        while True:
            yield current
            proposed_state, proposed_norm = self.kernel(current[0], False)
            proposed_log_prob = self.log_prob_fn(proposed_state)
            current = _step(current, (proposed_state, proposed_norm, proposed_log_prob))


class SamplingOptions:
    r"""Options for Monte Carlo sampling spin configurations."""

    def __init__(
        self,
        number_samples: int,
        number_chains: int = 1,
        number_discarded: Optional[int] = None,
    ):
        r"""Initialises the options.

        :param number_samples: specifies the number of samples per Markov
            chain. Must be a positive integer.
        :param number_chains: specifies the number of independent Markov
            chains. Must be a positive integer.
        :param number_discarded: specifies the number of samples to discard
            in the beginning of each Markov chain (i.e. how long should the
            thermalisation procedure be). If specified, must be a positive
            integer. Otherwise, 10% of ``number_samples`` is used.
        """
        self.number_samples = int(number_samples)
        if self.number_samples <= 0:
            raise ValueError(
                "invalid number_samples: {}; expected a positive integer"
                "".format(number_samples)
            )
        self.number_chains = int(number_chains)
        if self.number_chains <= 0:
            raise ValueError(
                "invalid number_chains: {}; expected a positive integer"
                "".format(number_chains)
            )
        if number_discarded is not None:
            self.number_discarded = int(number_discarded)
            if self.number_discarded <= 0:
                raise ValueError(
                    "invalid number_discarded: {}; expected either a positive "
                    "integer or None".format(number_chains)
                )
        else:
            self.number_discarded = self.number_samples // 10


def _sample_using_metropolis(
    fn: Callable[[Tensor], Tensor], basis, options: SamplingOptions
):
    kernel = MetropolisKernel(basis)
    log_prob_fn = lambda x: 2 * fn(unpack(x, basis.number_spins))
    sampler = Sampler(kernel, log_prob_fn, batch_size=options.number_chains)

    shape = (options.number_samples, options.number_chains)
    states = torch.empty(shape, dtype=torch.int64)
    log_prob = torch.empty(shape, dtype=torch.float32)
    for i, (x, _, y) in enumerate(
        islice(
            sampler,
            options.number_discarded * basis.number_spins,
            (options.number_discarded + options.number_samples) * basis.number_spins,
            basis.number_spins,
        )
    ):
        states[i] = x
        log_prob[i] = y
    return states, log_prob


def closeness_testing_l1(p, q, n, C, m):
    from math import sqrt

    def sample(fn, n, number_samples):
        counts = np.zeros(n)
        for _ in range(number_samples):
            counts[fn() - 1] += 1
        return counts

    def experiment(number_samples):
        Z = sum(
            ((x - y) ** 2 - x - y) / (x + y)
            for x, y in zip(sample(p, n, number_samples), sample(q, n, number_samples))
            if x + y > 0
        )
        print(Z)
        return Z <= C * sqrt(number_samples)

    results = [experiment(M) for M in np.random.poisson(m, size=10)]
    return np.array(results)


def benchmark():
    from itertools import islice
    from nqs_playground._C_nqs.v2 import SpinBasis, MetropolisKernel

    basis = SpinBasis([], number_spins=8, hamming_weight=4)
    basis.build()
    prob = torch.rand(basis.number_states)
    prob /= torch.sum(prob)

    def log_prob_fn(state):
        return 0.5 * torch.log(torch.tensor([prob[basis.index(x)] for x in state]))

    kernel = MetropolisKernel(basis)
    sampler = Sampler(kernel, log_prob_fn, 1)
    for _ in islice(sampler, 1000, 100000, 2 * basis.number_spins):
        pass


def test_uniform():
    from itertools import islice
    from nqs_playground._C_nqs.v2 import SpinBasis, MetropolisKernel

    basis = SpinBasis([], number_spins=10, hamming_weight=5)
    basis.build()

    class Target:
        def __init__(self):
            self.target_prob = torch.rand(basis.number_states)
            self.target_prob /= torch.sum(self.target_prob)
            self.buffer = None
            self._i = 0

        def log_prob(self, state):
            p = torch.tensor(
                [self.target_prob[basis.index(x)] for x in state], dtype=torch.float32
            )
            return torch.log(p)

        def __call__(self):
            if self._i == 0:
                self._fill()
            self._i -= 1
            return self.buffer[self._i]

        def _fill(self):
            # +1 Here is to ensure that we count from 1
            self.buffer = (
                torch.multinomial(self.target_prob, num_samples=10240, replacement=True)
                + 1
            )
            self._i = len(self.buffer)

    q = Target()
    sampler = islice(
        Sampler(MetropolisKernel(basis), lambda x: q.log_prob(x), batch_size=1),
        500 * basis.number_spins,
        1000000000,
        basis.number_spins,
    )

    def p():
        state, _, _ = next(sampler)
        return basis.index(state.item()) + 1

    return closeness_testing_l1(p, q, basis.number_states, 1.0, 10000)


def test():
    from nqs_playground._C_nqs.v2 import SpinBasis

    basis = SpinBasis([], 10, 5)
    m = torch.nn.Sequential(
        torch.nn.Linear(10, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
    options = SamplingOptions(number_samples=100, number_chains=4)
    return _sample_using_metropolis(m, basis, options)


@torch.jit.script
def _step(
    current: Tuple[Tensor, Tensor, Tensor], proposed: Tuple[Tensor, Tensor, Tensor]
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Internal function of the Sampler."""
    state, norm, log_prob = current
    proposed_state, proposed_norm, proposed_log_prob = proposed
    r = torch.where(
        proposed_norm > 0,
        torch.rand(state.size(0)) * proposed_norm / norm,
        torch.tensor(math.inf, dtype=torch.float32),
    )
    t = r <= torch.exp(proposed_log_prob - log_prob)
    return (
        torch.where(t, proposed_state, state),
        torch.where(t, proposed_norm, norm),
        torch.where(t, proposed_log_prob, log_prob),
    )


def _random_spin_configuration(n: int, hamming_weight: Optional[int] = None) -> int:
    assert 0 <= n and n <= 64, "invalid number of spins"
    if hamming_weight is not None:
        assert 0 <= hamming_weight and hamming_weight <= n, "invalid hamming weight"
        bits = ["1"] * hamming_weight + ["0"] * (n - hamming_weight)
        np.random.shuffle(bits)
        return int("".join(bits), base=2)
    else:
        return int(np.random.randint(0, 1 << n, dtype=np.uint64))


def _prepare_initial_state(basis, batch_size: int) -> torch.Tensor:
    r"""Generates a batch of valid spin configurations (i.e. representatives).
    """
    if batch_size <= 0:
        raise ValueError(
            "invalid batch size: {}; expected a positive integer".format(batch_size)
        )
    if basis.number_spins > 64:
        raise ValueError("such long spin configurations are not yet supported")
    # First, we generate a bunch of representatives, and then sample uniformly
    # from them.
    states = set()
    for _ in range(max(2 * batch_size, 10000)):
        spin = _random_spin_configuration(basis.number_spins, basis.hamming_weight)
        states.add(basis.representative(spin))
    states = np.array(list(states), dtype=np.uint64)
    batch = np.random.choice(states, size=batch_size, replace=True)
    return torch.from_numpy(batch.view(np.int64))
