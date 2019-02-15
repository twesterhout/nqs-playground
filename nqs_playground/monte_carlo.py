#!/usr/bin/env python3

# Copyright Tom Westerhout (c) 2019
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of Tom Westerhout nor the names of other
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cmath
import collections
# from copy import deepcopy
# import cProfile
# import importlib
from itertools import islice
from functools import reduce
import logging
import math
# import os
# import sys
# import time
from typing import Dict, List, Tuple, Optional

# import click
# import mpmath  # Just to be safe: for accurate computation of L2 norms
from numba import jit, jitclass, uint8, int64, uint64, float32
# from numba.types import Bytes
# import numba.extending
import numpy as np
import scipy.special
# from scipy.sparse.linalg import lgmres, LinearOperator
import torch
# import torch.nn as nn
# import torch.nn.functional as F

from .core import CompactSpin, MonteCarloState, Machine, random_spin, normalisation_constant




@jitclass([("_ups", int64[:]), ("_downs", int64[:]), ("_n", int64), ("_i", int64)])
class _Flipper(object):
    """
    Magnetisation-preserving spin flipper.
    """

    def __init__(self, spin: np.ndarray):
        """
        Initialises the flipper with the given spin. Magnetisation is deduced
        from the spin and is kept constant.
        """
        self._ups = np.where(spin == 1.0)[0]
        self._downs = np.where(spin != 1.0)[0]
        self._n = min(self._ups.size, self._downs.size)
        self._i = 0
        if self._i >= self._n:
            raise ValueError("Failed to initialise the Flipper.")

    def read(self) -> List[int]:
        """
        Suggests the next spins to flip.
        """
        return [self._ups[self._i], self._downs[self._i]]

    def next(self, accepted: bool):
        """
        Updates the internal state.

        :param bool accepted: Specifies whether the last proposed flips were
        accepted.
        """
        if accepted:
            i = self._i
            t = self._ups[i]
            self._ups[i] = self._downs[i]
            self._downs[i] = t
        self._i += 1
        if self._i == self._n:
            self._i = 0
            np.random.shuffle(self._ups)
            np.random.shuffle(self._downs)


class MetropolisMarkovChain(object):
    """
    Markov chain constructed using Metropolis-Hasting algorithm. Elements of
    the chain are ``MonteCarloState``s.
    """

    def __init__(self, machine: Machine, spin: np.ndarray):
        """
        Initialises the Markov chain.

        :param Machine machine: The variational state
        :param np.ndarray spin: Initial spin configuration
        """
        self._flipper = _Flipper(spin)
        self._machine = machine
        self._spin = spin
        self._log_wf = self._machine.log_wf(self._spin)
        self._steps = 0
        self._accepted = 0

    def __iter__(self):
        def do_generate():
            while True:
                self._steps += 1
                yield MonteCarloState(
                    weight=1.0, spin=self._spin, machine=self._machine
                )

                flips = self._flipper.read()
                self._spin[flips] *= -1
                new_log_wf = self._machine.log_wf(self._spin)
                if min(
                    1.0, math.exp((new_log_wf - self._log_wf).real) ** 2
                ) > np.random.uniform(0, 1):
                    self._accepted += 1
                    self._log_wf = new_log_wf
                    self._flipper.next(True)
                else:
                    # Revert to the previous state
                    self._spin[flips] *= -1
                    self._flipper.next(False)

        return do_generate()

    @property
    def steps(self) -> int:
        """
        :return: number of steps performed till now.
        """
        return self._steps

    @property
    def accepted(self) -> int:
        """
        :return: number of transitions accepted till now.
        """
        return self._accepted


def perm_unique(elements):
    """
    Returns all unique permutations of ``elements``.

    .. note::

       The algorithm is taken from https://stackoverflow.com/a/6285203.
    """

    class UniqueElement(object):
        def __init__(self, value, count):
            self.value = value
            self.count = count

    def _helper(list_unique, result_list, depth):
        if depth < 0:
            yield tuple(result_list)
        else:
            for i in list_unique:
                if i.count > 0:
                    result_list[depth] = i.value
                    i.count -= 1
                    for g in _helper(list_unique, result_list, depth - 1):
                        yield g
                    i.count += 1

    list_unique = [UniqueElement(i, elements.count(i)) for i in set(elements)]
    n = len(elements)
    return _helper(list_unique, [0] * n, n - 1)


def all_spins(n: int, m: Optional[int]) -> torch.Tensor:
    if m is not None:
        n_ups = (n + m) // 2
        n_downs = (n - m) // 2
        size = int(scipy.special.comb(n, n_ups))
        spins = torch.empty((size, n), dtype=torch.float32)
        for i, s in enumerate(
            map(
                lambda x: torch.tensor(x, dtype=torch.float32).view(1, -1),
                perm_unique([1] * n_ups + [-1] * n_downs),
            )
        ):
            spins[i, :] = s
        return spins
    else:
        raise NotImplementedError()


def sample_one(
    ψ,
    steps,
    magnetisation: Optional[int] = None,
    unique: bool = True,
    convert: bool = True,
    chain_fn=MetropolisMarkovChain,
):
    """
    Samples some spin configurations from the probability distribution defined by ψ.
    
    :param Machine ψ: NQS state to sample
    :param steps: How many steps to perform: ``(start, stop, step)``.
    :param magnetisation: Magnetisation of the system.
    :param unique: If true, each configuration will appear at most once.
    :param convert:
        May be used in combination with ``unique``. If true, the samples will
        be converted to ``torch.FloatTensor``. Otherwise a ``Set[Spin]`` will
        be returned.
    :param chain_fn:
        A function which, given a ``Machine`` and initial spin, creates the Markov Chain.
    """
    if not isinstance(ψ, Machine):
        raise TypeError("Expected a 'Machine', but got '{}'".format(type(ψ)))
    number_spins = ψ.number_spins
    if unique:
        samples_set = set()

        def store(_, s):
            samples_set.add(CompactSpin(s))

    else:
        # NOTE: len(range(*steps)) is a hack, but it works :)
        number_samples = len(range(*steps))
        samples = torch.empty((number_samples, number_spins), dtype=torch.float32)

        def store(i, s):
            samples[i, :] = torch.from_numpy(s)

    for i, s in enumerate(
        map(
            lambda state: state.spin,
            islice(chain_fn(ψ, random_spin(number_spins, magnetisation)), *steps),
        )
    ):
        store(i, s)

    if unique:
        if not convert:
            return samples_set
        samples = torch.empty((len(samples_set), number_spins), dtype=torch.float32)
        for i, s in enumerate(samples_set):
            samples[i, :] = torch.from_numpy(s.numpy())
    return samples


def sample_some(
    ψ, steps: Tuple[int, int, int, int], magnetisation: Optional[int] = None
) -> torch.FloatTensor:
    """
    Samples some spin configurations. The returned tensor contains no duplicates.

    :param ψ: NQS state to sample. Should be either a ``Machine`` or a ``torch.nn.Module``.
    :param steps: ``(number_chains, start, stop, step)``.
    """
    number_chains, monte_carlo_steps = steps[0], steps[1:]
    machine = ψ if isinstance(ψ, Machine) else Machine(ψ)
    samples_set = set.union(
        *tuple(
            (
                sample_one(
                    machine,
                    steps=monte_carlo_steps,
                    magnetisation=magnetisation,
                    unique=True,
                    convert=False,
                )
                for _ in range(number_chains)
            )
        )
    )
    samples = torch.empty((len(samples_set), ψ.number_spins), dtype=torch.float32)
    for i, s in enumerate(samples_set):
        samples[i, :] = torch.from_numpy(s.numpy())
    return samples


def _sample_explicit(ψ, H, magnetisation, requires_energy, requires_grad):
    """
    """
    start = time.time()
    samples = all_spins(ψ.number_spins, magnetisation)
    ψ = ψ if isinstance(ψ, Machine) else Machine(ψ)
    result = _MonteCarloResult(
        energies=torch.empty(
            (samples.size(0), 2), dtype=torch.float32, requires_grad=False
        )
        if requires_energy
        else None,
        gradients=torch.empty(
            (samples.size(0), 2 * ψ.size), dtype=torch.float32, requires_grad=False
        )
        if requires_grad
        else None,
        weights=torch.empty(
            (samples.size(0),), dtype=torch.float32, requires_grad=False
        ),
        samples=samples,
    )

    with torch.no_grad():
        ψ_s = ψ.ψ(samples)
        scale = normalisation_constant(ψ_s[:, 0])

    ψ._cache = dict(
        zip(
            map(lambda s: Spin(s.numpy()), samples),
            map(
                lambda t: Machine.Cell(log_wf=complex(t[0], t[1]), der_log_wf=None), ψ_s
            ),
        )
    )
    if requires_energy:
        energies = result.energies.view(-1).numpy().view(dtype=np.complex64)
        for i in range(samples.size(0)):
            energies[i] = H(
                MonteCarloState(machine=ψ, spin=samples[i].numpy(), weight=None),
                cutoff=None,
            )

    if requires_grad:
        # TODO(twesterhout): This should be done in batches,
        # but I'm yet to figure out how to accomplish it in PyTorch.
        gradients = result.gradients.numpy().view(dtype=np.complex64)
        for i in range(samples.size(0)):
            ψ.der_log_wf(samples[i], out=gradients[i, :])

    with torch.no_grad():
        ψ_s[:, 0] += scale
        ψ_s[:, 0] *= 2
        torch.exp(ψ_s[:, 0], out=result.weights)

    finish = time.time()
    stats = _MonteCarloStats(
        acceptance=1.0, dimension=samples.size(0), time=finish - start
    )
    return result, stats


def _sample_monte_carlo_one_impl(ψ, H, initial_spin, steps, result):
    number_steps = len(range(*steps))
    samples = torch.empty(
        (number_steps, ψ.number_spins), dtype=torch.float32, requires_grad=False
    )
    with torch.no_grad():
        chain = MetropolisMarkovChain(ψ, initial_spin)
        for i, state in enumerate(islice(chain, *steps)):
            samples[i, :] = torch.from_numpy(state.spin)
        result.samples = samples

        if result.energies is not None:
            energies = result.energies.view(-1).numpy().view(dtype=np.complex64)
            energies_cache = {}
            for i in range(samples.size(0)):
                spin = samples[i].numpy()
                compact_spin = Spin.from_array(spin)
                e_loc = energies_cache.get(compact_spin)
                if e_loc is None:
                    e_loc = H(MonteCarloState(machine=ψ, spin=spin, weight=None))
                    energies_cache[compact_spin] = e_loc
                energies[i] = e_loc

    if result.gradients is not None:
        gradients = result.gradients.numpy().view(dtype=np.complex64)
        for i in range(samples.size(0)):
            ψ.der_log_wf(samples[i], out=gradients[i, :])

    return _MonteCarloStats(
        acceptance=chain.accepted / chain.steps, dimension=len(energies_cache) if result.energies is not None else 0
    )


def _sample_monte_carlo_one(
    ψ, H, steps, magnetisation, requires_energy, requires_grad, restarts=5
):
    start = time.time()
    number_steps = len(range(*steps))
    ψ = ψ if isinstance(ψ, Machine) else Machine(ψ)
    result = _MonteCarloResult(
        energies=torch.empty(
            (number_steps, 2), dtype=torch.float32, requires_grad=False
        )
        if requires_energy
        else None,
        gradients=torch.empty(
            (number_steps, 2 * ψ.size), dtype=torch.float32, requires_grad=False
        )
        if requires_grad
        else None,
        # NOTE(twesterhout): Weights for simple sampling of |ψ|² are all 1/N.
        weights=1.0
        / number_steps
        * torch.ones((number_steps,), dtype=torch.float32, requires_grad=False),
    )

    initial_spin = random_spin(ψ.number_spins, magnetisation)
    stats = None
    while stats is None:
        try:
            stats = _sample_monte_carlo_one_impl(ψ, H, initial_spin, steps, result)
        except WorthlessConfiguration as err:
            if restarts > 0:
                logging.warning("Restarting the Monte Carlo loop...")
                restarts -= 1
                spin[err.suggestion] *= -1
            else:
                raise
    finish = time.time()
    stats.time = finish - start
    return result, stats


def sample_state(
    ψ,
    H=None,
    steps=None,
    magnetisation=None,
    explicit=False,
    requires_energy=False,
    requires_grad=False,
):
    if explicit:
        return _sample_explicit(
            ψ,
            H=H,
            magnetisation=magnetisation,
            requires_energy=requires_energy,
            requires_grad=requires_grad,
        )
    else:
        assert steps is not None
        ψ = ψ if isinstance(ψ, Machine) else Machine(ψ)
        outputs = [
            _sample_monte_carlo_one(
                ψ,
                H,
                steps=steps[1:],
                magnetisation=magnetisation,
                requires_energy=requires_energy,
                requires_grad=requires_grad,
            )
            for _ in range(steps[0])
        ]
        result = _MonteCarloResult(
            energies=torch.cat(tuple((x.energies for (x, _) in outputs)), dim=0)
            if requires_energy
            else None,
            gradients=torch.cat(tuple((x.gradients for (x, _) in outputs)), dim=0)
            if requires_grad
            else None,
            weights=torch.cat(tuple((x.weights for (x, _) in outputs)), dim=0),
            samples=torch.cat(tuple((x.samples for (x, _) in outputs)), dim=0),
        )
        # Rescaling the weights
        result.weights *= 1.0 / len(outputs)
        stats = _MonteCarloStats(
            acceptance=[x.acceptance for (_, x) in outputs],
            dimension=[x.dimension for (_, x) in outputs],
            time=[x.time for (_, x) in outputs],
        )
        return result, stats



