#!/usr/bin/env python3

# Copyright Tom Westerhout (c) 2018
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
import cProfile
import importlib
from itertools import islice
from functools import reduce
import logging
import math
import os
import sys
import time
from typing import Dict, List, Tuple, Optional

import click
from numba import jit, jitclass, uint8, int64, float32
import numpy as np
import scipy
from scipy.sparse.linalg import lgmres, LinearOperator
import torch
import torch.nn as nn
import torch.nn.functional as F


@jit(uint8[:](float32[:]), nopython=True)
def to_bytes(spin: np.ndarray) -> np.ndarray:
    """
    Converts a spin to a bit array. It is assumed that a spin-up corresponds to
    1.0.
    """
    chunks, rest = divmod(spin.size, 8)
    b = np.empty(chunks + int(rest > 0), dtype=np.uint8)
    if rest != 0:
        b[0] = spin[0] == 1.0
        for i in range(1, rest):
            b[0] = (b[0] << 1) | (spin[i] == 1.0)
    for i in range(chunks):
        j = 8 * i + rest
        b[int(rest > 0) + i] = ((spin[j + 0] == 1.0) << 7) \
                    | ((spin[j + 1] == 1.0) << 6) \
                    | ((spin[j + 2] == 1.0) << 5) \
                    | ((spin[j + 3] == 1.0) << 4) \
                    | ((spin[j + 4] == 1.0) << 3) \
                    | ((spin[j + 5] == 1.0) << 2) \
                    | ((spin[j + 6] == 1.0) << 1) \
                    | ((spin[j + 7] == 1.0) << 0)
    return b


class CompactSpin(bytes):
    """
    Compact representation of a spin.
    """
    def __new__(cls, spin: np.ndarray):
        """
        Creates a new ``CompactSpin`` given the spin (ℝⁿ).
        """
        return bytes.__new__(cls, to_bytes(spin).tobytes())

    def __int__(self):
        """
        Returns an int representation of the spin.
        """
        return int.from_bytes(self, byteorder='big')


def _make_machine(BaseNet):
    """
    Creates the ``Machine`` class by deriving from a user-defined Neural
    Network ``BaseNet``.
    """
    class Machine(BaseNet):
        """
        Our variational ansatz |Ψ〉.
        """
        class Cell(object):
            """
            Cache cell corresponding to a spin configuration |S〉. A cell stores
            log(〈S|Ψ〉) and ∂log(〈S|Ψ〉)/∂W where W are the variational
            parameters.

            :param complex wave_function: log(〈S|Ψ〉).
            :param gradient: ∇log(〈S|Ψ〉).
            :type gradient: np.ndarray of float32 or None.
            """
            def __init__(self, wave_function: complex,
                         gradient: Optional[np.ndarray] = None):
                self.log_wf = wave_function
                self.der_log_wf = gradient

        def __init__(self, n_spins: int):
            """
            Initialises the state with random values for the variational parameters.

            :param int n_spins: Number of spins in the system.
            """
            if n_spins <= 0:
                raise ValueError("Invalid number of spins: {}".format(n_spins))
            super().__init__(n_spins)
            self._size = sum(map(lambda p: reduce(int.__mul__, p.size()), self.parameters()))
            # Hash-table mapping CompactSpin to Machine.Cell
            self._cache = {}

        def log_wf(self, x: np.ndarray) -> complex:
            """
            Computes log(Ψ(x)).

            :param np.ndarray x: Spin configuration. Must be a numpy array of
                                 ``float32``.
            :return: log(Ψ(x))
            :rtype: complex
            """
            key = CompactSpin(x)
            cell = self._cache.get(key)
            if cell is not None:
                return cell.log_wf
            else:
                with torch.no_grad():
                    (a, b) = self.forward(torch.from_numpy(x))
                    log_wf = complex(a, b)
                    self._cache[key] = Machine.Cell(log_wf)
                    return log_wf

        @property
        def size(self) -> int:
            """
            Returns the number of variational parameters.
            """
            return self._size

        def der_log_wf(self, x: np.ndarray, out: np.ndarray = None,
                       key: Optional[CompactSpin] = None) -> np.ndarray:
            """
            Computes ∇log(Ψ(x)).

            :param np.ndarray x:   Spin configuration. Must be a numpy array of ``float32``.
            :param np.ndarray out: Destination array. Must be a numpy array of ``complex64``.
            :param key: Precomputed ``CompactSpin``-representation of x.
            :type key: CompactSpin or None.
            :return: ∇log(Ψ(x)) as a numpy array of ``complex64``.
                     __Don't you dare modify it!__.
            """
            if key is None:
                key = CompactSpin(x)
            # If out is not given, allocate a new array
            if out is None:
                out = np.empty((self.size,), dtype=np.complex64)
            cell = self._cache.get(key)
            if cell is not None and cell.der_log_wf is not None:
                # Copy already known gradient
                out[:] = cell.der_log_wf
            else:
                # Forward-propagation to construct the graph
                result = self.forward(torch.from_numpy(x))
                # Computes ∇Re[log(Ψ(x))]
                self.zero_grad()
                result.backward(torch.tensor([1, 0], dtype=torch.float32),
                                retain_graph=True)
                # TODO(twesterhout): This is ugly and error-prone.
                i = 0
                for p in map(lambda p_: p_.grad.view(-1).numpy(), self.parameters()):
                    out.real[i:i + p.size] = p
                    i += p.size
                # Computes ∇Im[log(Ψ(x))]
                self.zero_grad()
                result.backward(torch.tensor([0, 1], dtype=torch.float32))
                # TODO(twesterhout): This is ugly and error-prone.
                i = 0
                for p in map(lambda p_: p_.grad.view(-1).numpy(), self.parameters()):
                    out.imag[i:i + p.size] = p
                    i += p.size
                # Save the results
                # TODO(twesterhout): Remove the copy when it's safe to do so.
                self._cache[key] = Machine.Cell(complex(result[0].item(), result[1].item()),
                                                np.copy(out))
            return out

        def clear_cache(self):
            """
            Clears the internal cache. This function must be called when the
            variational parameters are updated.
            """
            self._cache = {}

        def set_gradients(self, x: np.ndarray):
            """
            Performs ∇W = x, i.e. sets the gradients of the variational parameters.

            :param np.ndarray x: New value for ∇W. Must be a numpy array of
            ``float32`` of length ``self.size``.
            """
            with torch.no_grad():
                gradients = torch.from_numpy(x)
                i = 0
                for dp in map(lambda p_: p_.grad.data.view(-1), self.parameters()):
                    (n,) = dp.size()
                    dp.copy_(gradients[i:i + n])
                    i += n

        def __isub__(self, x: np.ndarray):
            """
            In-place subtracts ``x`` from the parameters. This is useful when
            implementing optimizers by hand.

            :param np.ndarray x: A numpy array of length ``self.size`` of ``complex64``.
            """
            with torch.no_grad():
                delta = torch.from_numpy(x)
                i = 0
                for p in map(lambda p_: p_.data.view(-1), self.parameters()):
                    (n,) = p.size()
                    p.add_(-1, delta[i:i + n])
                    i += n
            # Changing the weights invalidates the cache.
            self._cache = {}
            return self
    return Machine


class MonteCarloState(object):
    """
    Monte-Carlo state keeps track of the current variational state and spin
    configuration.
    """

    def __init__(self, machine, spin):
        """
        Initialises the Monte-Carlo state.

        :param machine: Variational state
        :param np.ndarray spin: Initial spin configuration
        """
        self._machine = machine
        self._spin = np.copy(spin)
        self._log_wf = self._machine.log_wf(self._spin)
        # TODO(twesterhout): Remove this.
        # with torch.no_grad():
        #     for p in self._machine.parameters():
        #         logging.info('{}'.format(torch.norm(p.data)))

    @property
    def spin(self) -> np.ndarray:
        """
        Returns the current spin configuration.
        """
        return self._spin

    @property
    def machine(self):
        return self._machine

    def log_wf(self) -> complex:
        """
        Returns log(〈S|ψ〉) where S is the current spin configuration.
        """
        return self._log_wf

    def log_quot_wf(self, flips: List[int]) -> complex:
        """
        Calculates log(〈S'|Ψ〉/ 〈S|Ψ〉) where S is the current spin
        configuration and S' is obtained from S by flipping spins indicated by
        ``flips``.
        """
        # TODO(twesterhout): Yes, this is ugly, but it does avoid copying :)
        self._spin[flips] *= -1
        new_log_wf = self._machine.log_wf(self._spin)
        self._spin[flips] *= -1
        return new_log_wf - self.log_wf()

    def der_log_wf(self):
        return self._machine.der_log_wf(self._spin)

    def update(self, flips: List[int]):
        """
        "Accepts" the flips.
        """
        self._spin[flips] *= -1
        self._log_wf = self._machine.log_wf(self._spin)
        return self


@jitclass([
    ('_ups', int64[:]),
    ('_downs', int64[:]),
    ('_n', int64),
    ('_i', int64)
])
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
            raise ValueError('Failed to initialise the Flipper.')

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


class MetropolisMC(object):
    """
    Markov chain constructed using Metropolis-Hasting algorithm. Elements of
    the chain are ``MonteCarloState``s.
    """
    def __init__(self, machine, spin: np.ndarray):
        """
        Initialises a Markov chain.

        :param machine: The variational state
        :param np.ndarray spin: Initial spin configuration
        """
        self._state = MonteCarloState(machine, spin)
        self._flipper = _Flipper(spin)
        self._steps = 0
        self._accepted = 0

    def __iter__(self):
        def do_generate():
            while True:
                self._steps += 1
                yield self._state
                flips = self._flipper.read()
                if min(1.0, math.exp(self._state.log_quot_wf(flips).real)**2) \
                        > np.random.uniform(0, 1):
                    self._accepted += 1
                    self._state.update(flips)
                    self._flipper.next(True)
                else:
                    self._flipper.next(False)

        return do_generate()


class WorthlessConfiguration(Exception):
    def __init__(self, flips):
        super().__init__("The current spin configuration has too low a weight.")
        self.suggestion = flips


class Heisenberg(object):
    """
    Isotropic Heisenberg Hamiltonian on a lattice.
    """
    def __init__(self, edges: List[Tuple[int, int]]):
        """
        Initialises the Hamiltonian given a list of edges.
        """
        self._graph = edges
        smallest = min(map(min, edges))
        largest = max(map(max, edges))
        if smallest != 0:
            ValueError('Invalid graph: Counting from 0, but the minimal index '
                       'present is {}.'.format(smallest))
        self._number_spins = largest + 1

    def __call__(self, state: MonteCarloState) -> np.complex64:
        """
        Calculates local energy in the given state.
        """
        spin = state.spin
        energy = 0
        for (i, j) in self._graph:
            if spin[i] == spin[j]:
                energy += 1
            else:
                assert spin[i] == -spin[j]
                x = state.log_quot_wf([i, j])
                if x.real > 5.5:
                    raise WorthlessConfiguration([i, j])
                energy += -1 + 2 * cmath.exp(x)
        return np.complex64(energy)

    def reachable_from(self, spin):
        reachable = []
        for (i, j) in filter(lambda x: spin[x[0]] != spin[x[1]], self._graph):
            assert spin[i] == -spin[j]
            reachable.append(spin.copy())
            reachable[-1][[i, j]] *= -1
        return reachable

    @property
    def number_spins(self) -> int:
        return self._number_spins


def monte_carlo_loop(machine, hamiltonian, initial_spin, steps):
    """
    Runs the Monte-Carlo simulation.

    :return: (all gradients, mean gradient, mean local energy, force)
    """
    derivatives = []
    energies = []
    energies_cache = {}
    chain = MetropolisMC(machine, initial_spin)
    for state in islice(chain, *steps):
        derivatives.append(state.der_log_wf())
        spin = CompactSpin(state.spin)
        e_loc = energies_cache.get(spin)
        if e_loc is None:
            e_loc = hamiltonian(state)
            energies_cache[spin] = e_loc
        energies.append(e_loc)
    derivatives = np.array(derivatives, dtype=np.complex64)
    energies = np.array(energies, dtype=np.complex64)
    mean_O = np.mean(derivatives, axis=0)
    mean_E = np.mean(energies)
    force = np.mean(energies * derivatives.conj().transpose(), axis=1)
    force -= mean_O.conj() * mean_E
    logging.info('Subspace dimension: {}'.format(len(energies_cache)))
    logging.info('Acceptance rate: {:.2f}%'.format(chain._accepted / chain._steps * 100))
    return derivatives, mean_O, mean_E, force


def monte_carlo_loop_for_lanczos(machine, hamiltonian, initial_spin, steps):
    logging.info('Running Monte Carlo...')
    energies = []
    energies_cache = {}
    wave_function = {}
    chain = MetropolisMC(machine, initial_spin)
    for state in islice(chain, *steps):
        spin = CompactSpin(state.spin)
        e_loc = energies_cache.get(spin)
        if e_loc is None:
            e_loc = hamiltonian(state)
            energies_cache[spin] = e_loc
        energies.append(e_loc)
        wave_function[spin] = cmath.exp(state.log_wf())
        for s in hamiltonian.reachable_from(state.spin):
            wave_function[CompactSpin(s)] = cmath.exp(state.machine.log_wf(s))
    energies = np.array(energies, dtype=np.complex64)
    mean_E = np.mean(energies)
    logging.info('Subspace dimension: {}'.format(len(wave_function)))
    logging.info('Acceptance rate: {:.2f}%'.format(chain._accepted / chain._steps * 100))
    logging.info('E = {}'.format(mean_E))
    return mean_E, wave_function


def monte_carlo(machine, hamiltonian, initial_spin, steps):
    logging.info('Running Monte-Carlo...')
    start = time.time()
    restarts = 5
    spin = np.copy(initial_spin)
    answer = None
    while answer is None:
        try:
            answer = monte_carlo_loop(machine, hamiltonian, spin, steps)
        except WorthlessConfiguration as err:
            if restarts > 0:
                logging.warning('Restarting the Monte-Carlo simulation...')
                restarts -= 1
                spin[err.suggestion] *= -1
            else:
                raise
    finish = time.time()
    logging.info('Done in {:.2f} seconds!'.format(finish - start))
    return answer


class Covariance(LinearOperator):
    """
    Covariance matrix matrix S.
    """
    def __init__(self, gradients, mean_gradient, regulariser):
        """
        """
        (steps, n) = gradients.shape
        super().__init__(np.float32, (n, n))
        self._gradients = gradients - mean_gradient
        self._conj_gradients = self._gradients.transpose().conj()
        self._lambda = regulariser
        self._scale = 1 / steps

    def _S(self, x: np.ndarray):
        assert x.dtype == np.complex64
        y = np.dot(self._gradients, x)
        z = np.dot(self._conj_gradients, y)
        z *= self._scale
        return z

    def _matvec(self, x):
        """
        Computes
        +-----------------+ +-------+ +-+     +-+
        | Re[S]^T Im[S]^T | | Re[S] | |x|  + λ|x|
        +-----------------+ |       | | |     | |
                            | Im[S] | +-+     +-+
                            +-------+
        =
        Re[S]^T Re[S] x + Im[S]^T Im[S] x
        =
        Re[S] Re[S] x - Im[S] Im[S] x
        =
        Re[SSx]
        """
        assert x.dtype == self.dtype
        return np.ascontiguousarray(
            self._S(
                self._S(
                    np.ascontiguousarray(x, dtype=np.complex64)
                )
            ).real + self._lambda * x,
            dtype=np.float32)

    def solve(self, b, x0 = None):
        """
        Solves

        +-----------------+ +-------+ +-+     +-----------------+ +-------+ +-+
        | Re[S]^T Im[S]^T | | Re[S] | |x|     | Re[S]^T Im[S]^T | | Re[b] | |F|
        +-----------------+ |       | | |  =  +-----------------+ |       | | |
                            | Im[S] | +-+                         | Im[b] | +-+
                            +-------+                             +-------+
        """
        assert b.dtype == np.complex64
        start = time.time()
        logging.info('Calculating S⁻¹F...')
        b_ = np.ascontiguousarray(self._S(b).real)
        x, info = scipy.sparse.linalg.lgmres(self, b_, x0)
        finish = time.time()
        if info == 0:
            logging.info('Done in {:.2f} seconds!'.format(finish - start))
            return x
        if info > 0:
            logging.error('Failed to converge')
            return 0.1 * b_
        raise ValueError('The hell has just happened?')


def random_spin(n, magnetisation=None):
    if n <= 0:
        raise ValueError('Invalid number of spins: {}'.format(n))
    if magnetisation is not None:
        if abs(magnetisation) > n:
            raise ValueError(
                'Magnetisation exceeds the number of spins: |{}| > {}'
                .format(magnetisation, n))
        if (n + magnetisation) % 2 != 0:
            raise ValueError('Invalid magnetisation: {}'.format(magnetisation))
        number_ups = (n + magnetisation) // 2
        number_downs = (n - magnetisation) // 2
        spin = np.empty((n,), dtype=np.float32)
        for i in range(number_ups):
            spin[i] = 1.0
        assert len(range(number_ups, n)) == number_downs
        for i in range(number_ups, n):
            spin[i] = -1.0
        np.random.shuffle(spin)
        assert int(spin.sum()) == magnetisation
        return spin
    else:
        return np.random.choice([np.float32(-1.0), np.float32(1.0)], size=n)


class Optimiser(object):
    def __init__(self,
                 machine,
                 hamiltonian,
                 magnetisation,
                 epochs,
                 monte_carlo_steps,
                 learning_rate,
                 use_sr,
                 regulariser):
        self._machine = machine
        self._hamiltonian = hamiltonian
        self._magnetisation = magnetisation
        self._epochs = epochs
        self._monte_carlo_steps = monte_carlo_steps
        self._learning_rate = learning_rate
        self._use_sr = use_sr
        if use_sr:
            self._regulariser = regulariser
            self._delta = None
            self._optimizer = torch.optim.Adam(self._machine.parameters(),
                                               lr=self._learning_rate)
        else:
            self._optimizer = torch.optim.SGD(self._machine.parameters(),
                                              lr=self._learning_rate)

    def learning_cycle(self, iteration):
        logging.info('==================== {} ===================='.format(iteration))
        # Monte Carlo
        spin = random_spin(self._machine.number_spins, self._magnetisation)
        (Os, mean_O, E, F) = \
            monte_carlo(self._machine, self._hamiltonian, spin,
                             self._monte_carlo_steps)
        logging.info('E = {}'.format(E))
        # Calculate the "true" gradients
        if self._use_sr:
            # We also cache δ to use it as a guess the next time we're computing
            # S⁻¹F.
            self._delta = Covariance(
                Os, mean_O, self._regulariser(iteration)).solve(F, x0=self._delta)
            self._machine.set_gradients(self._delta)
            logging.info('∥F∥₂ = {}, ∥δ∥₂ = {}'
                         .format(np.linalg.norm(F), np.linalg.norm(self._delta)))
        else:
            self._machine.set_gradients(F.real)
            logging.info('∥F∥₂ = {}, ∥Re[F]∥₂ = {}'
                         .format(np.linalg.norm(F), np.linalg.norm(F.real)))
        # Update the variational parameters
        self._optimizer.step()
        self._machine.clear_cache()

    def __call__(self):
        for i in range(self._epochs):
            self.learning_cycle(i)
        return self._machine


def heisenberg6():
    hamiltonian = Heisenberg([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
    machine = Machine(6)
    return hamiltonian


def kagome12():
    # Generated using tipsi, Kagome(2, 2) with periodic boundary conditions.
    edges = [(0, 1), (0, 2), (0, 4), (0, 8),
             (1, 2), (1, 3), (1, 11),
             (2, 6), (2, 10),
             (3, 4), (3, 5), (3, 11),
             (4, 5), (4, 8),
             (5, 7), (5, 9),
             (6, 7), (6, 8), (6, 10),
             (7, 8), (7, 9),
             (9, 10), (9, 11),
             (10, 11)]
    hamiltonian = Heisenberg(edges)
    return hamiltonian


def heisenberg3x3():
    hamiltonian = Heisenberg([(0, 1), (1, 2), (2, 0),
                              (3, 4), (4, 5), (5, 3),
                              (6, 7), (7, 8), (8, 6),
                              (0, 3), (3, 6), (6, 0),
                              (1, 4), (4, 7), (7, 1),
                              (2, 5), (5, 8), (8, 2)])
    return hamiltonian


def import_network(nn_file):
    module_name, extension = os.path.splitext(os.path.basename(nn_file))
    module_dir = os.path.dirname(nn_file)
    if extension != '.py':
        raise ValueError(
            'Could not import the network from {}: not a python source file.')
    # Insert in the beginning
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    sys.path.pop(0)
    return module.Net


@click.group()
def cli():
    pass


@cli.command()
@click.argument('nn-file',
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar='<arch_file>')
@click.option('-i', '--in-file',
    type=click.File(mode='rb'),
    required=True,
    help='File containing the Neural Network weights as a PyTorch `state_dict` '
         'serialised using `torch.save`. It is up to the user to ensure that '
         'the weights are compatible with the architecture read from '
         '<arch_file>.')
@click.option('-o', '--out-file',
    type=click.File(mode='w'),
    default=sys.stdout,
    show_default=True,
    help='Location where to save the sampled state.')
@click.option('--steps',
    type=click.IntRange(min=1),
    default=2000,
    show_default=True,
    help='Length of the Markov Chain.')
def sample(nn_file, in_file, out_file, steps):
    """
    Runs Monte Carlo on a NQS with given architecture and weights. The result
    is an explicit representation of the NQS, i.e. |ψ〉= ∑ψ(S)|S〉where
    {|S〉} are spin product states. The result is written in the following
    format:

    \b
    <S₁>\t<Re[ψ(S₁)]>\t<Im[ψ(S₁)]>
    <S₂>\t<Re[ψ(S₂)]>\t<Im[ψ(S₂)]>
    ...
    """
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s',
                        level=logging.DEBUG)
    Machine = _make_machine(import_network(nn_file))
    H = heisenberg3x3()
    psi = Machine(H.number_spins)
    psi.load_state_dict(torch.load(in_file))
    magnetisation = 0 if psi.number_spins % 2 == 0 else 1
    thermalisation = int(0.1 * steps)
    monte_carlo_steps = ( thermalisation * psi.number_spins
                        , (thermalisation + steps) * psi.number_spins
                        , psi.number_spins
                        )
    E, wave_function = monte_carlo_loop_for_lanczos(
        psi,
        H,
        random_spin(psi.number_spins, magnetisation),
        monte_carlo_steps
    )
    # For normalisation
    scale = 1.0 / math.sqrt(sum(map(lambda x: abs(x)**2, wave_function.values())))
    out_file.write('# E = {} + {}'.format(E.real, E.imag))
    fmt = '\n{:0' + str(psi.number_spins) + 'b}\t{}\t{}'
    for (spin, coeff) in wave_function.items():
        out_file.write(fmt.format(int(spin), scale * coeff.real, scale * coeff.imag))


@cli.command()
@click.argument('nn-file',
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar='<arch_file>')
@click.option('-i', '--in-file',
    type=click.File(mode='rb'),
    help='File containing the Neural Network weights as a PyTorch `state_dict` '
         'serialised using `torch.save`. It is up to the user to ensure that '
         'the weights are compatible with the architecture read from '
         '<arch_file>.')
@click.option('-o', '--out-file',
    type=click.File(mode='wb'),
    help='Where to save the final state to. It will contain a '
         'PyTorch `state_dict` serialised using `torch.save`. If no file is '
         'specified, the result will be discarded.')
@click.option('--use-sr',
    type=bool,
    default=True,
    show_default=True,
    help='Whether to use Stochastic Reconfiguration for optimisation.')
@click.option('--epochs',
    type=click.IntRange(min=0),
    default=200,
    show_default=True,
    help='Number of learning steps to perform.')
@click.option('--lr',
    type=click.FloatRange(min=1.0E-10),
    default=0.05,
    show_default=True, help='Learning rate.')
@click.option('--steps',
    type=click.IntRange(min=1),
    default=2000,
    show_default=True, help='Length of the Markov Chain.')
def optimise(nn_file, in_file, out_file, use_sr, epochs, lr, steps):
    """
    Variational Monte Carlo optimising E.
    """
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s',
                        level=logging.DEBUG)
    Machine = _make_machine(import_network(nn_file))
    H = heisenberg3x3()
    psi = Machine(H.number_spins)
    if in_file is not None:
        psi.load_state_dict(torch.load(in_file))
    magnetisation = 0 if psi.number_spins % 2 == 0 else 1
    thermalisation = int(0.1 * steps)
    opt = Optimiser(
        psi,
        H,
        magnetisation=magnetisation,
        epochs=epochs,
        monte_carlo_steps=(thermalisation * psi.number_spins,
                           (thermalisation + steps) * psi.number_spins,
                           psi.number_spins),
        learning_rate=lr,
        use_sr=use_sr,
        regulariser=lambda i: 100.0 * 0.9**i + 0.01
    )
    opt()
    if out_file is not None:
        torch.save(psi.state_dict(), out_file)


if __name__ == '__main__':
    cli()
    # cProfile.run('main()')
