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
from itertools import islice
from functools import reduce
import logging
import math
import time
from typing import Dict, List, Tuple, Optional

from numba import jit, jitclass, uint8, int64, float32
import numpy as np
import scipy
from scipy.sparse.linalg import lgmres, LinearOperator
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    The Neural Network used to encode the wave function.

    It is basically a function ℝⁿ -> ℝ² where n is the number of spins.
    """
    def __init__(self, n: int):
        super(Net, self).__init__()
        self._number_spins = n
        self._dense1 = nn.Linear(n, 10)
        self._dense2 = nn.Linear(10, 10)
        # self._dense3 = nn.Linear(10, 10)
        # self._dense4 = nn.Linear(10, 10)
        # self._dense5 = nn.Linear(10, 10)
        self._dense6 = nn.Linear(10, 2, bias=False)
        nn.init.normal_(self._dense1.weight, mean=0, std=5e-1)
        nn.init.normal_(self._dense1.bias, mean=0, std=1e-1)
        nn.init.normal_(self._dense2.weight, std=5e-1)
        nn.init.normal_(self._dense2.bias, std=1e-1)
        # nn.init.normal_(self._dense3.weight, std=1e-1)
        # nn.init.normal_(self._dense3.bias, std=1e-1)
        # nn.init.normal_(self._dense4.weight, std=1e-1)
        # nn.init.normal_(self._dense4.bias, std=1e-1)
        # nn.init.normal_(self._dense5.weight, std=1e-1)
        # nn.init.normal_(self._dense5.bias, std=1e-1)
        # nn.init.normal_(self._dense6.weight, std=1e-1)

    @property
    def number_spins(self) -> int:
        """
        Returns the number of spins the network expects as input.
        """
        return self._number_spins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward propagation.
        """
        x = torch.tanh(self._dense1(x))
        x = torch.tanh(self._dense2(x))
        # x = F.relu(self._dense3(x))
        # x = F.relu(self._dense4(x))
        # x = F.relu(self._dense5(x))
        x = self._dense6(x)
        # x[0].clamp_(-20, 5)
        # logging.info(x)
        return x

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
        b[rest + i] = ((spin[j + 0] == 1.0) << 7) \
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


class Machine(Net):
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


class Heisenberg(object):
    """
    Isotropic Heisenberg Hamiltonian on a lattice.
    """
    def __init__(self, edges: List[Tuple[int, int]]):
        """
        Initialises the Hamiltonian given a list of edges.
        """
        self._graph = edges

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
                if x.real > 10.0:
                    raise ValueError([i, j])
                energy += -1 + 2 * cmath.exp(x)
        return np.complex64(energy)


def monte_carlo_loop(machine, hamiltonian, initial_spin, steps):
    """
    Runs the Monte-Carlo simulation.

    :return: (all gradients, mean gradient, mean local energy, force)
    """
    logging.info('Running Monte-Carlo...')
    start = time.time()
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
    finish = time.time()
    logging.info('Done in {:.2f} seconds!'.format(finish - start))
    logging.info('Acceptance rate: {:.2f}%'.format(chain._accepted / chain._steps * 100))
    return derivatives, mean_O, mean_E, force


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
                 regulariser):
        self._machine = machine
        self._hamiltonian = hamiltonian
        self._magnetisation = magnetisation
        self._epochs = epochs
        self._monte_carlo_steps = monte_carlo_steps
        self._learning_rate = learning_rate
        self._regulariser = regulariser
        self._optimizer = torch.optim.Adam(self._machine.parameters(),
                                           lr=self._learning_rate)
        self._delta = None

    def learning_cycle(self, iteration):
        logging.info('==================== {} ===================='.format(iteration))
        # Random spin with 0 magnetisation
        # TODO(twesterhout): Generalise this to arbitrary magnetisation
        assert self._machine.number_spins % 2 == 0
        spin = random_spin(self._machine.number_spins, self._magnetisation)
        # Monte-Carlo
        (Os, mean_O, E, F) = \
            monte_carlo_loop(self._machine, self._hamiltonian, spin,
                             self._monte_carlo_steps)
        logging.info('E = {}'.format(E))
        # Calculate the "true" gradients
        # We also cache δ to use it as a guess the next time we're computing
        # S⁻¹F.
        self._delta = Covariance(
            Os, mean_O, self._regulariser(iteration)).solve(F, x0=self._delta)
        self._machine.set_gradients(self._delta)
        # Update the variational parameters
        self._optimizer.step()
        self._machine.clear_cache()
        logging.info('∥F∥₂ = {}, ∥δ∥₂ = {}'
                     .format(np.linalg.norm(F), np.linalg.norm(self._delta)))

    def __call__(self):
        for i in range(self._epochs):
            self.learning_cycle(i)
        return self._machine


def main():
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s',
                        level=logging.DEBUG)
    opt = Optimiser(
        Machine(6),
        Heisenberg([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]),
        magnetisation=0,
        epochs=4000,
        monte_carlo_steps=(1000, 13000, 6),
        learning_rate=0.01,
        regulariser=lambda i: 10.0 * 0.9**i + 0.001
    )
    opt()

if __name__ == '__main__':
    main()
    # cProfile.run('main()')
