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
from copy import deepcopy
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
# from numba import jit, jitclass, uint8, int64, uint64, float32
# from numba.types import Bytes
# import numba.extending
import numpy as np
# import scipy
# from scipy.sparse.linalg import lgmres, LinearOperator
import torch
# import torch.nn as nn
# import torch.nn.functional as F

from . import _C_nqs
from ._C_nqs import CompactSpin

################################################################################
# NQS
################################################################################

# def random_spin(n: int, magnetisation: int = None) -> np.ndarray:
#     """
#     :return:
#         a random spin configuration of length ``n``
#         with magnetisation ``magnetisation``.
#     """
#     if n <= 0:
#         raise ValueError("Invalid number of spins: {}".format(n))
#     if magnetisation is not None:
#         if abs(magnetisation) > n:
#             raise ValueError(
#                 "Magnetisation exceeds the number of spins: |{}| > {}".format(
#                     magnetisation, n
#                 )
#             )
#         if (n + magnetisation) % 2 != 0:
#             raise ValueError("Invalid magnetisation: {}".format(magnetisation))
#         number_ups = (n + magnetisation) // 2
#         number_downs = (n - magnetisation) // 2
#         spin = np.empty((n,), dtype=np.float32)
#         spin[:number_ups] = 1.0
#         spin[number_ups:] = -1.0
#         np.random.shuffle(spin)
#         assert int(spin.sum()) == magnetisation
#         return spin
#     else:
#         return np.random.choice([np.float32(-1.0), np.float32(1.0)], size=n)

def random_spin(n: int, magnetisation: int = None) -> np.ndarray:
    return _C_nqs.random_spin(n, magnetisation).numpy()


class Machine(torch.nn.Module):
    """
    Neural Network Quantum State.
    """

    Cell = collections.namedtuple("Cell", ["log_wf", "der_log_wf"])

    def __init__(self, ψ: torch.nn.Module):
        """
        Constructs a new NQS given a PyTorch neural network ψ.
        """
        super().__init__()
        self._ψ = ψ
        # Total number of variational (i.e. trainable parameters)
        self._size = sum(
            map(
                lambda p: reduce(int.__mul__, p.size()),
                filter(lambda p: p.requires_grad, self.parameters()),
            )
        )
        # Hash-table mapping Spin to _Machine.Cell
        self._cache = {}

    @property
    def size(self) -> int:
        """
        :return: number of variational parameters.
        """
        return self._size

    @property
    def number_spins(self):
        """
        :return: number of spins in the system
        """
        return self._ψ.number_spins

    @property
    def ψ(self) -> torch.nn.Module:
        """
        :return: underlying neural network.
        """
        return self._ψ

    @property
    def psi(self) -> torch.nn.Module:
        """
        Same as ``Machine.ψ``.

        :return: underlying neural network
        """
        return self.ψ

    def _log_wf(self, σ: np.ndarray):
        """
        Given a spin configuration ``σ``, calculates ``log(⟨σ|ψ⟩)``
        and returns it wrapped in a ``Cell``.
        """
        (amplitude, phase) = self._ψ.forward(torch.from_numpy(σ))
        return Machine.Cell(log_wf=complex(amplitude, phase), der_log_wf=None)

    def log_wf(self, σ: np.ndarray) -> complex:
        """
        Given a spin configuration ``σ``, returns ``log(⟨σ|ψ⟩)``.

        :param np.ndarray σ:
            Spin configuration. Must be a numpy array of ``float32``.
        """
        compact_spin = CompactSpin(σ)
        cell = self._cache.get(compact_spin)
        if cell is None:
            cell = self._log_wf(σ)
            self._cache[compact_spin] = cell
        return cell.log_wf

    def _copy_grad_to(self, out: np.ndarray):
        """
        Treats gradients of all the parameters of ``ψ`` as a long 1D vector
        and saves them to ``out``.
        """
        i = 0
        for p in map(
            lambda p_: p_.grad.view(-1).numpy(),
            filter(lambda p_: p_.requires_grad, self.parameters()),
        ):
            out[i : i + p.size] = p
            i += p.size

    def _der_log_wf(self, σ: np.ndarray):
        """
        Given a spin configuration ``σ``, calculates ``log(⟨σ|ψ⟩)`` and
        ``∂log(⟨σ|ψ⟩)/∂Wᵢ`` and returns them wrapped in a ``Cell``.
        """
        der_log_wf = np.empty((self.size,), dtype=np.complex64)
        # Forward-propagation to construct the graph
        result = self._ψ.forward(torch.from_numpy(σ))
        (amplitude, phase) = result
        # Computes ∇Re[log(Ψ(x))]
        self._ψ.zero_grad()
        result.backward(torch.tensor([1, 0], dtype=torch.float32), retain_graph=True)
        self._copy_grad_to(der_log_wf.real)
        # Computes ∇Im[log(Ψ(x))]
        self._ψ.zero_grad()
        result.backward(torch.tensor([0, 1], dtype=torch.float32))
        self._copy_grad_to(der_log_wf.imag)

        # Make sure the user doesn't modify our cached value
        der_log_wf.flags["WRITEABLE"] = False
        return Machine.Cell(log_wf=complex(amplitude, phase), der_log_wf=der_log_wf)

    def der_log_wf(self, σ: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes ``∇log(⟨σ|Ψ⟩) = ∂log(⟨σ|ψ⟩)/∂Wᵢ`` and saves it to out.

        .. warning:: Don't modify the returned array!

        :param np.ndarray σ:
            Spin configuration. Must be a numpy array of ``float32``.
        :param Optional[np.ndarray] out:
            Destination array. Must be a numpy array of ``complex64``.
        """
        compact_spin = CompactSpin(σ)
        cell = self._cache.get(compact_spin)
        if cell is None:
            cell = self._der_log_wf(σ)
            self._cache[compact_spin] = cell
        elif cell.der_log_wf is None:
            log_wf = cell.log_wf
            cell = self._der_log_wf(σ)
            self._cache[compact_spin] = cell
            assert np.allclose(log_wf, cell.log_wf)

        if out is None:
            return cell.der_log_wf
        out[:] = cell.der_log_wf
        return out

    def clear_cache(self):
        """
        Clears the internal cache.
        
        .. note::
            This function must be called when the variational
            parameters are updated.
        """
        self._cache = {}

    def set_gradients(self, x: np.ndarray):
        """
        Performs ``∇log(⟨σ|Ψ⟩) <- x``, i.e. sets the gradients of the
        variational parameters to the specified values.

        :param np.ndarray x:
            New value for ``∇log(⟨σ|Ψ⟩)``. Must be a numpy array of
            type ``float32`` of length ``self.size``.
        """
        with torch.no_grad():
            gradients = torch.from_numpy(x)
            i = 0
            for dp in map(
                lambda p_: p_.grad.data.view(-1),
                filter(lambda p_: p_.requires_grad, self.parameters()),
            ):
                (n,) = dp.size()
                dp.copy_(gradients[i : i + n])
                i += n


class ExplicitState(torch.nn.Module):
    """
    Wraps a ``Dict[Spin, complex]`` into a ``torch.nn.Module`` which can
    be used to construct a ``_Machine``.
    """

    def __init__(self, state: Dict[CompactSpin, complex], apply_log=False):
        """
        If ``apply_log=False``, then ``state`` is a dictionary mapping spins ``σ``
        to their corresponding ``log(⟨σ|ψ⟩)``. Otherwise, ``state`` maps ``σ`` to
        ``⟨σ|ψ⟩``.
        """
        super().__init__()
        self._state = state
        self._number_spins = (
            len(next(iter(self._state))) if len(self._state) != 0 else None
        )
        # function used to convert a complex number into a PyTorch tensor
        self._c2t = None
        if apply_log:

            def _f(z):
                z = cmath.log(z)
                return torch.tensor([z.real, z.imag], dtype=torch.float32)

            self._c2t = _f
        else:

            def _f(z):
                return torch.tensor([z.real, z.imag], dtype=torch.float32)

            self._c2t = _f
        # Make sure we don't try to compute log(0)
        self._default = complex(1e-45, 0.0) if apply_log else complex(0.0, 0.0)

    @property
    def number_spins(self) -> int:
        """
        :return:
            number of spins in the system. If ``self.dict`` is empty,
            ``None`` is returned instead.
        """
        return self._number_spins

    @property
    def dict(self) -> Dict[CompactSpin, complex]:
        """
        :return: the underlying dictionary
        """
        return self._state

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.dim() == 1:
            return self._c2t(
                self._state.get(CompactSpin(x.detach().numpy()), self._default)
            )
        else:
            assert x.dim() == 2
            out = torch.empty((x.size(0), 2), dtype=torch.float32)
            for i in range(x.size(0)):
                out[i] = self._c2t(
                    self._state.get(
                        CompactSpin(x[i].detach().numpy()), self._default
                    )
                )
            return out

    def backward(self, _):
        """
        No backward propagation, because there are no parameters to train.
        """
        raise NotImplementedError()

# 
# class CombiningState(torch.nn.Module):
#     """
#     Given two neural networks: one mapping ``σ``s to log amplitudes ``r``s and
#     the other mapping ``σ`` to phases ``φ``s, combines them into a single neural
#     network mapping ``σ``s to ``r + iφ``s.
#     """
# 
#     def __init__(self, log_amplitude: torch.nn.Module, phase: torch.nn.Module):
#         super().__init__()
#         assert log_amplitude.number_spins == phase.number_spins
#         self._log_amplitude = log_amplitude
#         self._phase = phase
# 
#     @property
#     def number_spins(self):
#         """
#         :return: number of spins in the system.
#         """
#         return self._log_amplitude.number_spins
# 
#     def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
#         if x.dim() == 1:
#             return torch.cat(
#                 [
#                     self._log_amplitude(x),
#                     math.pi * torch.max(self._phase(x), dim=0, keepdim=True)[1].float(),
#                 ]
#             )
#         else:
#             assert x.dim() == 2
#             return torch.cat(
#                 [
#                     self._log_amplitude(x),
#                     math.pi * torch.max(self._phase(x), dim=1)[1].view(-1, 1).float(),
#                 ],
#                 dim=1,
#             )
# 
#     def backward(self, x):
#         raise NotImplementedError()
 
# Calculates ``log(1/‖ψ‖₂)``
#
#     (‖ψ‖₂)² = ∑|ψᵢ|² = ∑exp(log(|ψᵢ|²)) = ∑exp(2 * log(|ψᵢ|) - log(ψₘₐₓ) + log(ψₘₐₓ))
#             = exp(log(ψₘₐₓ)) ∙ ∑exp(2 * log(|ψᵢ|) - log(ψₘₐₓ))
#
# where ψₘₐₓ = max({|ψᵢ|}ᵢ). Now
#
#     log(1/‖ψ‖₂) = -0.5 * log((‖ψ‖₂)²) = -0.5 * [log(ψₘₐₓ) + log(∑exp(2 * log(|ψᵢ|) - log(ψₘₐₓ)))]
#
# :param log_ψ: torch.FloatTensor ``[log(|ψᵢ|) for ∀i]``.
# :return:      ``log(1/‖ψ‖₂)``.
@torch.jit.script
def normalisation_constant(log_psi):
    log_psi = log_psi.view([-1])
    max_log_amplitude = torch.max(log_psi)
    scale = -0.5 * (
        max_log_amplitude
        + torch.log(torch.sum(torch.exp(2 * log_psi - max_log_amplitude)))
    )
    return scale


@torch.jit.script
def negative_log_overlap_real(predicted, expected):
    predicted = predicted.view([-1])
    expected = expected.view([-1])
    sqr_l2_expected = torch.dot(expected, expected)
    sqr_l2_predicted = torch.dot(predicted, predicted)
    expected_dot_predicted = torch.dot(expected, predicted)
    return -0.5 * torch.log(
        expected_dot_predicted * expected_dot_predicted
        / (sqr_l2_expected * sqr_l2_predicted)
    )

################################################################################
# Markov chains
################################################################################





# class AllSpins(object):
#     """
#     An iterator over all spin configurations with given magnetisation.
#     """
#
#     def __init__(self, machine, magnetisation: Optional[int]):
#         if machine.number_spins >= 64:
#             raise OverflowError(
#                 "Brute-force iteration is not supported for such long spin chains."
#             )
#         if magnetisation is not None:
#             if abs(magnetisation) > machine.number_spins:
#                 raise ValueError(
#                     "Magnetisation exceeds the number of spins: |{}| > {}".format(
#                         magnetisation, machine.number_spins
#                     )
#                 )
#             if (machine.number_spins + magnetisation) % 2 != 0:
#                 raise ValueError("Invalid magnetisation: {}".format(magnetisation))
#         self._machine = machine
#         self._magnetisation = magnetisation
#         self._steps = 0
#
#     def __len__(self):
#         if self._magnetisation is not None:
#             number_ups = (self._machine.number_spins + self._magnetisation) // 2
#             return int(scipy.special.comb(self._machine.number_spins, number_ups))
#         else:
#             return 2 ** self._machine.number_spins
#
#     def __iter__(self):
#         if self._magnetisation is not None:
#
#             def do_generate():
#                 n = self._machine.number_spins
#                 m = self._magnetisation
#                 number_ups = (n + m) // 2
#                 number_downs = (n - m) // 2
#
#                 for spin in map(
#                     lambda s: np.array(s, dtype=np.float32),
#                     perm_unique([1] * number_ups + [-1] * number_downs),
#                 ):
#                     self._steps += 1
#                     weight = mpmath.exp(2 * self._machine.log_wf(spin).real)
#                     yield MonteCarloState(
#                         weight=weight, spin=spin, machine=self._machine
#                     )
#
#             return do_generate()
#         else:
#
#             def do_generate():
#                 for spin in map(
#                     lambda x: Spin(_CompactSpin(0, 0, uint64(x), n)).numpy(),
#                     range(1, 2 ** self._machine.number_spins),
#                 ):
#                     self._steps += 1
#                     weight = mpmath.exp(2 * self._machine.log_wf(spin).real)
#                     yield MonteCarloState(
#                         weight=weight, spin=spin, machine=self._machine
#                     )
#
#             return do_generate()
#
#     @property
#     def accepted(self):
#         return self._steps
#
#     @property
#     def steps(self):
#         return self._steps


# class MetropolisImportanceMarkovChain(object):
#     def __init__(self, machine, prob_fn, spin: np.ndarray):
#         self._machine = machine
#         self._spin = spin
#         self._flipper = _Flipper(spin)
#         self._prob_fn = prob_fn
#
#         self._prob = self._prob_fn(self._spin)
#         self._weight = self._calculate_weight()
#         self._steps = 0
#         self._accepted = 0
#
#     def _calculate_weight(self):
#         return math.exp(2 * self._machine.log_wf(self._spin).real - self._prob)
#
#     def __iter__(self):
#         def do_generate():
#             while True:
#                 self._steps += 1
#                 yield MonteCarloState(
#                     weight=self._weight, spin=self._spin, machine=self._machine
#                 )
#
#                 flips = self._flipper.read()
#                 self._spin[flips] *= -1
#                 prob_new = self._prob_fn(self._spin)
#
#                 if min(1.0, prob_new / self._prob) > np.random.uniform(0, 1):
#                     self._accepted += 1
#                     self._prob = prob_new
#                     self._weight = self._calculate_weight()
#                     self._flipper.next(True)
#                 else:
#                     # Revert to the previous state
#                     self._spin[flips] *= -1
#                     self._flipper.next(False)
#
#         return do_generate()
#
#     @property
#     def steps(self):
#         return self._steps
#
#     @property
#     def accepted(self):
#         return self._accepted


################################################################################
# Hamiltonians
################################################################################


class WorthlessConfiguration(Exception):
    """
    An exception thrown by ``Heisenberg`` if it encounters a spin configuration
    with a very low weight.
    """

    def __init__(self, flips):
        super().__init__("The current spin configuration has too low a weight.")
        self.suggestion = flips


"""
A Monte Carlo state is an element of the Markov Chain and is a triple
``(wᵢ, σᵢ, ψ)``, where ``wᵢ`` is the *weight*, ``σᵢ`` is the current
spin configuration, and ``ψ`` is the NQS.
"""
MonteCarloState = collections.namedtuple(
    "MonteCarloState", ["weight", "spin", "machine"]
)


################################################################################
# Monte Carlo sampling
################################################################################


class _MonteCarloResult(object):
    def __init__(self, energies, gradients, weights, samples=None):
        self.energies = energies
        self.gradients = gradients
        self.weights = weights
        self.samples = samples


class _MonteCarloStats(object):
    def __init__(self, acceptance, dimension, time=None):
        self.acceptance = acceptance
        self.dimension = dimension
        self.time = time


# def _monte_carlo_kernel(machine, hamiltonian, initial_spin, steps, result):
#     number_steps = len(range(*steps))
#     assert result.energies.size(0) == 2 * number_steps
#     assert result.weights.size(0) == number_steps
#     assert result.gradients.size(0) == number_steps
#     assert result.gradients.size(1) == 2 * machine.size
#     assert result.gradients.stride(1) == 1
#
#     energies = result.energies.numpy().view(dtype=np.complex64)
#     gradients = result.gradients.numpy().view(dtype=np.complex64)
#     weights = []
#     total_weight = mpmath.mpf(0)
#     energies_cache = {}
#     chain = MetropolisMarkovChain(machine, initial_spin)
#     for i, state in enumerate(islice(chain, *steps)):
#         compact_spin = Spin.from_array(state.spin)
#         e_loc = energies_cache.get(compact_spin)
#         if e_loc is None:
#             e_loc = hamiltonian(state)
#             energies_cache[compact_spin] = e_loc
#         energies[i] = e_loc
#         # TODO(twesterhout): This should be done in batches,
#         # but I'm yet to figure out how to accomplish it in PyTorch.
#         machine.der_log_wf(state.spin, out=gradients[i, :])
#         weights.append(state.weight)
#         total_weight += state.weight
#
#     for i in range(len(weights)):
#         result.weights[i] = float(weights[i] / total_weight)
#
#     return _MonteCarloStats(
#         acceptance=chain.accepted / chain.steps, dimension=len(energies_cache)
#     )


# def monte_carlo_loop(machine, hamiltonian, initial_spin, steps):
#     number_steps = len(range(*steps))
#     result = _MonteCarloResult(
#         energies=torch.empty(
#             (2 * number_steps,), dtype=torch.float32, requires_grad=False
#         ),
#         gradients=torch.empty(
#             (number_steps, 2 * machine.size), dtype=torch.float32, requires_grad=False
#         ),
#         weights=torch.empty((number_steps,), dtype=torch.float32, requires_grad=False),
#     )
#
#     start = time.time()
#     spin = np.copy(initial_spin)
#     restarts = 5
#     stats = None
#     while stats is None:
#         try:
#             stats = _monte_carlo_kernel(machine, hamiltonian, spin, steps, result)
#         except WorthlessConfiguration as err:
#             if restarts > 0:
#                 logging.warning("Restarting the Monte Carlo loop...")
#                 restarts -= 1
#                 spin[err.suggestion] *= -1
#             else:
#                 raise
#     finish = time.time()
#     stats.time = finish - start
#     return result, stats




# def explicit_sum_loop(machine, hamiltonian, magnetisation=None):
#     chain = AllSpins(machine, magnetisation)
#     number_steps = len(chain)
#     result = _MonteCarloResult(
#         energies=torch.empty(
#             (2 * number_steps,), dtype=torch.float32, requires_grad=False
#         ),
#         gradients=torch.empty(
#             (number_steps, 2 * machine.size), dtype=torch.float32, requires_grad=False
#         ),
#         weights=torch.empty((number_steps,), dtype=torch.float32, requires_grad=False),
#     )
#
#     start = time.time()
#     energies = result.energies.numpy().view(dtype=np.complex64)
#     gradients = result.gradients.numpy().view(dtype=np.complex64)
#     weights = []
#     total_weight = mpmath.mpf(0)
#     for i, state in enumerate(chain):
#         energies[i] = hamiltonian(state, cutoff=None)
#         # TODO(twesterhout): This should be done in batches,
#         # but I'm yet to figure out how to accomplish it in PyTorch.
#         machine.der_log_wf(state.spin, out=gradients[i, :])
#         weights.append(state.weight)
#         total_weight += state.weight
#
#     for i in range(len(weights)):
#         result.weights[i] = float(weights[i] / total_weight)
#     finish = time.time()
#
#     stats = _MonteCarloStats(
#         acceptance=chain.accepted / chain.steps,
#         dimension=number_steps,
#         time=finish - start,
#     )
#     return result, stats


# def monte_carlo(machine, hamiltonian, steps, magnetisation=None, explicit=False):
#     # TODO(twesterhout): Later on, this function should do the parallelisation.
#     logging.info("Running Monte Carlo...")
#     if explicit:
#         result, stats = explicit_sum_loop(machine, hamiltonian, magnetisation)
#     else:
#         result, stats = monte_carlo_loop(
#             machine,
#             hamiltonian,
#             random_spin(machine.number_spins, magnetisation),
#             steps,
#         )
#     logging.info(
#         (
#             "Done in {:.2f} seconds, accepted {:.2f}% flips, "
#             "and sampled {} basis vectors"
#         ).format(stats.time, 100 * stats.acceptance, stats.dimension)
#     )
#     energies = result.energies.numpy().view(dtype=np.complex64)
#     gradients = result.gradients.numpy().view(dtype=np.complex64)
#     weights = result.weights.numpy()
#     return energies, gradients, weights


# def run_step(psi, samples, batch_size, hamitonian):
#     y = torch.empty((samples.size(0), 2), dtype=torch.float32, requires_grad=False)
#     with torch.no_grad():
#         for i, x_batch in enumerate(torch.split(samples, batch_size)):
#             y[i * batch_size : (i + 1) * batch_size] = psi(x_batch)
#         max_amplitude = torch.max(y[:, 0]).item()
#
#     local_energies = np.empty((samples.size(0),), dtype=np.complex64)
#     with torch.no_grad():
#         for i, x in enumerate(samples):
#             local_energies[i] = hamiltonian(psi, x)

# 
# def l2_error(φ: torch.Tensor, ψ: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
#     """
#     Computes ``1 - |⟨φ|ψ⟩| / (‖φ‖₂∙‖ψ‖₂)``.
# 
#       * ‖φ‖₂ = sqrt( ∑(Reφᵢ)² + (Imφᵢ)²  )
#       * ‖ψ‖₂ = sqrt( ∑(Reψᵢ)² + (Imψᵢ)²  )
#       * ⟨φ|ψ⟩ = Re⟨φ|ψ⟩ + i∙Im⟨φ|ψ⟩
#               = ∑Re[φᵢ*∙ψᵢ] + i∙∑Im[φᵢ*∙ψᵢ]
#               = ∑(Reφᵢ∙Reψᵢ + Imφᵢ∙Imψᵢ) + i∑(Reφᵢ∙Imψᵢ - Imφᵢ∙Reψᵢ)
# 
#     :param φ:
#         ``(n, 2)`` tensor of floats, where each row represents
#         a single complex number.
#     :param ψ:
#         ``(n, 2)`` tensor of floats, where each row represents
#         a single complex number.
#     :return:
#         A one-element tensor ``1 - |⟨φ|ψ⟩| / (‖φ‖₂∙‖ψ‖₂)``.
#     """
#     Re_φ, Im_φ = φ
#     Re_ψ, Im_ψ = ψ
#     sqr_l2_φ = torch.dot(w, Re_φ * Re_φ) + torch.dot(w, Im_φ * Im_φ)
#     sqr_l2_ψ = torch.dot(w, Re_ψ * Re_ψ) + torch.dot(w, Im_ψ * Im_ψ)
#     Re_φ_dot_ψ = torch.dot(w, Re_φ * Re_ψ) + torch.dot(w, Im_φ * Im_ψ)
#     Im_φ_dot_ψ = torch.dot(w, Re_φ * Im_ψ) - torch.dot(w, Im_φ * Re_ψ)
#     return 1 - torch.sqrt((Re_φ_dot_ψ ** 2 + Im_φ_dot_ψ ** 2) / (sqr_l2_φ * sqr_l2_ψ))


def negative_log_overlap(φ: torch.Tensor, ψ: torch.Tensor) -> torch.Tensor:
    """
    Computes ``-log(|⟨φ|ψ⟩| / (‖φ‖₂∙‖ψ‖₂))``.

      * ‖φ‖₂ = sqrt( ∑(Reφᵢ)² + (Imφᵢ)²  )
      * ‖ψ‖₂ = sqrt( ∑(Reψᵢ)² + (Imψᵢ)²  )
      * ⟨φ|ψ⟩ = Re⟨φ|ψ⟩ + i∙Im⟨φ|ψ⟩
              = ∑Re[φᵢ*∙ψᵢ] + i∙∑Im[φᵢ*∙ψᵢ]
              = ∑(Reφᵢ∙Reψᵢ + Imφᵢ∙Imψᵢ) + i∑(Reφᵢ∙Imψᵢ - Imφᵢ∙Reψᵢ)

    :param φ:
        ``(n, 2)`` tensor of floats, where each row represents
        a single complex number.
    :param ψ:
        ``(n, 2)`` tensor of floats, where each row represents
        a single complex number.
    :return:
        A one-element tensor ``-log(|⟨φ|ψ⟩| / (‖φ‖₂∙‖ψ‖₂))``.
    """
    Re_φ, Im_φ = φ
    Re_ψ, Im_ψ = ψ
    sqr_l2_φ = torch.dot(Re_φ, Re_φ) + torch.dot(Im_φ, Im_φ)
    sqr_l2_ψ = torch.dot(Re_ψ, Re_ψ) + torch.dot(Im_ψ, Im_ψ)
    Re_φ_dot_ψ = torch.dot(Re_φ, Re_ψ) + torch.dot(Im_φ, Im_ψ)
    Im_φ_dot_ψ = torch.dot(Re_φ, Im_ψ) - torch.dot(Im_φ, Re_ψ)
    # return -0.5 * torch.log((Re_φ_dot_ψ ** 2 + Im_φ_dot_ψ ** 2) / (sqr_l2_φ * sqr_l2_ψ))
    return 1 - torch.sqrt((Re_φ_dot_ψ ** 2 + Im_φ_dot_ψ ** 2) / (sqr_l2_φ * sqr_l2_ψ))


#
# def to_explicit_dict(
#     ψ, steps=None, magnetisation=None, explicit=False
# ) -> Dict[Spin, complex]:
#     ψ = ψ if isinstance(ψ, Machine) else Machine(ψ)
#     old_cache = deepcopy(ψ._cache)
#     ψ.clear_cache()
#     result, stats = sample_state(
#         ψ, steps=steps, magnetisation=magnetisation, explicit=explicit
#     )
# 
#     ψ_s = torch.empty((len(ψ._cache),), dtype=torch.float32)
#     for i, value in enumerate(map(lambda x: x.log_wf.real, ψ._cache.values())):
#         ψ_s[i] = value
#     scale = normalisation_constant(ψ_s).item()
#     explicit = {}
#     for spin, cell in ψ._cache.items():
#         explicit[spin] = cmath.exp(cell.log_wf + complex(scale, 0.0))
# 
#     ψ._cache.update(old_cache)
#     return explicit


################################################################################
# SR
################################################################################

# 
# def make_force(energies, gradients, weights) -> np.ndarray:
#     mean_energy = np.dot(weights, energies)
#     mean_gradient = np.dot(weights, gradients)
#     gradients = gradients.conj().transpose()
#     force = np.dot(weights, (energies * gradients).transpose())
#     force -= mean_gradient.conj() * mean_energy
#     return force, mean_energy, mean_gradient
# 
# 
# class DenseCovariance(LinearOperator):
#     """
#     Dense representation of the covariance matrix matrix S in Stochastic
#     Reconfiguration method [1].
#     """
# 
#     def __init__(self, gradients: np.ndarray, weights: np.ndarray, regulariser: float):
#         (steps, n) = gradients.shape
#         super().__init__(np.float32, (2 * n, n))
# 
#         mean_gradient = np.dot(weights, gradients)
#         gradients -= mean_gradient
# 
#         conj_gradients = gradients.transpose().conj()
# 
#         # Viewing weights as a column vector for proper broadcasting
#         weights_view = weights.view()
#         weights_view.shape = (-1, 1)
#         gradients = np.multiply(weights_view, gradients, out=gradients)
# 
#         S = np.dot(conj_gradients, gradients)
#         S += regulariser * np.eye(n, dtype=np.float32)
# 
#         self._matrix = np.concatenate([S.real, S.imag], axis=0)
#         assert not np.any(np.isnan(self._matrix))
#         assert not np.any(np.isinf(self._matrix))
# 
#     def _matvec(self, x):
#         assert x.dtype == self.dtype
#         return np.matmul(self._matrix, x)
# 
#     def _rmatvec(self, y):
#         assert y.dtype == self.dtype
#         return np.matmul(self._matrix.transpose(), y)
# 
#     def solve(self, b, x0=None):
#         start = time.time()
#         assert b.dtype == np.complex64
#         assert not np.any(np.isnan(b))
#         assert not np.any(np.isinf(b))
#         logging.info("Calculating S⁻¹F...")
#         b = np.concatenate([b.real, b.imag])
#         assert b.dtype == np.float32
#         x = scipy.linalg.lstsq(self._matrix, b)[0]
#         assert x.dtype == self.dtype
#         finish = time.time()
#         logging.info("Done in {:.2f} seconds!".format(finish - start))
#         return x
# 
# 
# class DeltaMachine(torch.nn.Module):
#     class Cell(object):
#         def __init__(
#             self, wave_function: complex, gradient: Optional[np.ndarray] = None
#         ):
#             self.log_wf = wave_function
#             self.der_log_wf = gradient
# 
#     def __init__(self, ψ: torch.nn.Module, δψ: torch.nn.Module):
#         """
#         Given neural networks (i.e. instances of ``torch.nn.Module``) representing
#         ψ and δψ, constructs an NQS given by ψ + δψ with ∂ψ/∂Wᵢ assumed to be 0.
#         """
#         super().__init__()
#         self._ψ = ψ
#         self._δψ = δψ
#         if self._ψ.number_spins != self._δψ.number_spins:
#             raise ValueError(
#                 "ψ and δψ represent systems of different number of particles"
#             )
#         self._size = sum(
#             map(
#                 lambda p: reduce(int.__mul__, p.size()),
#                 filter(lambda p: p.requires_grad, self._δψ.parameters()),
#             )
#         )
#         self._cache = {}
# 
#     @property
#     def number_spins(self) -> int:
#         return self._ψ.number_spins
# 
#     @property
#     def size(self) -> int:
#         return self._size
# 
#     @property
#     def ψ(self):
#         return self._ψ
# 
#     @property
#     def δψ(self):
#         return self._δψ
# 
#     def _log_wf(self, spin: np.ndarray, compact_spin: Spin) -> complex:
#         cell = self._cache.get(compact_spin)
#         if cell is not None:
#             return cell.log_wf
#         with torch.no_grad():
#             log_ψ = self._ψ.log_wf(spin)
#             log_δψ = _tensor2complex(self._δψ.forward(torch.from_numpy(spin)))
#         log_wf = cmath.log(cmath.exp(log_ψ) + cmath.exp(log_δψ))
#         self._cache[compact_spin] = DeltaMachine.Cell(log_wf)
#         return log_wf
# 
#     def log_wf(self, spin: np.ndarray) -> complex:
#         return self._log_wf(spin, Spin.from_array(spin))
# 
#     def _copy_grad_to(self, out: np.ndarray):
#         i = 0
#         for p in map(
#             lambda p_: p_.grad.view(-1).numpy(),
#             filter(lambda p_: p_.requires_grad, self._δψ.parameters()),
#         ):
#             out[i : i + p.size] = p
#             i += p.size
# 
#     def _der_log_wf(self, spin: np.ndarray, compact_spin: Spin, out=None):
#         if out is None:
#             out = np.empty((self.size,), dtype=np.complex64)
#         cell = self._cache.get(compact_spin)
#         if cell is None:
#             _ = self._log_wf(spin, compact_spin)
#             cell = self._cache.get(compact_spin)
#         elif cell.der_log_wf is not None:
#             out[:] = cell.der_log_wf
#             return out
# 
#         # Forward-propagation to construct the graph
#         grad_δψ = np.empty((self.size,), dtype=np.complex64)
#         result = self._δψ.forward(torch.from_numpy(spin))
#         δψ = cmath.exp(_tensor2complex(result))
#         # Computes ∇Re[log(δΨ(σ))]
#         self._δψ.zero_grad()
#         result.backward(torch.tensor([1, 0], dtype=torch.float32), retain_graph=True)
#         self._copy_grad_to(grad_δψ.real)
#         # Computes ∇Im[log(δΨ(σ))]
#         self._δψ.zero_grad()
#         result.backward(torch.tensor([0, 1], dtype=torch.float32))
#         self._copy_grad_to(grad_δψ.imag)
# 
#         wf = cmath.exp(cell.log_wf)
#         norm = abs(wf) ** 2
# 
#         # Magic :)
#         out.real[:] = (wf.real * δψ.real + wf.imag * δψ.imag) / norm * grad_δψ.real
#         out.real += (wf.imag * δψ.real - wf.real * δψ.imag) / norm * grad_δψ.imag
# 
#         out.imag[:] = (wf.real * δψ.imag - wf.imag * δψ.real) / norm * grad_δψ.real
#         out.imag += (wf.imag * δψ.real + wf.imag * δψ.imag) / norm * grad_δψ.imag
# 
#         # Save the results
#         # TODO(twesterhout): Remove the copy when it's safe to do so.
#         cell.der_log_wf = np.copy(out)
#         return out
# 
#     def der_log_wf(self, spin: np.ndarray, out: np.ndarray = None) -> np.ndarray:
#         return self._der_log_wf(spin, Spin.from_array(spin), out)
# 
#     def set_gradients(self, x: np.ndarray):
#         """
#         Performs ∇W = x, i.e. sets the gradients of the variational parameters.
# 
#         :param np.ndarray x: New value for ∇W. Must be a numpy array of
#         ``float32`` of length ``self.size``.
#         """
#         with torch.no_grad():
#             gradients = torch.from_numpy(x)
#             i = 0
#             for dp in map(
#                 lambda p_: p_.grad.data.view(-1),
#                 filter(lambda p_: p_.requires_grad, self._δψ.parameters()),
#             ):
#                 (n,) = dp.size()
#                 dp.copy_(gradients[i : i + n])
#                 i += n
# 
#     def clear_cache(self):
#         self._cache = {}
# 
#     def state_dict(self, *args, **kwargs):
#         return self._δψ.state_dict(*args, **kwargs)
# 
# 
# class Optimiser(object):
#     def __init__(
#         self,
#         machine,
#         hamiltonian,
#         magnetisation,
#         epochs,
#         monte_carlo_steps,
#         learning_rate,
#         use_sr,
#         regulariser,
#         model_file,
#         time_limit,
#     ):
#         self._machine = machine
#         self._hamiltonian = hamiltonian
#         self._magnetisation = magnetisation
#         self._epochs = epochs
#         self._monte_carlo_steps = monte_carlo_steps
#         self._learning_rate = learning_rate
#         self._use_sr = use_sr
#         self._model_file = model_file
#         self._time_limit = time_limit
#         if use_sr:
#             self._regulariser = regulariser
#             self._delta = None
#             self._optimizer = torch.optim.SGD(
#                 filter(lambda p: p.requires_grad, self._machine.parameters()),
#                 lr=self._learning_rate,
#             )
#         else:
#             self._optimizer = torch.optim.Adam(
#                 filter(lambda p: p.requires_grad, self._machine.parameters()),
#                 lr=self._learning_rate,
#             )
# 
#     def _monte_carlo(self):
#         return monte_carlo(
#             self._machine,
#             self._hamiltonian,
#             self._monte_carlo_steps,
#             self._magnetisation,
#             explicit=True,
#         )
# 
#     def learning_cycle(self, iteration):
#         logging.info("==================== {} ====================".format(iteration))
#         # Monte Carlo
#         energies, gradients, weights = self._monte_carlo()
#         force, mean_energy, mean_gradient = make_force(energies, gradients, weights)
#         logging.info("E = {}, ∥F∥₂ = {}".format(mean_energy, np.linalg.norm(force)))
# 
#         if self._use_sr:
#             covariance = DenseCovariance(
#                 gradients, weights, self._regulariser(iteration)
#             )
#             self._delta = covariance.solve(force)
#         else:
#             self._delta = force.real
#         logging.info("∥δW∥₂ = {}".format(np.linalg.norm(self._delta)))
#         self._machine.set_gradients(self._delta)
#         self._optimizer.step()
#         self._machine.clear_cache()
# 
#     def __call__(self):
#         if self._model_file is not None:
# 
#             def save():
#                 # NOTE: This is important, because we want to overwrite the
#                 # previous weights
#                 self._model_file.seek(0)
#                 self._model_file.truncate()
#                 torch.save(self._machine.ψ.state_dict(), self._model_file)
# 
#         else:
#             save = lambda: None
# 
#         save()
#         if self._time_limit is not None:
#             start = time.time()
#             for i in range(self._epochs):
#                 if time.time() - start > self._time_limit:
#                     save()
#                     start = time.time()
#                 self.learning_cycle(i)
#         else:
#             for i in range(self._epochs):
#                 self.learning_cycle(i)
#         save()
#         return self._machine
# 

################################################################################
# SWO
################################################################################

# "Borrowed" from pytorch/torch/serialization.py.
# All credit goes to PyTorch developers.
def _with_file_like(f, mode, body):
    """
    Executes a body function with a file object for f, opening
    it in 'mode' if it is a string filename.
    """
    new_fd = False
    if isinstance(f, str) or \
            (sys.version_info[0] == 2 and isinstance(f, unicode)) or \
            (sys.version_info[0] == 3 and isinstance(f, pathlib.Path)):
        new_fd = True
        f = open(f, mode)
    try:
        return body(f)
    finally:
        if new_fd:
            f.close()

def _load_explicit(stream):
    psi = {}
    number_spins = None
    for line in stream:
        if line.startswith(b"#"):
            continue
        (spin, real, imag) = line.split()
        number_spins = len(spin)
        psi[CompactSpin(spin)] = complex(float(real), float(imag))
        break
    for line in stream:
        if line.startswith(b"#"):
            continue
        (spin, real, imag) = line.split()
        if len(spin) != number_spins:
            raise ValueError("A single state cannot contain spin "
                             "configurations of different sizes")
        psi[CompactSpin(spin)] = complex(float(real), float(imag))
    return psi, number_spins

def load_explicit(stream):
    return _with_file_like(stream, "rb", _load_explicit)
    # psi = {}
    # number_spins = None
    # for line in stream:
    #     if line.startswith(b"#"):
    #         continue
    #     (spin, real, imag) = line.split()
    #     if number_spins is None:
    #         number_spins = len(spin)
    #     else:
    #         assert number_spins == len(spin)
    #     spin = CompactSpin(spin)  # int(spin, base=2)
    #     coeff = complex(float(real), float(imag))
    #     psi[spin] = coeff
    # return psi, number_spins


# class ChebyshevTargetState(torch.nn.Module):
#     jk

# 
# def _swo_to_explicit(f, xs):
#     f_xs = f(xs)
#     scale = normalisation_constant(f_xs[:, 0])
#     amplitude = torch.exp(f_xs[:, 0] + scale)
#     real = amplitude * torch.cos(f_xs[:, 1])
#     imag = amplitude * torch.sin(f_xs[:, 1])
#     return real, imag, scale
# 
# 
# def _apply_normalised(f, xs):
#     f_xs = f(xs)
#     max_log_amplitude = torch.max(f_xs)
#     scale = -0.5 * (
#         max_log_amplitude
#         + torch.log(torch.sum(torch.exp(2 * f_xs - max_log_amplitude)))
#     )
#     return torch.exp(f_xs + scale)
# 
# 
# def _swo_compute_weights(f, g):
#     f_real, f_imag = f
#     g_real, g_imag = g
#     δ_real = f_real - g_real
#     δ_imag = f_imag - g_imag
#     weights = δ_real * δ_real + δ_imag * δ_imag
#     weights *= 1.0 / weights.sum()
#     return weights
# 
# 
# def optimise_inner(
#     optimiser, ψ: torch.nn.Module, H, samples: torch.Tensor, epochs: int, τ: float
# ):
#     φ = TargetState(deepcopy(ψ), H, τ)
#     φ_real, φ_imag, _ = _swo_to_explicit(φ, samples)
# 
#     important_iterations = list(range(0, epochs, epochs // 10))
#     if epochs - 1 not in important_iterations:
#         important_iterations.append(epochs - 1)
# 
#     for i in range(epochs):
#         optimiser.zero_grad()
#         ψ_real, ψ_imag, ψ_scale = _swo_to_explicit(ψ, samples)
#         # with torch.no_grad():
#         #     weights = _swo_compute_weights((φ_real, φ_imag), (ψ_real, ψ_imag))
#         # loss = l2_error((φ_real, φ_imag), (ψ_real, ψ_imag), weights)
#         loss = negative_log_overlap((φ_real, φ_imag), (ψ_real, ψ_imag))
#         if i in important_iterations:
#             logging.info("{}%: loss = {}".format(100 * (i + 1) // epochs, loss))
#         loss.backward()
#         optimiser.step()
# 
#     return ψ
# 

# 
# 
# 
# 
# 
# @click.group()
# def cli():
#     pass
# 
# 
# def import_network(nn_file):
#     module_name, extension = os.path.splitext(os.path.basename(nn_file))
#     module_dir = os.path.dirname(nn_file)
#     if extension != ".py":
#         raise ValueError(
#             "Could not import the network from {}: not a python source file."
#         )
#     # Insert in the beginning
#     sys.path.insert(0, module_dir)
#     module = importlib.import_module(module_name)
#     sys.path.pop(0)
#     return module.Net
# 
# 
# 
# 
# @cli.command()
# @click.argument(
#     "nn-file",
#     type=click.Path(exists=True, resolve_path=True, path_type=str),
#     metavar="<torch.nn.Module file>",
# )
# @click.option(
#     "-i",
#     "--in-file",
#     type=click.File(mode="rb"),
#     help="File containing the Neural Network weights as a PyTorch `state_dict` "
#     "serialised using `torch.save`. It is up to the user to ensure that "
#     "the weights are compatible with the architecture read from "
#     "<arch_file>.",
# )
# @click.option(
#     "-o",
#     "--out-file",
#     type=click.File(mode="wb"),
#     help="Where to save the final state to. It will contain a "
#     "PyTorch `state_dict` serialised using `torch.save`. If no file is "
#     "specified, the result will be discarded.",
# )
# @click.option(
#     "-H",
#     "--hamiltonian",
#     "hamiltonian_file",
#     type=click.File(mode="r"),
#     required=True,
#     help="File containing the Heisenberg Hamiltonian specifications.",
# )
# @click.option(
#     "--use-sr",
#     type=bool,
#     default=True,
#     show_default=True,
#     help="Whether to use Stochastic Reconfiguration for optimisation.",
# )
# @click.option(
#     "-n",
#     "--epochs",
#     type=click.IntRange(min=0),
#     default=200,
#     show_default=True,
#     help="Number of learning steps to perform.",
# )
# @click.option(
#     "--lr",
#     type=click.FloatRange(min=1.0e-10),
#     default=0.05,
#     show_default=True,
#     help="Learning rate.",
# )
# @click.option(
#     "--time",
#     "time_limit",
#     type=click.FloatRange(min=1.0e-10),
#     show_default=True,
#     help="Time interval that specifies how often the model is written to the "
#     "output file. If not specified, the weights are saved only once -- "
#     "after all the iterations.",
# )
# @click.option(
#     "--steps",
#     type=click.IntRange(min=1),
#     default=2000,
#     show_default=True,
#     help="Length of the Markov Chain.",
# )
# def optimise(
#     nn_file, in_file, out_file, hamiltonian_file, use_sr, epochs, lr, steps, time_limit
# ):
#     """
#     Variational Monte Carlo optimising E.
#     """
#     logging.basicConfig(
#         format="[%(asctime)s] [%(levelname)s] %(message)s",
#         datefmt="%H:%M:%S",
#         level=logging.DEBUG,
#     )
#     BaseNet = import_network(nn_file)
#     H = read_hamiltonian(hamiltonian_file)
#     machine = _Machine(BaseNet(H.number_spins))
#     if in_file is not None:
#         logging.info("Reading the weights...")
#         machine.psi.load_state_dict(torch.load(in_file))
# 
#     magnetisation = 0 if machine.number_spins % 2 == 0 else 1
#     thermalisation = int(0.1 * steps)
#     opt = Optimiser(
#         machine,
#         H,
#         magnetisation=magnetisation,
#         epochs=epochs,
#         monte_carlo_steps=(
#             thermalisation * machine.number_spins,
#             (thermalisation + steps) * machine.number_spins,
#             machine.number_spins,
#         ),
#         learning_rate=lr,
#         use_sr=use_sr,
#         regulariser=lambda i: 100.0 * 0.9 ** i + 0.001,
#         model_file=out_file,
#         time_limit=time_limit,
#     )
#     opt()
# 
# 
# @cli.command()
# @click.argument(
#     "amplitude-nn-file",
#     type=click.Path(exists=True, resolve_path=True, path_type=str),
#     metavar="<amplitude torch.nn.Module>",
# )
# @click.argument(
#     "phase-nn-file",
#     type=click.Path(exists=True, resolve_path=True, path_type=str),
#     metavar="<phase torch.nn.Module>",
# )
# @click.option(
#     "--in-amplitude",
#     type=click.File(mode="rb"),
#     help="Pickled initial `state_dict` of the neural net predicting log amplitudes.",
# )
# @click.option(
#     "--in-phase",
#     type=click.File(mode="rb"),
#     help="Pickled initial `state_dict` of the neural net predicting phases.",
# )
# @click.option("-o", "--out-file", type=str, help="Basename for output files.")
# @click.option(
#     "-H",
#     "--hamiltonian",
#     "hamiltonian_file",
#     type=click.File(mode="r"),
#     required=True,
#     help="File containing the Heisenberg Hamiltonian specifications.",
# )
# @click.option(
#     "--epochs-outer",
#     type=click.IntRange(min=0),
#     default=200,
#     show_default=True,
#     help="Number of outer learning steps (i.e. Lanczos steps) to perform.",
# )
# @click.option(
#     "--epochs-amplitude",
#     type=click.IntRange(min=0),
#     default=2000,
#     show_default=True,
#     help="Number of epochs when learning the amplitude",
# )
# @click.option(
#     "--epochs-phase",
#     type=click.IntRange(min=0),
#     default=2000,
#     show_default=True,
#     help="Number of epochs when learning the phase",
# )
# @click.option(
#     "--lr-amplitude",
#     type=click.FloatRange(min=1.0e-10),
#     default=0.05,
#     show_default=True,
#     help="Learning rate for training the amplitude net.",
# )
# @click.option(
#     "--lr-phase",
#     type=click.FloatRange(min=1.0e-10),
#     default=0.05,
#     show_default=True,
#     help="Learning rate for training the phase net.",
# )
# @click.option(
#     "--tau",
#     type=click.FloatRange(min=1.0e-10),
#     default=0.2,
#     show_default=True,
#     help="τ",
# )
# @click.option(
#     "--steps",
#     type=click.IntRange(min=1),
#     default=2000,
#     show_default=True,
#     help="Length of the Markov Chain.",
# )
# def swo(
#     amplitude_nn_file,
#     phase_nn_file,
#     in_amplitude,
#     in_phase,
#     out_file,
#     hamiltonian_file,
#     epochs_outer,
#     epochs_amplitude,
#     epochs_phase,
#     lr_amplitude,
#     lr_phase,
#     tau,
#     steps,
# ):
#     logging.basicConfig(
#         format="[%(asctime)s] [%(levelname)s] %(message)s",
#         datefmt="%H:%M:%S",
#         level=logging.DEBUG,
#     )
#     H = read_hamiltonian(hamiltonian_file)
#     number_spins = H.number_spins
#     magnetisation = 0 if number_spins % 2 == 0 else 1
# 
#     thermalisation = steps // 10
#     m_c_steps = (
#         thermalisation * number_spins,
#         (thermalisation + steps) * number_spins,
#         1,
#     )
# 
#     ψ_amplitude = import_network(amplitude_nn_file)(number_spins)
#     if in_amplitude is not None:
#         logging.info("Reading initial state...")
#         ψ_amplitude.load_state_dict(torch.load(in_amplitude))
# 
#     ψ_phase = import_network(phase_nn_file)(number_spins)
#     if in_phase is not None:
#         logging.info("Reading initial state...")
#         ψ_phase.load_state_dict(torch.load(in_phase))
# 
#     # def save_explicit(iteration):
#     #     explicit_dict = to_explicit_dict(
#     #         CombiningState(ψ_amplitude, ψ_phase),
#     #         explicit=True,
#     #         magnetisation=magnetisation,
#     #     )
#     #     with open("{}.{}.explicit".format(out_file, iteration), "wb") as f:
#     #         for spin, value in explicit_dict.items():
#     #             f.write(
#     #                 "{}\t{:.10e}\t{:.10e}\n".format(
#     #                     spin, value.real, value.imag
#     #                 ).encode("utf-8")
#     #             )
# 
#     for i in range(epochs_outer):
#         logging.info("\t#{}".format(i + 1))
#         _swo_step(
#             (ψ_amplitude, ψ_phase),
#             SWOConfig(
#                 H=H,
#                 τ=tau,
#                 steps=(5,) + m_c_steps,
#                 magnetisation=magnetisation,
#                 lr_amplitude=lr_amplitude,
#                 lr_phase=lr_phase,
#                 epochs_amplitude=epochs_amplitude,
#                 epochs_phase=epochs_phase,
#             ),
#         )
#         torch.save(
#             ψ_amplitude.state_dict(), "{}.{}.amplitude.weights".format(out_file, i)
#         )
#         torch.save(ψ_phase.state_dict(), "{}.{}.phase.weights".format(out_file, i))
#         # save_explicit(i)
#     # save_explicit(i)
# 
# 
# @cli.command()
# @click.argument(
#     "nn-file",
#     type=click.Path(exists=True, resolve_path=True, path_type=str),
#     metavar="<arch_file>",
# )
# @click.option(
#     "-t",
#     "--train-file",
#     type=click.File(mode="rb"),
#     required=True,
#     help="File containing the explicit wave function representation as "
#     "generated by `sample`. It will be used as the training data set.",
# )
# @click.option(
#     "-i",
#     "--in-file",
#     type=click.File(mode="rb"),
#     help="File containing the initial Neural Network weights as a PyTorch "
#     "`state_dict` serialised using `torch.save`. It is up to the user to ensure "
#     "that the weights are compatible with the architecture read from <arch_file>.",
# )
# @click.option(
#     "-o",
#     "--out-file",
#     type=str,
#     required=True,
#     help="File where the final weights will be stored. It will contain a "
#     "PyTorch `state_dict` serialised using `torch.save`.",
# )
# @click.option(
#     "--lr",
#     type=click.FloatRange(min=1.0e-10),
#     default=0.05,
#     show_default=True,
#     help="Learning rate.",
# )
# @click.option(
#     "--batch-size", type=int, default=None, show_default=True, help="Learning rate."
# )
# @click.option(
#     "--epochs",
#     type=click.IntRange(min=0),
#     default=200,
#     show_default=True,
#     help="Number of learning steps to perform. Here step is defined as a single "
#     "update of the parameters.",
# )
# @click.option(
#     "--optimiser",
#     type=str,
#     default="SGD",
#     help="Optimizer to use. Valid values are names of classes in torch.optim "
#     "(i.e. 'SGD', 'Adam', 'Adamax', etc.)",
# )
# def train(nn_file, train_file, out_file, in_file, lr, optimiser, epochs, batch_size):
#     """
#     Supervised learning.
#     """
#     logging.basicConfig(
#         format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.DEBUG
#     )
#     φ_dict, number_spins = load_explicit(train_file)
#     logging.info("Generating training data set...")
#     samples = torch.empty((len(φ_dict), number_spins), dtype=torch.float32)
#     φ_real = torch.empty((len(φ_dict),), dtype=torch.float32)
#     φ_imag = torch.empty((len(φ_dict),), dtype=torch.float32)
#     for (i, (key, value)) in enumerate(φ_dict.items()):
#         samples[i, :] = torch.from_numpy(key.numpy())
#         φ_real[i] = value.real
#         φ_imag[i] = value.imag
# 
#     ψ = import_network(nn_file)(number_spins)
#     if in_file is not None:
#         logging.info("Reading initial state...")
#         ψ.load_state_dict(torch.load(in_file))
# 
#     # NOTE(twesterhout): This is a hack :)
#     optimiser = getattr(torch.optim, optimiser)(
#         filter(lambda p: p.requires_grad_, ψ.parameters()), lr=lr
#     )
# 
#     def save(i):
#         out_file_name = "{}.{}.weights".format(out_file, i)
#         torch.save(ψ.state_dict(), out_file_name)
# 
#     important_iterations = list(range(0, epochs, epochs // 10))
#     if epochs - 1 not in important_iterations:
#         important_iterations.append(epochs - 1)
# 
#     if batch_size is not None:
#         indices = torch.empty(samples.size(0), dtype=torch.int64)
#         for i in range(epochs):
#             torch.randperm(samples.size(0), out=indices)
#             for batch in torch.split(indices, batch_size):
#                 optimiser.zero_grad()
#                 ψ_real, ψ_imag, _ = _swo_to_explicit(ψ, samples[batch])
#                 loss = negative_log_overlap(
#                     (φ_real[batch], φ_imag[batch]), (ψ_real, ψ_imag)
#                 )
#                 # logging.info("{}: loss = {}".format(100 * (i + 1) // epochs, loss))
#                 loss.backward()
#                 optimiser.step()
#             if i in important_iterations:
#                 with torch.no_grad():
#                     ψ_real, ψ_imag, _ = _swo_to_explicit(ψ, samples)
#                     loss = negative_log_overlap((φ_real, φ_imag), (ψ_real, ψ_imag))
#                 logging.info("{}%: loss = {}".format(100 * (i + 1) // epochs, loss))
#                 save(i)
#     else:
#         for i in range(epochs):
#             optimiser.zero_grad()
#             ψ_real, ψ_imag, _ = _swo_to_explicit(ψ, samples)
#             loss = negative_log_overlap((φ_real, φ_imag), (ψ_real, ψ_imag))
#             if i in important_iterations:
#                 logging.info("{}%: loss = {}".format(100 * (i + 1) // epochs, loss))
#                 save(i)
#             loss.backward()
#             optimiser.step()
# 
# 
# @cli.command()
# @click.argument(
#     "nn-file",
#     type=click.Path(exists=True, resolve_path=True, path_type=str),
#     metavar="<arch_file>",
# )
# @click.option(
#     "-t",
#     "--train-file",
#     type=click.File(mode="rb"),
#     required=True,
#     help="File containing the explicit wave function representation as "
#     "generated by `sample`. It will be used as the training data set.",
# )
# @click.option(
#     "-i",
#     "--in-file",
#     type=click.File(mode="rb"),
#     help="File containing the initial Neural Network weights as a PyTorch "
#     "`state_dict` serialised using `torch.save`. It is up to the user to ensure "
#     "that the weights are compatible with the architecture read from <arch_file>.",
# )
# @click.option(
#     "-o",
#     "--out-file",
#     type=str,
#     required=True,
#     help="File where the final weights will be stored. It will contain a "
#     "PyTorch `state_dict` serialised using `torch.save`.",
# )
# @click.option(
#     "--lr",
#     type=click.FloatRange(min=1.0e-10),
#     default=0.05,
#     show_default=True,
#     help="Learning rate.",
# )
# @click.option(
#     "--batch-size", type=int, default=None, show_default=True, help="Learning rate."
# )
# @click.option(
#     "--epochs",
#     type=click.IntRange(min=0),
#     default=200,
#     show_default=True,
#     help="Number of learning steps to perform. Here step is defined as a single "
#     "update of the parameters.",
# )
# @click.option(
#     "--optimiser",
#     type=str,
#     default="SGD",
#     help="Optimizer to use. Valid values are names of classes in torch.optim "
#     "(i.e. 'SGD', 'Adam', 'Adamax', etc.)",
# )
# def train_amplitude(
#     nn_file, train_file, out_file, in_file, lr, optimiser, epochs, batch_size
# ):
#     """
#     Supervised learning.
#     """
#     logging.basicConfig(
#         format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.DEBUG
#     )
#     φ_dict, number_spins = load_explicit(train_file)
#     logging.info("Generating training data set...")
#     samples = torch.empty((len(φ_dict), number_spins), dtype=torch.float32)
#     φ = torch.empty((len(φ_dict),), dtype=torch.float32)
#     for (i, (key, value)) in enumerate(φ_dict.items()):
#         samples[i, :] = torch.from_numpy(key.numpy())
#         φ[i] = value.real
# 
#     φ *= 100
# 
#     ψ = import_network(nn_file)(number_spins)
#     if in_file is not None:
#         logging.info("Reading initial state...")
#         ψ.load_state_dict(torch.load(in_file))
# 
#     # NOTE(twesterhout): This is a hack :)
#     optimiser = getattr(torch.optim, optimiser)(
#         filter(lambda p: p.requires_grad_, ψ.parameters()), lr=lr
#     )
# 
#     def save(i):
#         out_file_name = "{}.{}.weights".format(out_file, i)
#         torch.save(ψ.state_dict(), out_file_name)
# 
#     important_iterations = list(range(0, epochs, epochs // 10))
#     if epochs - 1 not in important_iterations:
#         important_iterations.append(epochs - 1)
# 
#     if batch_size is not None:
#         indices = torch.empty(samples.size(0), dtype=torch.int64)
#         for i in range(epochs):
#             torch.randperm(samples.size(0), out=indices)
#             for batch in torch.split(indices, batch_size):
#                 optimiser.zero_grad()
#                 # ψ_ = _apply_normalised(ψ, samples[batch])
#                 # loss = torch.sum(torch.log(torch.cosh(φ[batch] - ψ_)))
#                 # logging.info("{}: loss = {}".format(100 * (i + 1) // epochs, loss))
#                 x = φ[batch] - torch.exp(ψ(samples[batch])).view(-1)
#                 loss = torch.dot(x, x)
#                 loss.backward()
#                 optimiser.step()
#             if i in important_iterations:
#                 with torch.no_grad():
#                     # ψ_ = _apply_normalised(ψ, samples)
#                     # loss = torch.sum(torch.log(torch.cosh(φ - ψ_)))
#                     x = φ - torch.exp(ψ(samples)).view(-1)
#                     loss = torch.dot(x, x)
#                 logging.info("{}%: loss = {}".format(100 * (i + 1) // epochs, loss))
#                 save(i)
#     else:
#         for i in range(epochs):
#             optimiser.zero_grad()
#             # x = φ - torch.exp(ψ(samples)).view(-1)
#             # loss = torch.dot(x, x)
#             loss = negative_log_overlap_real(φ, torch.exp(ψ(samples).view(-1)))
#             if i in important_iterations:
#                 logging.info("{}%: loss = {}".format(100 * (i + 1) // epochs, loss))
#                 save(i)
#             loss.backward()
#             optimiser.step()
# 
# 
# @cli.command()
# @click.argument(
#     "nn-file",
#     type=click.Path(exists=True, resolve_path=True, path_type=str),
#     metavar="<arch_file>",
# )
# @click.option(
#     "-t",
#     "--train-file",
#     type=click.File(mode="rb"),
#     required=True,
#     help="File containing the explicit wave function representation as "
#     "generated by `sample`. It will be used as the training data set.",
# )
# @click.option(
#     "-i",
#     "--in-file",
#     type=click.File(mode="rb"),
#     required=True,
#     help="File containing the initial Neural Network weights as a PyTorch "
#     "`state_dict` serialised using `torch.save`. It is up to the user to ensure "
#     "that the weights are compatible with the architecture read from <arch_file>.",
# )
# @click.option(
#     "-o",
#     "--out-file",
#     type=click.File(mode="wb"),
#     required=True,
#     help="File where the final weights will be stored. It will contain a "
#     "PyTorch `state_dict` serialised using `torch.save`.",
# )
# def analyse_amplitude(nn_file, train_file, out_file, in_file):
#     """
#     Supervised learning.
#     """
#     logging.basicConfig(
#         format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.DEBUG
#     )
#     φ_dict, number_spins = load_explicit(train_file)
#     logging.info("Generating training data set...")
#     samples = torch.empty((len(φ_dict), number_spins), dtype=torch.float32)
#     φ_phase = torch.empty((len(φ_dict),), dtype=torch.float32)
#     for (i, (key, value)) in enumerate(φ_dict.items()):
#         samples[i, :] = torch.from_numpy(key.numpy())
#         φ_phase[i] = math.atan2(value.imag, value.real)
# 
#     ψ = import_network(nn_file)(number_spins)
#     logging.info("Reading initial state...")
#     ψ.load_state_dict(torch.load(in_file))
# 
#     ψ_abs = torch.exp(ψ(samples)).view(-1)
# 
#     for i in range(samples.size(0)):
#         out_file.write(
#             "{}\t{:.10e}\t{:.10e}\n".format(
#                 Spin.from_array(samples[i].numpy()),
#                 ψ_abs[i] * math.cos(φ_phase[i]),
#                 ψ_abs[i] * math.sin(φ_phase[i]),
#             ).encode("utf-8")
#         )
# 
# 
# @cli.command()
# @click.argument(
#     "nn-file",
#     type=click.Path(exists=True, resolve_path=True, path_type=str),
#     metavar="<arch_file>",
# )
# @click.option(
#     "-t",
#     "--train-file",
#     type=click.File(mode="rb"),
#     required=True,
#     help="File containing the explicit wave function representation as "
#     "generated by `sample`. It will be used as the training data set.",
# )
# @click.option(
#     "-i",
#     "--in-file",
#     type=click.File(mode="rb"),
#     help="File containing the initial Neural Network weights as a PyTorch "
#     "`state_dict` serialised using `torch.save`. It is up to the user to ensure "
#     "that the weights are compatible with the architecture read from <arch_file>.",
# )
# @click.option(
#     "-o",
#     "--out-file",
#     type=str,
#     required=True,
#     help="File where the final weights will be stored. It will contain a "
#     "PyTorch `state_dict` serialised using `torch.save`.",
# )
# @click.option(
#     "--lr",
#     type=click.FloatRange(min=1.0e-10),
#     default=0.05,
#     show_default=True,
#     help="Learning rate.",
# )
# @click.option(
#     "--batch-size", type=int, default=None, show_default=True, help="Learning rate."
# )
# @click.option(
#     "--epochs",
#     type=click.IntRange(min=0),
#     default=200,
#     show_default=True,
#     help="Number of learning steps to perform. Here step is defined as a single "
#     "update of the parameters.",
# )
# @click.option(
#     "--optimiser",
#     type=str,
#     default="SGD",
#     help="Optimizer to use. Valid values are names of classes in torch.optim "
#     "(i.e. 'SGD', 'Adam', 'Adamax', etc.)",
# )
# def train_phase(
#     nn_file, train_file, out_file, in_file, lr, optimiser, epochs, batch_size
# ):
#     """
#     Supervised learning.
#     """
#     logging.basicConfig(
#         format="[%(asctime)s] [%(levelname)s] %(message)s",
#         datefmt="%H:%M:%S",
#         level=logging.DEBUG,
#     )
# 
#     φ_dict, number_spins = load_explicit(train_file)
#     # φ_dict_log = dict(map(lambda x: (x[0], cmath.log(x[1])), φ_dict.items()))
#     φ = ExplicitState(φ_dict, apply_log=True)
# 
#     logging.info("Generating training data set...")
#     do_monte_carlo = True
#     magnetisation = 0 if number_spins % 2 == 0 else 1
#     if do_monte_carlo:
#         number_chains = 3
#         steps = 2000
#         thermalisation = steps // 10
#         monte_carlo_steps = (
#             thermalisation * number_spins,
#             (thermalisation + steps) * number_spins,
#             1,
#         )
#         samples = sample_some(φ, (number_chains,) + monte_carlo_steps, magnetisation)
#         logging.info("#samples: {}".format(samples.size(0)))
#     else:
#         samples = all_spins(number_spins, magnetisation)
# 
#     ψ = import_network(nn_file)(number_spins)
#     if in_file is not None:
#         logging.info("Reading initial state...")
#         ψ.load_state_dict(torch.load(in_file))
#     else:
#         logging.info("Starting with random parameters...")
# 
#     # _train_phase(ψ, φ, samples, epochs, lr)
# 
#     # ψ = import_network("small.py")(number_spins)
#     # logging.info("Starting with random parameters...")
# 
#     _train_amplitude(ψ, φ, samples, epochs, lr)
# 
#     ψ_phase = import_network("phase-1.py")(number_spins)
#     ψ_phase.load_state_dict(torch.load("Kagome-18.phase.1.weights"))
# 
#     φ_xs = φ(samples)
#     φ_scale = normalisation_constant(φ_xs[:, 0])
#     φ_real = torch.empty(samples.size(0), dtype=torch.float32)
#     φ_imag = torch.empty(samples.size(0), dtype=torch.float32)
#     for i in range(samples.size(0)):
#         t = cmath.exp(complex(φ_xs[i, 0] + φ_scale, φ_xs[i, 1]))
#         φ_real[i] = t.real
#         φ_imag[i] = t.imag
# 
#     ψ_combo = CombiningState(ψ, ψ_phase)
#     ψ_xs = ψ_combo(samples)
#     ψ_scale = normalisation_constant(ψ_xs[:, 0])
#     ψ_real = torch.empty(samples.size(0), dtype=torch.float32)
#     ψ_imag = torch.empty(samples.size(0), dtype=torch.float32)
#     for i in range(samples.size(0)):
#         t = cmath.exp(complex(ψ_xs[i, 0] + ψ_scale, ψ_xs[i, 1]))
#         ψ_real[i] = t.real
#         ψ_imag[i] = t.imag
# 
#     logging.info(
#         "Negative log overlap: {}".format(
#             negative_log_overlap((ψ_real, ψ_imag), (φ_real, φ_imag))
#         )
#     )
# 
#     with open("hello_world.dat", "wb") as f:
#         for i in range(samples.size(0)):
#             f.write(
#                 "{}\t{:.10e}\t{:.10e}\n".format(
#                     Spin.from_array(samples[i].numpy()), ψ_real[i], ψ_imag[i]
#                 ).encode("utf-8")
#             )
#     return
# 
#     target = (lambda y: torch.atan2(y[:, 1], y[:, 0]))(φ(samples))
#     target -= (torch.max(target) + torch.min(target)) / 2
#     target = ((target.sign() + 1) // 2).long()
#     print(target)
# 
#     # NOTE(twesterhout): This is a hack :)
#     optimiser = getattr(torch.optim, optimiser)(
#         filter(lambda p: p.requires_grad_, ψ.parameters()), lr=lr
#     )
# 
#     def accuracy():
#         with torch.no_grad():
#             _, prediction = torch.max(ψ(samples), dim=1)
#         return float(torch.sum(target == prediction)) / target.size(0)
# 
#     def save(i):
#         out_file_name = "{}.{}.weights".format(out_file, i)
#         torch.save(ψ.state_dict(), out_file_name)
# 
#     if epochs < 10:
#         important_iterations = list(range(epochs))
#     else:
#         important_iterations = list(range(0, epochs, epochs // 10))
#         if epochs - 1 not in important_iterations:
#             important_iterations.append(epochs - 1)
# 
#     loss_fn = torch.nn.CrossEntropyLoss()
# 
#     logging.info("Initial accuracy: {:.2f}%".format(100 * accuracy()))
# 
#     if batch_size is not None:
#         raise NotImplementedError()
#         indices = torch.empty(samples.size(0), dtype=torch.int64)
#         for i in range(epochs):
#             torch.randperm(samples.size(0), out=indices)
#             for batch in torch.split(indices, batch_size):
#                 optimiser.zero_grad()
#                 # ψ_ = _apply_normalised(ψ, samples[batch])
#                 # loss = torch.sum(torch.log(torch.cosh(φ[batch] - ψ_)))
#                 # logging.info("{}: loss = {}".format(100 * (i + 1) // epochs, loss))
#                 x = φ[batch] - torch.exp(ψ(samples[batch])).view(-1)
#                 loss = torch.dot(x, x)
#                 loss.backward()
#                 optimiser.step()
#             if i in important_iterations:
#                 with torch.no_grad():
#                     # ψ_ = _apply_normalised(ψ, samples)
#                     # loss = torch.sum(torch.log(torch.cosh(φ - ψ_)))
#                     x = φ - torch.exp(ψ(samples)).view(-1)
#                     loss = torch.dot(x, x)
#                 logging.info("{}%: loss = {}".format(100 * (i + 1) // epochs, loss))
#                 save(i)
#     else:
#         for i in range(epochs):
#             optimiser.zero_grad()
#             # x = φ - torch.exp(ψ(samples)).view(-1)
#             # loss = torch.dot(x, x)
# 
#             loss = loss_fn(ψ(samples), target)
#             if i in important_iterations:
#                 logging.info("{}%: loss = {}".format(100 * (i + 1) // epochs, loss))
#                 save(i)
#             loss.backward()
#             optimiser.step()
# 
#     logging.info("Final accuracy: {:.2f}".format(100 * accuracy()))
# 
# 
# @cli.command()
# @click.argument(
#     "amplitude-nn-file",
#     type=click.Path(exists=True, resolve_path=True, path_type=str),
#     metavar="<amplitude torch.nn.Module>",
# )
# @click.argument(
#     "phase-nn-file",
#     type=click.Path(exists=True, resolve_path=True, path_type=str),
#     metavar="<phase torch.nn.Module>",
# )
# @click.option(
#     "--in-amplitude",
#     type=click.File(mode="rb"),
#     help="File containing the Neural Network weights as a PyTorch `state_dict` "
#     "serialised using `torch.save`. It is up to the user to ensure that "
#     "the weights are compatible with the architecture read from "
#     "<arch_file>.",
# )
# @click.option(
#     "--in-phase",
#     type=click.File(mode="rb"),
#     help="File containing the Neural Network weights as a PyTorch `state_dict` "
#     "serialised using `torch.save`. It is up to the user to ensure that "
#     "the weights are compatible with the architecture read from "
#     "<arch_file>.",
# )
# @click.option(
#     "-o",
#     "--out-file",
#     type=str,
#     help="Where to save the final state to. It will contain a "
#     "PyTorch `state_dict` serialised using `torch.save`. If no file is "
#     "specified, the result will be discarded.",
# )
# @click.option(
#     "-H",
#     "--hamiltonian",
#     "hamiltonian_file",
#     type=click.File(mode="r"),
#     required=True,
#     help="File containing the Heisenberg Hamiltonian specifications.",
# )
# @click.option(
#     "--steps",
#     type=click.IntRange(min=1),
#     default=2000,
#     show_default=True,
#     help="Length of the Markov Chain.",
# )
# def energy(
#     amplitude_nn_file,
#     phase_nn_file,
#     in_amplitude,
#     in_phase,
#     out_file,
#     hamiltonian_file,
#     steps,
# ):
#     logging.basicConfig(
#         format="[%(asctime)s] [%(levelname)s] %(message)s",
#         datefmt="%H:%M:%S",
#         level=logging.DEBUG,
#     )
#     H = read_hamiltonian(hamiltonian_file)
#     number_spins = H.number_spins
#     magnetisation = 0 if number_spins % 2 == 0 else 1
# 
#     thermalisation = steps // 10
#     m_c_steps = (
#         thermalisation * number_spins,
#         (thermalisation + steps) * number_spins,
#         number_spins,
#     )
# 
#     ψ_amplitude = import_network(amplitude_nn_file)(number_spins)
#     logging.info("Reading initial state...")
#     ψ_amplitude.load_state_dict(torch.load(in_amplitude))
# 
#     ψ_phase = import_network(phase_nn_file)(number_spins)
#     logging.info("Reading initial state...")
#     ψ_phase.load_state_dict(torch.load(in_phase))
# 
#     ψ = CombiningState(ψ_amplitude, ψ_phase)
#     machine = Machine(ψ)
# 
#     samples = sample_one(machine, m_c_steps, unique=False)
# 
#     E = np.empty((samples.size(0),), dtype=np.complex64)
#     for i in range(samples.size(0)):
#         E[i] = H(MonteCarloState(machine=machine, spin=samples[i].numpy(), weight=None))
# 
#     logging.info("E = {} ± {}".format(np.mean(E), np.std(E)))
# 
# 
# if __name__ == "__main__":
#     cli()
#     # cProfile.run('main()')
