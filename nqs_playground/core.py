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
from functools import reduce
import importlib
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch

from . import _C_nqs
from ._C_nqs import CompactSpin


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
        :param state:
            a dictionary mapping spins ``σ`` to their corresponding ``⟨σ|ψ⟩``
            or ``log(⟨σ|ψ⟩)``.
        :param apply_log:
            specifies whether ``log_wf`` and co. should apply logarithm to
            the values in the dictionary.
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
            return self._c2t(self._state.get(CompactSpin(x), self._default))
        else:
            assert x.dim() == 2
            out = torch.empty((x.size(0), 2), dtype=torch.float32)
            for i in range(x.size(0)):
                out[i] = self._c2t(self._state.get(CompactSpin(x[i]), self._default))
            return out

    def backward(self, _):
        """
        No backward propagation, because there are no parameters to train.
        """
        raise NotImplementedError()


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
        expected_dot_predicted
        * expected_dot_predicted
        / (sqr_l2_expected * sqr_l2_predicted)
    )


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


# "Borrowed" from pytorch/torch/serialization.py.
# All credit goes to PyTorch developers.
def _with_file_like(f, mode, body):
    """
    Executes a body function with a file object for f, opening
    it in 'mode' if it is a string filename.
    """
    new_fd = False
    if (
        isinstance(f, str)
        or (sys.version_info[0] == 2 and isinstance(f, unicode))
        or (sys.version_info[0] == 3 and isinstance(f, pathlib.Path))
    ):
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
            raise ValueError(
                "A single state cannot contain spin "
                "configurations of different sizes"
            )
        psi[CompactSpin(spin)] = complex(float(real), float(imag))
    return psi, number_spins


def load_explicit(stream):
    return _with_file_like(stream, "rb", _load_explicit)


def import_network(filename: str):
    module_name, extension = os.path.splitext(os.path.basename(filename))
    module_dir = os.path.dirname(filename)
    if extension != ".py":
        raise ValueError(
            "Could not import the network from '{}': not a python source file.".format(
                filename
            )
        )
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    sys.path.pop(0)
    return module.Net
