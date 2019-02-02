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
import mpmath  # Just to be safe: for accurate computation of L2 norms
from numba import jit, jitclass, uint8, int64, uint64, float32
from numba.types import Bytes
import numba.extending
import numpy as np
import scipy
from scipy.sparse.linalg import lgmres, LinearOperator
import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################################
# NQS
################################################################################


@jit(uint64(uint64), nopython=True)
def _hash_uint64(x: int) -> int:
    """
    Hashes a 64-bit integer.

    .. note::

       The algorithm is taken from https://stackoverflow.com/a/12996028.
    """
    x = (x ^ (x >> 30)) * uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> 27)) * uint64(0x94D049BB133111EB)
    x = x ^ (x >> 31)
    return x


@jitclass([("_d0", uint64), ("_d1", uint64), ("_d2", uint64), ("_size", int64)])
class _CompactSpin(object):
    """
    .. warning::
        
        This class is an implementation detail! **Use at your own risk.**

    Compact representation of a spin configuration.

    Spin quantum numbers are represented by bits:

      * unset (``0``) means *spin down* and
      * set (``1``) means *spin up*.

    Currently, the maximum supported size is 192.
    """

    def __init__(self, d0: uint64, d1: uint64, d2: uint64, n: uint64):
        """
        .. warning::

           **Do not use this function unless you know what you're doing!**

        ``_d0``, ``_d1``, and ``_d2`` form a bitarray of at most 192 bits.
        """
        assert n > 0
        self._d0 = d0
        self._d1 = d1
        self._d2 = d2
        self._size = n

    @property
    def size(self) -> uint64:
        """
        Returns the number of spins.
        """
        return self._size

    @property
    def hash(self) -> uint64:
        """
        Computes the hash of the spin configuration.

        .. note::

           The algorithm is taken from ``boost::hash_combine``.
        """
        seed = _hash_uint64(self._d0)
        seed ^= _hash_uint64(self._d1) + uint64(0x9E3779B9) + (seed << 6) + (seed >> 2)
        seed ^= _hash_uint64(self._d2) + uint64(0x9E3779B9) + (seed << 6) + (seed >> 2)
        return seed

    def as_int(self):
        """
        Returns the integer representation of the spin configuration.

        .. warning::

           Only spin configurations shorter than 64 spins support this function.
        """
        if self._size > 63:
            raise OverflowError(
                "Spin configuration is too long to be represented by an int"
            )
        return self._d2

    def as_str(self) -> np.ndarray:
        """
        Returns the string representation of the spin configuration.

        .. warning::

           This function is quite slow. Do not use it in performance-critical
           parts.

        :return: NumPy array of ``uint8``.
        """
        code_of_zero = 48
        code_of_one = 49
        s = np.empty(self._size, dtype=np.uint8)
        for i in range(self._size):
            s[i] = code_of_one if self.get(i) else code_of_zero
        return s

    def equal_to(self, other) -> bool:
        """
        :return: ``self == other``.
        """
        assert self._size == other._size
        return self._d0 == other._d0 and self._d1 == other._d1 and self._d2 == other._d2

    def less_than(self, other) -> bool:
        """
        :return: ``self < other``.
        """
        assert self._size == other._size
        if self._d0 < other._d0:
            return True
        elif self._d0 > other._d0:
            return False
        elif self._d1 < other._d1:
            return True
        elif self._d1 > other._d1:
            return False
        elif self._d2 < other._d2:
            return True
        else:
            return False

    def less_or_equal_to(self, other) -> bool:
        """
        :return: ``self <= other``.
        """
        assert self._size == other._size
        if self._d0 < other._d0:
            return True
        elif self._d0 > other._d0:
            return False
        elif self._d1 < other._d1:
            return True
        elif self._d1 > other._d1:
            return False
        elif self._d2 <= other._d2:
            return True
        else:
            return False

    def get(self, i):
        """
        Returns the spin at index ``i``.
        """
        i = self._size - 1 - i
        chunk = i // 64
        if chunk == 0:
            return (self._d2 >> i) & 0x01
        elif chunk == 1:
            return (self._d1 >> (i - 64)) & 0x01
        else:
            assert chunk == 2
            return (self._d0 >> (i - 128)) & 0x01


@jit(numba.types.uint32(numba.types.uint32), nopython=True)
def _popcount_32(v: int) -> int:
    """
    .. warning::

       **Do not use this function unless you know what you're doing!**

    :return: Number of set (1) bits in a 32-bit integer.
    """
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    return (((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) & 0xFFFFFFFF) >> 24


@jit(uint64(uint64), nopython=True)
def _popcount(x: int) -> int:
    """
    .. warning::

       **Do not use this function unless you know what you're doing!**

    :return: Number of set (1) bits in a 64-bit integer.
    """
    return _popcount_32(x & uint64(0x00000000FFFFFFFF)) + _popcount_32(x >> 32)


@jit(
    locals={
        "chunks": uint64,
        "rest": uint64,
        "d0": uint64,
        "d1": uint64,
        "d2": uint64,
        "i": uint64,
    },
    nopython=True,
)
def _array2spin(x: np.ndarray) -> _CompactSpin:
    """
    Given a spin configuration as a numpy arrray, returns the compact
    representation of it.
    """
    chunks = x.size // 64
    rest = x.size % 64
    d0 = 0
    d1 = 0
    d2 = 0
    if chunks == 0:
        for i in range(rest):
            d2 |= uint64(x[i] > 0) << (rest - 1 - i)
    elif chunks == 1:
        for i in range(rest):
            d1 |= uint64(x[i] > 0) << (rest - 1 - i)
        for i in range(64):
            d2 |= uint64(x[rest + i] > 0) << (63 - i)
    else:
        assert chunks == 2
        for i in range(rest):
            d0 |= uint64(x[i] > 0) << (rest - 1 - i)
        for i in range(64):
            d1 |= uint64(x[rest + i] > 0) << (63 - i)
        for i in range(64):
            d2 |= uint64(x[64 + rest + i] > 0) << (63 - i)
    return _CompactSpin(d0, d1, d2, x.size)


@jit(
    locals={"size": uint64, "chunks": uint64, "rest": uint64, "t": uint64, "i": int64},
    nopython=True,
)
def _spin2array(
    spin: _CompactSpin, out: Optional[np.ndarray] = None, dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Unpacks a compact spin into a numpy array.
    """
    size = spin.size
    if out is None:
        out = np.empty(size, dtype=dtype)
    chunks = size // 64
    rest = size % 64
    if chunks == 0:
        t = spin._d2
        i = size - 1
        for _ in range(rest):
            out[i] = -1.0 + 2.0 * (t & 0x01)
            t >>= 1
            i -= 1
        assert i == -1
    elif chunks == 1:
        t = spin._d2
        i = size - 1
        for _ in range(64):
            out[i] = -1.0 + 2.0 * (t & 0x01)
            t >>= 1
            i -= 1
        assert i == rest - 1
        t = spin._d1
        for _ in range(rest):
            out[i] = -1.0 + 2.0 * (t & 0x01)
            t >>= 1
            i -= 1
        assert i == -1
    else:  # chunks == 2
        assert chunks == 2
        t = spin._d2
        i = size - 1
        for _ in range(64):
            out[i] = -1.0 + 2.0 * (t & 0x01)
            t >>= 1
            i -= 1
        t = spin._d1
        for _ in range(64):
            out[i] = -1.0 + 2.0 * (t & 0x01)
            t >>= 1
            i -= 1
        assert i == rest - 1
        t = spin._d0
        for _ in range(rest):
            out[i] = -1.0 + 2.0 * (t & 0x01)
            t >>= 1
            i -= 1
        assert i == -1
    return out


class Spin(object):
    """
    Compact representation of a spin configuration.
    """

    def __init__(self, x):
        """
        .. warning::

           **Do not use this function unless you know what you're doing!**
           Instead, have a look at :py:func:`Spin.from_str` and :py:func:`from_array` functions.
        """
        assert isinstance(x, _CompactSpin)
        self._jitted = x

    def __copy__(self):
        """
        :return: a copy of the spin configuration.
        """
        return Spin(
            _CompactSpin(
                self._jitted._d0, self._jitted._d1, self._jitted._d2, self._jitted._size
            )
        )

    def __deepcopy__(self):
        """
        :return: a copy of the spin configuration.
        """
        return self.__copy__()

    def __str__(self) -> str:
        """
        .. warning::

           This function is quite slow, so don't use it in performance-critical
           parts.

        :return: the string representation of the spin configuration.
        """
        return bytes(self._jitted.as_str().data).decode("utf-8")

    def __int__(self) -> int:
        """
        .. note::

           This function is only works for spin chains shorted than 64

        :return: the integer representation of the spin configuration.
        """
        return self._jitted.as_int()

    def __len__(self) -> int:
        """
        :return: number of spins in the spin configuration
        """
        return self._jitted._size

    def __hash__(self) -> int:
        """
        Calculates the hash of the spin configuration.
        """
        return self._jitted.hash

    def __eq__(self, other) -> bool:
        """
        :return: ``self == other``
        """
        return self._jitted.equal_to(other._jitted)

    def __lt__(self, other) -> bool:
        """
        :return: ``self < other``
        """
        return self._jitted.less_than(other._jitted)

    def __le__(self, other) -> bool:
        """
        :return: ``self <= other``
        """
        return self._jitted.less_or_equal_to(other._jitted)

    def __gt__(self, other) -> bool:
        """
        :return: ``self > other``
        """
        return other._jitted.less_than(self._jitted)

    def __ge__(self, other) -> bool:
        """
        :return: ``self >= other``
        """
        return other._jitted.less_or_equal_to(self._jitted)

    def numpy(self) -> np.ndarray:
        """
        Unpacks the spin configuration into a numpy array of ``float32``. Spin
        down is represented by ``-1.0`` and spin up -- by ``1.0``.
        """
        return _spin2array(self._jitted)

    @property
    def size(self) -> int:
        """
        :return: number of spins in the spin configuration
        """
        return self._jitted._size

    @staticmethod
    def from_array(x: np.ndarray):
        """
        Packs a numpy array of ``float32`` into a ``Spin``.
        """
        return Spin(_array2spin(x))

    @staticmethod
    def from_str(s: str):
        """
        Constructs a spin configuration from the string representation: '0' means spin down
        and '1' means spin up.

        .. warning:: Do not use this function in performance-critical code!
        """
        n = len(s)
        if n > 192:
            raise OverflowError(
                "Spin configurations longer than 192 are not (yet) supported."
            )
        chunks = n // 64
        if chunks == 0:
            return Spin(_CompactSpin(0, 0, uint64(int(s, base=2)), n))
        elif chunks == 1:
            return Spin(
                _CompactSpin(
                    0, uint64(int(s[:-64], base=2)), uint64(int(s[-64:], base=2)), n
                )
            )
        else:
            assert chunks == 2
            return Spin(
                _CompactSpin(
                    uint64(int(s[:-128], base=2)),
                    uint64(int(s[-128:-64], base=2)),
                    uint64(int(s[-64:], base=2)),
                    n,
                )
            )


def random_spin(n: int, magnetisation: int = None) -> np.ndarray:
    """
    :return:
        a random spin configuration of length ``n``
        with magnetisation ``magnetisation``.
    """
    if n <= 0:
        raise ValueError("Invalid number of spins: {}".format(n))
    if magnetisation is not None:
        if abs(magnetisation) > n:
            raise ValueError(
                "Magnetisation exceeds the number of spins: |{}| > {}".format(
                    magnetisation, n
                )
            )
        if (n + magnetisation) % 2 != 0:
            raise ValueError("Invalid magnetisation: {}".format(magnetisation))
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


class _Machine(torch.nn.Module):
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
        Same as ``_Machine.ψ``.

        :return: underlying neural network
        """
        return self.ψ

    def _log_wf(self, σ: np.ndarray):
        """
        Given a spin configuration ``σ``, calculates ``log(⟨σ|ψ⟩)``
        and returns it wrapped in a ``Cell``.
        """
        (amplitude, phase) = self._ψ.forward(torch.from_numpy(σ))
        return _Machine.Cell(log_wf=complex(amplitude, phase), der_log_wf=None)

    def log_wf(self, σ: np.ndarray) -> complex:
        """
        Given a spin configuration ``σ``, returns ``log(⟨σ|ψ⟩)``.

        :param np.ndarray σ:
            Spin configuration. Must be a numpy array of ``float32``.
        """
        compact_spin = Spin.from_array(σ)
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
        return _Machine.Cell(log_wf=complex(amplitude, phase), der_log_wf=der_log_wf)

    def der_log_wf(self, σ: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes ``∇log(⟨σ|Ψ⟩) = ∂log(⟨σ|ψ⟩)/∂Wᵢ`` and saves it to out.

        .. warning:: Don't modify the returned array!

        :param np.ndarray σ:
            Spin configuration. Must be a numpy array of ``float32``.
        :param Optional[np.ndarray] out:
            Destination array. Must be a numpy array of ``complex64``.
        """
        compact_spin = Spin.from_array(σ)
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

    # def forward(self, x, use_cache=False):
    #     if not use_cache:
    #         return self._ψ.forward(x)
    #     c2t = lambda z: torch.tensor([z.real, z.imag], dtype=torch.float32)
    #     if x.dim() == 1:
    #         return c2t(self.log_wf(x.numpy()))
    #     else:
    #         assert x.dim() == 2
    #         out = torch.empty((x.size(0), 2), dtype=torch.float32)
    #         for i in range(x.size(0)):
    #             out[i] = c2t(self.log_wf(x[i].numpy()))
    #         return out

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


Machine = _Machine


class ExplicitState(torch.nn.Module):
    """
    Wraps a ``Dict[Spin, complex]`` into a ``torch.nn.Module`` which can
    be used to construct a ``_Machine``.
    """

    def __init__(self, state: Dict[Spin, complex], apply_log=False):
        """
        If ``apply_log=False``, then ``state`` is a dictionary mapping spins ``σ``
        to their corresponding ``log(⟨σ|ψ⟩)``.  Otherwise, ``state`` maps ``σ``
        to ``⟨σ|ψ⟩``.
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
    def dict(self) -> Dict[Spin, complex]:
        """
        :return: the underlying dictionary
        """
        return self._state

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.dim() == 1:
            return self._c2t(
                self._state.get(Spin.from_array(x.detach().numpy()), self._default)
            )
        else:
            assert x.dim() == 2
            out = torch.empty((x.size(0), 2), dtype=torch.float32)
            for i in range(x.size(0)):
                out[i] = self._c2t(
                    self._state.get(
                        Spin.from_array(x[i].detach().numpy()), self._default
                    )
                )
            return out

    def backward(self, x):
        """
        No backward propagation, because there are no parameters to train.
        """
        raise NotImplementedError()


class CombiningState(torch.nn.Module):
    """
    Given two neural networks: one mapping ``σ``s to log amplitudes ``r``s and
    the other mapping ``σ`` to phases ``φ``s, combines them into a single neural
    network mapping ``σ``s to ``r + iφ``s.
    """

    def __init__(self, log_amplitude: torch.nn.Module, phase: torch.nn.Module):
        super().__init__()
        assert log_amplitude.number_spins == phase.number_spins
        self._log_amplitude = log_amplitude
        self._phase = phase

    @property
    def number_spins(self):
        """
        :return: number of spins in the system.
        """
        return self._log_amplitude.number_spins

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.dim() == 1:
            return torch.cat(
                [
                    self._log_amplitude(x),
                    math.pi * torch.max(self._phase(x), dim=0, keepdim=True)[1].float(),
                ]
            )
        else:
            assert x.dim() == 2
            return torch.cat(
                [
                    self._log_amplitude(x),
                    math.pi * torch.max(self._phase(x), dim=1)[1].view(-1, 1).float(),
                ],
                dim=1,
            )

    def backward(self, x):
        raise NotImplementedError()


################################################################################
# Markov chains
################################################################################


"""
A Monte Carlo state is an element of the Markov Chain and is a triple
``(wᵢ, σᵢ, ψ)``, where ``wᵢ`` is the *weight*, ``σᵢ`` is the current
spin configuration, and ``ψ`` is the NQS.
"""
MonteCarloState = collections.namedtuple(
    "MonteCarloState", ["weight", "spin", "machine"]
)


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
            samples_set.add(Spin.from_array(s))

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
            ValueError(
                "Invalid graph: Counting from 0, but the minimal index "
                "present is {}.".format(smallest)
            )
        self._number_spins = largest + 1

    def __call__(self, state: MonteCarloState, cutoff=5.5) -> np.complex64:
        """
        Calculates local energy in the given state.
        """
        spin = state.spin

        def log_quot_wf(flips: List[int]) -> complex:
            state.spin[flips] *= -1
            log_wf = state.machine.log_wf(state.spin)
            state.spin[flips] *= -1
            return log_wf - log_quot_wf.log_wf_old

        log_quot_wf.log_wf_old = state.machine.log_wf(state.spin)

        energy = 0
        for (i, j) in self._graph:
            if spin[i] == spin[j]:
                energy += 1
            else:
                assert spin[i] == -spin[j]
                x = log_quot_wf([i, j])
                if cutoff is not None and x.real > cutoff:
                    raise WorthlessConfiguration([i, j])
                energy += -1 + 2 * cmath.exp(x)
        energy = np.complex64(energy)
        return energy if not cmath.isinf(energy) else np.complex64(1e38)

    # def reachable_from(self, spin):
    #     reachable = []
    #     for (i, j) in filter(lambda x: spin[x[0]] != spin[x[1]], self._graph):
    #         assert spin[i] == -spin[j]
    #         reachable.append(spin.copy())
    #         reachable[-1][[i, j]] *= -1
    #     return reachable

    @property
    def number_spins(self) -> int:
        """
        :return: number of spins in the system.
        """
        return self._number_spins


def read_hamiltonian(in_file):
    """
    Reads the Hamiltonian from ``in_file``.
    """
    specs = []
    for (coupling, edges) in map(
        lambda x: x.strip().split(maxsplit=1),
        filter(lambda x: not x.startswith("#"), in_file),
    ):
        coupling = float(coupling)
        # TODO: Parse the edges properly, it's not that difficult...
        edges = eval(edges)
        specs.append((coupling, edges))
    # TODO: Generalise Heisenberg to support multiple graphs with different
    # couplings
    if len(specs) != 1:
        raise NotImplementedError("Multiple couplings are not yet supported.")
    (_, edges) = specs[0]
    return Heisenberg(edges)


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


def l2_error(φ: torch.Tensor, ψ: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Computes ``1 - |⟨φ|ψ⟩| / (‖φ‖₂∙‖ψ‖₂)``.

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
        A one-element tensor ``1 - |⟨φ|ψ⟩| / (‖φ‖₂∙‖ψ‖₂)``.
    """
    Re_φ, Im_φ = φ
    Re_ψ, Im_ψ = ψ
    sqr_l2_φ = torch.dot(w, Re_φ * Re_φ) + torch.dot(w, Im_φ * Im_φ)
    sqr_l2_ψ = torch.dot(w, Re_ψ * Re_ψ) + torch.dot(w, Im_ψ * Im_ψ)
    Re_φ_dot_ψ = torch.dot(w, Re_φ * Re_ψ) + torch.dot(w, Im_φ * Im_ψ)
    Im_φ_dot_ψ = torch.dot(w, Re_φ * Im_ψ) - torch.dot(w, Im_φ * Re_ψ)
    return 1 - torch.sqrt((Re_φ_dot_ψ ** 2 + Im_φ_dot_ψ ** 2) / (sqr_l2_φ * sqr_l2_ψ))


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


def negative_log_overlap_real(φ: torch.Tensor, ψ: torch.Tensor) -> torch.Tensor:
    sqr_l2_φ = torch.dot(φ, φ)
    sqr_l2_ψ = torch.dot(ψ, ψ)
    φ_dot_ψ = torch.dot(φ, ψ)
    return -0.5 * torch.log(φ_dot_ψ ** 2 / (sqr_l2_φ * sqr_l2_ψ))
    # return 1 - (φ_dot_ψ ** 2 / (sqr_l2_φ * sqr_l2_ψ))**4


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
            map(lambda s: Spin.from_array(s.numpy()), samples),
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


def to_explicit_dict(
    ψ, steps=None, magnetisation=None, explicit=False
) -> Dict[Spin, complex]:
    ψ = ψ if isinstance(ψ, Machine) else Machine(ψ)
    old_cache = deepcopy(ψ._cache)
    ψ.clear_cache()
    result, stats = sample_state(
        ψ, steps=steps, magnetisation=magnetisation, explicit=explicit
    )

    ψ_s = torch.empty((len(ψ._cache),), dtype=torch.float32)
    for i, value in enumerate(map(lambda x: x.log_wf.real, ψ._cache.values())):
        ψ_s[i] = value
    scale = normalisation_constant(ψ_s).item()
    explicit = {}
    for spin, cell in ψ._cache.items():
        explicit[spin] = cmath.exp(cell.log_wf + complex(scale, 0.0))

    ψ._cache.update(old_cache)
    return explicit


################################################################################
# SR
################################################################################


def make_force(energies, gradients, weights) -> np.ndarray:
    mean_energy = np.dot(weights, energies)
    mean_gradient = np.dot(weights, gradients)
    gradients = gradients.conj().transpose()
    force = np.dot(weights, (energies * gradients).transpose())
    force -= mean_gradient.conj() * mean_energy
    return force, mean_energy, mean_gradient


class DenseCovariance(LinearOperator):
    """
    Dense representation of the covariance matrix matrix S in Stochastic
    Reconfiguration method [1].
    """

    def __init__(self, gradients: np.ndarray, weights: np.ndarray, regulariser: float):
        (steps, n) = gradients.shape
        super().__init__(np.float32, (2 * n, n))

        mean_gradient = np.dot(weights, gradients)
        gradients -= mean_gradient

        conj_gradients = gradients.transpose().conj()

        # Viewing weights as a column vector for proper broadcasting
        weights_view = weights.view()
        weights_view.shape = (-1, 1)
        gradients = np.multiply(weights_view, gradients, out=gradients)

        S = np.dot(conj_gradients, gradients)
        S += regulariser * np.eye(n, dtype=np.float32)

        self._matrix = np.concatenate([S.real, S.imag], axis=0)
        assert not np.any(np.isnan(self._matrix))
        assert not np.any(np.isinf(self._matrix))

    def _matvec(self, x):
        assert x.dtype == self.dtype
        return np.matmul(self._matrix, x)

    def _rmatvec(self, y):
        assert y.dtype == self.dtype
        return np.matmul(self._matrix.transpose(), y)

    def solve(self, b, x0=None):
        start = time.time()
        assert b.dtype == np.complex64
        assert not np.any(np.isnan(b))
        assert not np.any(np.isinf(b))
        logging.info("Calculating S⁻¹F...")
        b = np.concatenate([b.real, b.imag])
        assert b.dtype == np.float32
        x = scipy.linalg.lstsq(self._matrix, b)[0]
        assert x.dtype == self.dtype
        finish = time.time()
        logging.info("Done in {:.2f} seconds!".format(finish - start))
        return x


class DeltaMachine(torch.nn.Module):
    class Cell(object):
        def __init__(
            self, wave_function: complex, gradient: Optional[np.ndarray] = None
        ):
            self.log_wf = wave_function
            self.der_log_wf = gradient

    def __init__(self, ψ: torch.nn.Module, δψ: torch.nn.Module):
        """
        Given neural networks (i.e. instances of ``torch.nn.Module``) representing
        ψ and δψ, constructs an NQS given by ψ + δψ with ∂ψ/∂Wᵢ assumed to be 0.
        """
        super().__init__()
        self._ψ = ψ
        self._δψ = δψ
        if self._ψ.number_spins != self._δψ.number_spins:
            raise ValueError(
                "ψ and δψ represent systems of different number of particles"
            )
        self._size = sum(
            map(
                lambda p: reduce(int.__mul__, p.size()),
                filter(lambda p: p.requires_grad, self._δψ.parameters()),
            )
        )
        self._cache = {}

    @property
    def number_spins(self) -> int:
        return self._ψ.number_spins

    @property
    def size(self) -> int:
        return self._size

    @property
    def ψ(self):
        return self._ψ

    @property
    def δψ(self):
        return self._δψ

    def _log_wf(self, spin: np.ndarray, compact_spin: Spin) -> complex:
        cell = self._cache.get(compact_spin)
        if cell is not None:
            return cell.log_wf
        with torch.no_grad():
            log_ψ = self._ψ.log_wf(spin)
            log_δψ = _tensor2complex(self._δψ.forward(torch.from_numpy(spin)))
        log_wf = cmath.log(cmath.exp(log_ψ) + cmath.exp(log_δψ))
        self._cache[compact_spin] = DeltaMachine.Cell(log_wf)
        return log_wf

    def log_wf(self, spin: np.ndarray) -> complex:
        return self._log_wf(spin, Spin.from_array(spin))

    def _copy_grad_to(self, out: np.ndarray):
        i = 0
        for p in map(
            lambda p_: p_.grad.view(-1).numpy(),
            filter(lambda p_: p_.requires_grad, self._δψ.parameters()),
        ):
            out[i : i + p.size] = p
            i += p.size

    def _der_log_wf(self, spin: np.ndarray, compact_spin: Spin, out=None):
        if out is None:
            out = np.empty((self.size,), dtype=np.complex64)
        cell = self._cache.get(compact_spin)
        if cell is None:
            _ = self._log_wf(spin, compact_spin)
            cell = self._cache.get(compact_spin)
        elif cell.der_log_wf is not None:
            out[:] = cell.der_log_wf
            return out

        # Forward-propagation to construct the graph
        grad_δψ = np.empty((self.size,), dtype=np.complex64)
        result = self._δψ.forward(torch.from_numpy(spin))
        δψ = cmath.exp(_tensor2complex(result))
        # Computes ∇Re[log(δΨ(σ))]
        self._δψ.zero_grad()
        result.backward(torch.tensor([1, 0], dtype=torch.float32), retain_graph=True)
        self._copy_grad_to(grad_δψ.real)
        # Computes ∇Im[log(δΨ(σ))]
        self._δψ.zero_grad()
        result.backward(torch.tensor([0, 1], dtype=torch.float32))
        self._copy_grad_to(grad_δψ.imag)

        wf = cmath.exp(cell.log_wf)
        norm = abs(wf) ** 2

        # Magic :)
        out.real[:] = (wf.real * δψ.real + wf.imag * δψ.imag) / norm * grad_δψ.real
        out.real += (wf.imag * δψ.real - wf.real * δψ.imag) / norm * grad_δψ.imag

        out.imag[:] = (wf.real * δψ.imag - wf.imag * δψ.real) / norm * grad_δψ.real
        out.imag += (wf.imag * δψ.real + wf.imag * δψ.imag) / norm * grad_δψ.imag

        # Save the results
        # TODO(twesterhout): Remove the copy when it's safe to do so.
        cell.der_log_wf = np.copy(out)
        return out

    def der_log_wf(self, spin: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        return self._der_log_wf(spin, Spin.from_array(spin), out)

    def set_gradients(self, x: np.ndarray):
        """
        Performs ∇W = x, i.e. sets the gradients of the variational parameters.

        :param np.ndarray x: New value for ∇W. Must be a numpy array of
        ``float32`` of length ``self.size``.
        """
        with torch.no_grad():
            gradients = torch.from_numpy(x)
            i = 0
            for dp in map(
                lambda p_: p_.grad.data.view(-1),
                filter(lambda p_: p_.requires_grad, self._δψ.parameters()),
            ):
                (n,) = dp.size()
                dp.copy_(gradients[i : i + n])
                i += n

    def clear_cache(self):
        self._cache = {}

    def state_dict(self, *args, **kwargs):
        return self._δψ.state_dict(*args, **kwargs)


class Optimiser(object):
    def __init__(
        self,
        machine,
        hamiltonian,
        magnetisation,
        epochs,
        monte_carlo_steps,
        learning_rate,
        use_sr,
        regulariser,
        model_file,
        time_limit,
    ):
        self._machine = machine
        self._hamiltonian = hamiltonian
        self._magnetisation = magnetisation
        self._epochs = epochs
        self._monte_carlo_steps = monte_carlo_steps
        self._learning_rate = learning_rate
        self._use_sr = use_sr
        self._model_file = model_file
        self._time_limit = time_limit
        if use_sr:
            self._regulariser = regulariser
            self._delta = None
            self._optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self._machine.parameters()),
                lr=self._learning_rate,
            )
        else:
            self._optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self._machine.parameters()),
                lr=self._learning_rate,
            )

    def _monte_carlo(self):
        return monte_carlo(
            self._machine,
            self._hamiltonian,
            self._monte_carlo_steps,
            self._magnetisation,
            explicit=True,
        )

    def learning_cycle(self, iteration):
        logging.info("==================== {} ====================".format(iteration))
        # Monte Carlo
        energies, gradients, weights = self._monte_carlo()
        force, mean_energy, mean_gradient = make_force(energies, gradients, weights)
        logging.info("E = {}, ∥F∥₂ = {}".format(mean_energy, np.linalg.norm(force)))

        if self._use_sr:
            covariance = DenseCovariance(
                gradients, weights, self._regulariser(iteration)
            )
            self._delta = covariance.solve(force)
        else:
            self._delta = force.real
        logging.info("∥δW∥₂ = {}".format(np.linalg.norm(self._delta)))
        self._machine.set_gradients(self._delta)
        self._optimizer.step()
        self._machine.clear_cache()

    def __call__(self):
        if self._model_file is not None:

            def save():
                # NOTE: This is important, because we want to overwrite the
                # previous weights
                self._model_file.seek(0)
                self._model_file.truncate()
                torch.save(self._machine.ψ.state_dict(), self._model_file)

        else:
            save = lambda: None

        save()
        if self._time_limit is not None:
            start = time.time()
            for i in range(self._epochs):
                if time.time() - start > self._time_limit:
                    save()
                    start = time.time()
                self.learning_cycle(i)
        else:
            for i in range(self._epochs):
                self.learning_cycle(i)
        save()
        return self._machine


################################################################################
# SWO
################################################################################

# """
# Calculates ``log(1/‖ψ‖₂)``

#     (‖ψ‖₂)² = ∑|ψᵢ|² = ∑exp(log(|ψᵢ|²)) = ∑exp(2 * log(|ψᵢ|) - log(ψₘₐₓ) + log(ψₘₐₓ))
#             = exp(log(ψₘₐₓ)) ∙ ∑exp(2 * log(|ψᵢ|) - log(ψₘₐₓ))

# where ψₘₐₓ = max({|ψᵢ|}ᵢ). Now

#     log(1/‖ψ‖₂) = -0.5 * log((‖ψ‖₂)²) = -0.5 * [log(ψₘₐₓ) + log(∑exp(2 * log(|ψᵢ|) - log(ψₘₐₓ)))]

# :param log_ψ: torch.FloatTensor ``[log(|ψᵢ|) for ∀i]``.
# :return:      ``log(1/‖ψ‖₂)``.
# """
@torch.jit.script
def normalisation_constant(log_psi):
    max_log_amplitude = torch.max(log_psi)
    scale = -0.5 * (
        max_log_amplitude
        + torch.log(torch.sum(torch.exp(2 * log_psi.view([-1]) - max_log_amplitude)))
    )
    return scale


class TargetState(torch.nn.Module):
    """
    Represents (1 - τH)|ψ⟩.
    """

    def __init__(self, ψ, H, τ):
        super().__init__()
        self._ψ = ψ if isinstance(ψ, Machine) else Machine(ψ)
        self._ψ.clear_cache()
        self._H = H
        self._τ = τ
        for p in self._ψ.parameters():
            p.requires_grad = False

    def _forward_single(self, σ: torch.FloatTensor):
        with torch.no_grad():
            σ = σ.numpy()
            E_loc = self._H(
                MonteCarloState(spin=σ, machine=self._ψ, weight=None), cutoff=None
            )
            log_wf = self._ψ.log_wf(σ) + cmath.log(1.0 - self._τ * E_loc)
            return torch.tensor([log_wf.real, log_wf.imag], dtype=torch.float32)

    @property
    def number_spins(self):
        return self._ψ.number_spins

    def forward(self, x):
        if x.dim() == 1:
            return self._forward_single(x)
        else:
            assert x.dim() == 2
            out = torch.empty((x.size(0), 2), dtype=torch.float32, requires_grad=False)
            for i in range(x.size(0)):
                out[i] = self._forward_single(x[i])
            return out


# class ChebyshevTargetState(torch.nn.Module):
#     jk


def _swo_to_explicit(f, xs):
    f_xs = f(xs)
    scale = normalisation_constant(f_xs[:, 0])
    amplitude = torch.exp(f_xs[:, 0] + scale)
    real = amplitude * torch.cos(f_xs[:, 1])
    imag = amplitude * torch.sin(f_xs[:, 1])
    return real, imag, scale


def _apply_normalised(f, xs):
    f_xs = f(xs)
    max_log_amplitude = torch.max(f_xs)
    scale = -0.5 * (
        max_log_amplitude
        + torch.log(torch.sum(torch.exp(2 * f_xs - max_log_amplitude)))
    )
    return torch.exp(f_xs + scale)


def _swo_compute_weights(f, g):
    f_real, f_imag = f
    g_real, g_imag = g
    δ_real = f_real - g_real
    δ_imag = f_imag - g_imag
    weights = δ_real * δ_real + δ_imag * δ_imag
    weights *= 1.0 / weights.sum()
    return weights


def optimise_inner(
    optimiser, ψ: torch.nn.Module, H, samples: torch.Tensor, epochs: int, τ: float
):
    φ = TargetState(deepcopy(ψ), H, τ)
    φ_real, φ_imag, _ = _swo_to_explicit(φ, samples)

    important_iterations = list(range(0, epochs, epochs // 10))
    if epochs - 1 not in important_iterations:
        important_iterations.append(epochs - 1)

    for i in range(epochs):
        optimiser.zero_grad()
        ψ_real, ψ_imag, ψ_scale = _swo_to_explicit(ψ, samples)
        # with torch.no_grad():
        #     weights = _swo_compute_weights((φ_real, φ_imag), (ψ_real, ψ_imag))
        # loss = l2_error((φ_real, φ_imag), (ψ_real, ψ_imag), weights)
        loss = negative_log_overlap((φ_real, φ_imag), (ψ_real, ψ_imag))
        if i in important_iterations:
            logging.info("{}%: loss = {}".format(100 * (i + 1) // epochs, loss))
        loss.backward()
        optimiser.step()

    return ψ


def make_checkpoints_for(n: int):
    if n <= 10:
        return list(range(0, n))
    important_iterations = list(range(0, n, n // 10))
    if important_iterations[-1] != n - 1:
        important_iterations.append(n - 1)
    return important_iterations


TrainConfig = collections.namedtuple("TrainConfig", ["optimiser", "loss", "epochs"])


def _train_amplitude(ψ: torch.nn.Module, φ: torch.nn.Module, samples, config):
    logging.info("Learning amplitude...")
    start = time.time()

    def make_target(f, xs):
        with torch.no_grad():
            f_xs = f(xs)[:, 0]
            scale = normalisation_constant(f_xs)
            f_xs += scale
            return torch.exp(f_xs)

    with torch.no_grad():
        expected = make_target(φ, samples).view(-1)
    epochs = config.epochs
    checkpoints = make_checkpoints_for(epochs)
    # optimiser = config.optimiser(filter(lambda p: p.requires_grad, ψ.parameters()))
    optimiser = config.optimiser(ψ.parameters())
    loss_fn = config.loss

    for i in range(epochs):
        optimiser.zero_grad()
        predicted = ψ(samples).view(-1)
        scale = normalisation_constant(predicted.detach()).item()
        loss = loss_fn(torch.exp(scale + predicted), expected)
        if i in checkpoints:
            logging.info("{}%: Loss = {}".format(100 * (i + 1) // epochs, loss))
        loss.backward()
        optimiser.step()

    finish = time.time()
    logging.info("Done in {:.2f} seconds!".format(finish - start))
    return ψ


def _train_phase(ψ: torch.nn.Module, φ: torch.nn.Module, samples, config):
    logging.info("Learning phase...")
    start = time.time()

    def make_target(f, xs):
        with torch.no_grad():
            f_xs = f(xs)[:, 1]
            f_xs = torch.fmod(
                torch.round(torch.abs(torch.div(f_xs, math.pi))).long(), 2
            )
            return f_xs

    with torch.no_grad():
        expected = make_target(φ, samples).view(-1)
    epochs = config.epochs
    checkpoints = make_checkpoints_for(epochs)
    # optimiser = config.optimiser(filter(lambda p: p.requires_grad, ψ.parameters()))
    optimiser = config.optimiser(ψ.parameters())
    loss_fn = config.loss  # torch.nn.CrossEntropyLoss()

    def accuracy():
        with torch.no_grad():
            _, predicted = torch.max(ψ(samples), dim=1)
            return float(torch.sum(expected == predicted)) / expected.size(0)

    logging.info("Initial accuracy: {:.2f}%".format(100 * accuracy()))
    for i in range(epochs):
        optimiser.zero_grad()
        loss = loss_fn(ψ(samples), expected)
        if i in checkpoints:
            logging.info("{}%: Loss = {}".format(100 * (i + 1) // epochs, loss))
        loss.backward()
        optimiser.step()
    logging.info("Final accuracy: {:.2f}%".format(100 * accuracy()))

    finish = time.time()
    logging.info("Done in {:.2f} seconds!".format(finish - start))
    return ψ


SWOConfig = collections.namedtuple(
    "SWOConfig",
    [
        "H",
        "τ",
        "steps",
        "magnetisation",
        "lr_amplitude",
        "lr_phase",
        "epochs_amplitude",
        "epochs_phase",
    ],
)


def _swo_step(ψ, config):
    ψ_amplitude, ψ_phase = ψ
    φ = TargetState(
        CombiningState(deepcopy(ψ_amplitude), deepcopy(ψ_phase)), config.H, config.τ
    )
    φ = TargetState(φ, config.H, config.τ)
    φ = TargetState(φ, config.H, config.τ)
    φ = TargetState(φ, config.H, config.τ)
    # temp = sample_state(φ, H=config.H, magnetisation=0, explicit=True, requires_energy=True)
    # print(torch.mm(temp[0].weights.view(1, -1), temp[0].energies.view(-1, 2)))

    samples = all_spins(φ.number_spins, config.magnetisation)
    # sample_some(φ, config.steps, config.magnetisation)
    # all_spins(φ.number_spins, config.magnetisation)
    logging.info("Training on {} spin configurations...".format(samples.size(0)))

    ψ_amplitude = _train_amplitude(
        ψ_amplitude,
        φ,
        samples,
        TrainConfig(
            optimiser=lambda p: torch.optim.Adam(p, lr=config.lr_amplitude),
            loss=lambda predicted, expected: negative_log_overlap_real(
                expected, predicted
            ),
            epochs=config.epochs_amplitude,
        ),
    )
    ψ_phase = _train_phase(
        ψ_phase,
        φ,
        samples,
        TrainConfig(
            optimiser=lambda p: torch.optim.Adam(p, lr=config.lr_phase),
            loss=torch.nn.CrossEntropyLoss(),
            epochs=config.epochs_phase,
        ),
    )
    return ψ_amplitude, ψ_phase


@click.group()
def cli():
    pass


def import_network(nn_file):
    module_name, extension = os.path.splitext(os.path.basename(nn_file))
    module_dir = os.path.dirname(nn_file)
    if extension != ".py":
        raise ValueError(
            "Could not import the network from {}: not a python source file."
        )
    # Insert in the beginning
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    sys.path.pop(0)
    return module.Net


def load_explicit(stream):
    psi = {}
    number_spins = None
    for line in stream:
        if line.startswith(b"#"):
            continue
        (spin, real, imag) = line.split()
        if number_spins is None:
            number_spins = len(spin)
        else:
            assert number_spins == len(spin)
        spin = Spin.from_str(spin)  # int(spin, base=2)
        coeff = complex(float(real), float(imag))
        psi[spin] = coeff
    return psi, number_spins


@cli.command()
@click.argument(
    "nn-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar="<torch.nn.Module file>",
)
@click.option(
    "-i",
    "--in-file",
    type=click.File(mode="rb"),
    help="File containing the Neural Network weights as a PyTorch `state_dict` "
    "serialised using `torch.save`. It is up to the user to ensure that "
    "the weights are compatible with the architecture read from "
    "<arch_file>.",
)
@click.option(
    "-o",
    "--out-file",
    type=click.File(mode="wb"),
    help="Where to save the final state to. It will contain a "
    "PyTorch `state_dict` serialised using `torch.save`. If no file is "
    "specified, the result will be discarded.",
)
@click.option(
    "-H",
    "--hamiltonian",
    "hamiltonian_file",
    type=click.File(mode="r"),
    required=True,
    help="File containing the Heisenberg Hamiltonian specifications.",
)
@click.option(
    "--use-sr",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to use Stochastic Reconfiguration for optimisation.",
)
@click.option(
    "-n",
    "--epochs",
    type=click.IntRange(min=0),
    default=200,
    show_default=True,
    help="Number of learning steps to perform.",
)
@click.option(
    "--lr",
    type=click.FloatRange(min=1.0e-10),
    default=0.05,
    show_default=True,
    help="Learning rate.",
)
@click.option(
    "--time",
    "time_limit",
    type=click.FloatRange(min=1.0e-10),
    show_default=True,
    help="Time interval that specifies how often the model is written to the "
    "output file. If not specified, the weights are saved only once -- "
    "after all the iterations.",
)
@click.option(
    "--steps",
    type=click.IntRange(min=1),
    default=2000,
    show_default=True,
    help="Length of the Markov Chain.",
)
def optimise(
    nn_file, in_file, out_file, hamiltonian_file, use_sr, epochs, lr, steps, time_limit
):
    """
    Variational Monte Carlo optimising E.
    """
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )
    BaseNet = import_network(nn_file)
    H = read_hamiltonian(hamiltonian_file)
    machine = _Machine(BaseNet(H.number_spins))
    if in_file is not None:
        logging.info("Reading the weights...")
        machine.psi.load_state_dict(torch.load(in_file))

    magnetisation = 0 if machine.number_spins % 2 == 0 else 1
    thermalisation = int(0.1 * steps)
    opt = Optimiser(
        machine,
        H,
        magnetisation=magnetisation,
        epochs=epochs,
        monte_carlo_steps=(
            thermalisation * machine.number_spins,
            (thermalisation + steps) * machine.number_spins,
            machine.number_spins,
        ),
        learning_rate=lr,
        use_sr=use_sr,
        regulariser=lambda i: 100.0 * 0.9 ** i + 0.001,
        model_file=out_file,
        time_limit=time_limit,
    )
    opt()


@cli.command()
@click.argument(
    "amplitude-nn-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar="<amplitude torch.nn.Module>",
)
@click.argument(
    "phase-nn-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar="<phase torch.nn.Module>",
)
@click.option(
    "--in-amplitude",
    type=click.File(mode="rb"),
    help="Pickled initial `state_dict` of the neural net predicting log amplitudes.",
)
@click.option(
    "--in-phase",
    type=click.File(mode="rb"),
    help="Pickled initial `state_dict` of the neural net predicting phases.",
)
@click.option("-o", "--out-file", type=str, help="Basename for output files.")
@click.option(
    "-H",
    "--hamiltonian",
    "hamiltonian_file",
    type=click.File(mode="r"),
    required=True,
    help="File containing the Heisenberg Hamiltonian specifications.",
)
@click.option(
    "--epochs-outer",
    type=click.IntRange(min=0),
    default=200,
    show_default=True,
    help="Number of outer learning steps (i.e. Lanczos steps) to perform.",
)
@click.option(
    "--epochs-amplitude",
    type=click.IntRange(min=0),
    default=2000,
    show_default=True,
    help="Number of epochs when learning the amplitude",
)
@click.option(
    "--epochs-phase",
    type=click.IntRange(min=0),
    default=2000,
    show_default=True,
    help="Number of epochs when learning the phase",
)
@click.option(
    "--lr-amplitude",
    type=click.FloatRange(min=1.0e-10),
    default=0.05,
    show_default=True,
    help="Learning rate for training the amplitude net.",
)
@click.option(
    "--lr-phase",
    type=click.FloatRange(min=1.0e-10),
    default=0.05,
    show_default=True,
    help="Learning rate for training the phase net.",
)
@click.option(
    "--tau",
    type=click.FloatRange(min=1.0e-10),
    default=0.2,
    show_default=True,
    help="τ",
)
@click.option(
    "--steps",
    type=click.IntRange(min=1),
    default=2000,
    show_default=True,
    help="Length of the Markov Chain.",
)
def swo(
    amplitude_nn_file,
    phase_nn_file,
    in_amplitude,
    in_phase,
    out_file,
    hamiltonian_file,
    epochs_outer,
    epochs_amplitude,
    epochs_phase,
    lr_amplitude,
    lr_phase,
    tau,
    steps,
):
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )
    H = read_hamiltonian(hamiltonian_file)
    number_spins = H.number_spins
    magnetisation = 0 if number_spins % 2 == 0 else 1

    thermalisation = steps // 10
    m_c_steps = (
        thermalisation * number_spins,
        (thermalisation + steps) * number_spins,
        1,
    )

    ψ_amplitude = import_network(amplitude_nn_file)(number_spins)
    if in_amplitude is not None:
        logging.info("Reading initial state...")
        ψ_amplitude.load_state_dict(torch.load(in_amplitude))

    ψ_phase = import_network(phase_nn_file)(number_spins)
    if in_phase is not None:
        logging.info("Reading initial state...")
        ψ_phase.load_state_dict(torch.load(in_phase))

    # def save_explicit(iteration):
    #     explicit_dict = to_explicit_dict(
    #         CombiningState(ψ_amplitude, ψ_phase),
    #         explicit=True,
    #         magnetisation=magnetisation,
    #     )
    #     with open("{}.{}.explicit".format(out_file, iteration), "wb") as f:
    #         for spin, value in explicit_dict.items():
    #             f.write(
    #                 "{}\t{:.10e}\t{:.10e}\n".format(
    #                     spin, value.real, value.imag
    #                 ).encode("utf-8")
    #             )

    for i in range(epochs_outer):
        logging.info("\t#{}".format(i + 1))
        _swo_step(
            (ψ_amplitude, ψ_phase),
            SWOConfig(
                H=H,
                τ=tau,
                steps=(5,) + m_c_steps,
                magnetisation=magnetisation,
                lr_amplitude=lr_amplitude,
                lr_phase=lr_phase,
                epochs_amplitude=epochs_amplitude,
                epochs_phase=epochs_phase,
            ),
        )
        torch.save(
            ψ_amplitude.state_dict(), "{}.{}.amplitude.weights".format(out_file, i)
        )
        torch.save(ψ_phase.state_dict(), "{}.{}.phase.weights".format(out_file, i))
        # save_explicit(i)
    # save_explicit(i)


@cli.command()
@click.argument(
    "nn-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar="<arch_file>",
)
@click.option(
    "-t",
    "--train-file",
    type=click.File(mode="rb"),
    required=True,
    help="File containing the explicit wave function representation as "
    "generated by `sample`. It will be used as the training data set.",
)
@click.option(
    "-i",
    "--in-file",
    type=click.File(mode="rb"),
    help="File containing the initial Neural Network weights as a PyTorch "
    "`state_dict` serialised using `torch.save`. It is up to the user to ensure "
    "that the weights are compatible with the architecture read from <arch_file>.",
)
@click.option(
    "-o",
    "--out-file",
    type=str,
    required=True,
    help="File where the final weights will be stored. It will contain a "
    "PyTorch `state_dict` serialised using `torch.save`.",
)
@click.option(
    "--lr",
    type=click.FloatRange(min=1.0e-10),
    default=0.05,
    show_default=True,
    help="Learning rate.",
)
@click.option(
    "--batch-size", type=int, default=None, show_default=True, help="Learning rate."
)
@click.option(
    "--epochs",
    type=click.IntRange(min=0),
    default=200,
    show_default=True,
    help="Number of learning steps to perform. Here step is defined as a single "
    "update of the parameters.",
)
@click.option(
    "--optimiser",
    type=str,
    default="SGD",
    help="Optimizer to use. Valid values are names of classes in torch.optim "
    "(i.e. 'SGD', 'Adam', 'Adamax', etc.)",
)
def train(nn_file, train_file, out_file, in_file, lr, optimiser, epochs, batch_size):
    """
    Supervised learning.
    """
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.DEBUG
    )
    φ_dict, number_spins = load_explicit(train_file)
    logging.info("Generating training data set...")
    samples = torch.empty((len(φ_dict), number_spins), dtype=torch.float32)
    φ_real = torch.empty((len(φ_dict),), dtype=torch.float32)
    φ_imag = torch.empty((len(φ_dict),), dtype=torch.float32)
    for (i, (key, value)) in enumerate(φ_dict.items()):
        samples[i, :] = torch.from_numpy(key.numpy())
        φ_real[i] = value.real
        φ_imag[i] = value.imag

    ψ = import_network(nn_file)(number_spins)
    if in_file is not None:
        logging.info("Reading initial state...")
        ψ.load_state_dict(torch.load(in_file))

    # NOTE(twesterhout): This is a hack :)
    optimiser = getattr(torch.optim, optimiser)(
        filter(lambda p: p.requires_grad_, ψ.parameters()), lr=lr
    )

    def save(i):
        out_file_name = "{}.{}.weights".format(out_file, i)
        torch.save(ψ.state_dict(), out_file_name)

    important_iterations = list(range(0, epochs, epochs // 10))
    if epochs - 1 not in important_iterations:
        important_iterations.append(epochs - 1)

    if batch_size is not None:
        indices = torch.empty(samples.size(0), dtype=torch.int64)
        for i in range(epochs):
            torch.randperm(samples.size(0), out=indices)
            for batch in torch.split(indices, batch_size):
                optimiser.zero_grad()
                ψ_real, ψ_imag, _ = _swo_to_explicit(ψ, samples[batch])
                loss = negative_log_overlap(
                    (φ_real[batch], φ_imag[batch]), (ψ_real, ψ_imag)
                )
                # logging.info("{}: loss = {}".format(100 * (i + 1) // epochs, loss))
                loss.backward()
                optimiser.step()
            if i in important_iterations:
                with torch.no_grad():
                    ψ_real, ψ_imag, _ = _swo_to_explicit(ψ, samples)
                    loss = negative_log_overlap((φ_real, φ_imag), (ψ_real, ψ_imag))
                logging.info("{}%: loss = {}".format(100 * (i + 1) // epochs, loss))
                save(i)
    else:
        for i in range(epochs):
            optimiser.zero_grad()
            ψ_real, ψ_imag, _ = _swo_to_explicit(ψ, samples)
            loss = negative_log_overlap((φ_real, φ_imag), (ψ_real, ψ_imag))
            if i in important_iterations:
                logging.info("{}%: loss = {}".format(100 * (i + 1) // epochs, loss))
                save(i)
            loss.backward()
            optimiser.step()


@cli.command()
@click.argument(
    "nn-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar="<arch_file>",
)
@click.option(
    "-t",
    "--train-file",
    type=click.File(mode="rb"),
    required=True,
    help="File containing the explicit wave function representation as "
    "generated by `sample`. It will be used as the training data set.",
)
@click.option(
    "-i",
    "--in-file",
    type=click.File(mode="rb"),
    help="File containing the initial Neural Network weights as a PyTorch "
    "`state_dict` serialised using `torch.save`. It is up to the user to ensure "
    "that the weights are compatible with the architecture read from <arch_file>.",
)
@click.option(
    "-o",
    "--out-file",
    type=str,
    required=True,
    help="File where the final weights will be stored. It will contain a "
    "PyTorch `state_dict` serialised using `torch.save`.",
)
@click.option(
    "--lr",
    type=click.FloatRange(min=1.0e-10),
    default=0.05,
    show_default=True,
    help="Learning rate.",
)
@click.option(
    "--batch-size", type=int, default=None, show_default=True, help="Learning rate."
)
@click.option(
    "--epochs",
    type=click.IntRange(min=0),
    default=200,
    show_default=True,
    help="Number of learning steps to perform. Here step is defined as a single "
    "update of the parameters.",
)
@click.option(
    "--optimiser",
    type=str,
    default="SGD",
    help="Optimizer to use. Valid values are names of classes in torch.optim "
    "(i.e. 'SGD', 'Adam', 'Adamax', etc.)",
)
def train_amplitude(
    nn_file, train_file, out_file, in_file, lr, optimiser, epochs, batch_size
):
    """
    Supervised learning.
    """
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.DEBUG
    )
    φ_dict, number_spins = load_explicit(train_file)
    logging.info("Generating training data set...")
    samples = torch.empty((len(φ_dict), number_spins), dtype=torch.float32)
    φ = torch.empty((len(φ_dict),), dtype=torch.float32)
    for (i, (key, value)) in enumerate(φ_dict.items()):
        samples[i, :] = torch.from_numpy(key.numpy())
        φ[i] = value.real

    φ *= 100

    ψ = import_network(nn_file)(number_spins)
    if in_file is not None:
        logging.info("Reading initial state...")
        ψ.load_state_dict(torch.load(in_file))

    # NOTE(twesterhout): This is a hack :)
    optimiser = getattr(torch.optim, optimiser)(
        filter(lambda p: p.requires_grad_, ψ.parameters()), lr=lr
    )

    def save(i):
        out_file_name = "{}.{}.weights".format(out_file, i)
        torch.save(ψ.state_dict(), out_file_name)

    important_iterations = list(range(0, epochs, epochs // 10))
    if epochs - 1 not in important_iterations:
        important_iterations.append(epochs - 1)

    if batch_size is not None:
        indices = torch.empty(samples.size(0), dtype=torch.int64)
        for i in range(epochs):
            torch.randperm(samples.size(0), out=indices)
            for batch in torch.split(indices, batch_size):
                optimiser.zero_grad()
                # ψ_ = _apply_normalised(ψ, samples[batch])
                # loss = torch.sum(torch.log(torch.cosh(φ[batch] - ψ_)))
                # logging.info("{}: loss = {}".format(100 * (i + 1) // epochs, loss))
                x = φ[batch] - torch.exp(ψ(samples[batch])).view(-1)
                loss = torch.dot(x, x)
                loss.backward()
                optimiser.step()
            if i in important_iterations:
                with torch.no_grad():
                    # ψ_ = _apply_normalised(ψ, samples)
                    # loss = torch.sum(torch.log(torch.cosh(φ - ψ_)))
                    x = φ - torch.exp(ψ(samples)).view(-1)
                    loss = torch.dot(x, x)
                logging.info("{}%: loss = {}".format(100 * (i + 1) // epochs, loss))
                save(i)
    else:
        for i in range(epochs):
            optimiser.zero_grad()
            # x = φ - torch.exp(ψ(samples)).view(-1)
            # loss = torch.dot(x, x)
            loss = negative_log_overlap_real(φ, torch.exp(ψ(samples).view(-1)))
            if i in important_iterations:
                logging.info("{}%: loss = {}".format(100 * (i + 1) // epochs, loss))
                save(i)
            loss.backward()
            optimiser.step()


@cli.command()
@click.argument(
    "nn-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar="<arch_file>",
)
@click.option(
    "-t",
    "--train-file",
    type=click.File(mode="rb"),
    required=True,
    help="File containing the explicit wave function representation as "
    "generated by `sample`. It will be used as the training data set.",
)
@click.option(
    "-i",
    "--in-file",
    type=click.File(mode="rb"),
    required=True,
    help="File containing the initial Neural Network weights as a PyTorch "
    "`state_dict` serialised using `torch.save`. It is up to the user to ensure "
    "that the weights are compatible with the architecture read from <arch_file>.",
)
@click.option(
    "-o",
    "--out-file",
    type=click.File(mode="wb"),
    required=True,
    help="File where the final weights will be stored. It will contain a "
    "PyTorch `state_dict` serialised using `torch.save`.",
)
def analyse_amplitude(nn_file, train_file, out_file, in_file):
    """
    Supervised learning.
    """
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.DEBUG
    )
    φ_dict, number_spins = load_explicit(train_file)
    logging.info("Generating training data set...")
    samples = torch.empty((len(φ_dict), number_spins), dtype=torch.float32)
    φ_phase = torch.empty((len(φ_dict),), dtype=torch.float32)
    for (i, (key, value)) in enumerate(φ_dict.items()):
        samples[i, :] = torch.from_numpy(key.numpy())
        φ_phase[i] = math.atan2(value.imag, value.real)

    ψ = import_network(nn_file)(number_spins)
    logging.info("Reading initial state...")
    ψ.load_state_dict(torch.load(in_file))

    ψ_abs = torch.exp(ψ(samples)).view(-1)

    for i in range(samples.size(0)):
        out_file.write(
            "{}\t{:.10e}\t{:.10e}\n".format(
                Spin.from_array(samples[i].numpy()),
                ψ_abs[i] * math.cos(φ_phase[i]),
                ψ_abs[i] * math.sin(φ_phase[i]),
            ).encode("utf-8")
        )


@cli.command()
@click.argument(
    "nn-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar="<arch_file>",
)
@click.option(
    "-t",
    "--train-file",
    type=click.File(mode="rb"),
    required=True,
    help="File containing the explicit wave function representation as "
    "generated by `sample`. It will be used as the training data set.",
)
@click.option(
    "-i",
    "--in-file",
    type=click.File(mode="rb"),
    help="File containing the initial Neural Network weights as a PyTorch "
    "`state_dict` serialised using `torch.save`. It is up to the user to ensure "
    "that the weights are compatible with the architecture read from <arch_file>.",
)
@click.option(
    "-o",
    "--out-file",
    type=str,
    required=True,
    help="File where the final weights will be stored. It will contain a "
    "PyTorch `state_dict` serialised using `torch.save`.",
)
@click.option(
    "--lr",
    type=click.FloatRange(min=1.0e-10),
    default=0.05,
    show_default=True,
    help="Learning rate.",
)
@click.option(
    "--batch-size", type=int, default=None, show_default=True, help="Learning rate."
)
@click.option(
    "--epochs",
    type=click.IntRange(min=0),
    default=200,
    show_default=True,
    help="Number of learning steps to perform. Here step is defined as a single "
    "update of the parameters.",
)
@click.option(
    "--optimiser",
    type=str,
    default="SGD",
    help="Optimizer to use. Valid values are names of classes in torch.optim "
    "(i.e. 'SGD', 'Adam', 'Adamax', etc.)",
)
def train_phase(
    nn_file, train_file, out_file, in_file, lr, optimiser, epochs, batch_size
):
    """
    Supervised learning.
    """
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )

    φ_dict, number_spins = load_explicit(train_file)
    # φ_dict_log = dict(map(lambda x: (x[0], cmath.log(x[1])), φ_dict.items()))
    φ = ExplicitState(φ_dict, apply_log=True)

    logging.info("Generating training data set...")
    do_monte_carlo = True
    magnetisation = 0 if number_spins % 2 == 0 else 1
    if do_monte_carlo:
        number_chains = 3
        steps = 2000
        thermalisation = steps // 10
        monte_carlo_steps = (
            thermalisation * number_spins,
            (thermalisation + steps) * number_spins,
            1,
        )
        samples = sample_some(φ, (number_chains,) + monte_carlo_steps, magnetisation)
        logging.info("#samples: {}".format(samples.size(0)))
    else:
        samples = all_spins(number_spins, magnetisation)

    ψ = import_network(nn_file)(number_spins)
    if in_file is not None:
        logging.info("Reading initial state...")
        ψ.load_state_dict(torch.load(in_file))
    else:
        logging.info("Starting with random parameters...")

    # _train_phase(ψ, φ, samples, epochs, lr)

    # ψ = import_network("small.py")(number_spins)
    # logging.info("Starting with random parameters...")

    _train_amplitude(ψ, φ, samples, epochs, lr)

    ψ_phase = import_network("phase-1.py")(number_spins)
    ψ_phase.load_state_dict(torch.load("Kagome-18.phase.1.weights"))

    φ_xs = φ(samples)
    φ_scale = normalisation_constant(φ_xs[:, 0])
    φ_real = torch.empty(samples.size(0), dtype=torch.float32)
    φ_imag = torch.empty(samples.size(0), dtype=torch.float32)
    for i in range(samples.size(0)):
        t = cmath.exp(complex(φ_xs[i, 0] + φ_scale, φ_xs[i, 1]))
        φ_real[i] = t.real
        φ_imag[i] = t.imag

    ψ_combo = CombiningState(ψ, ψ_phase)
    ψ_xs = ψ_combo(samples)
    ψ_scale = normalisation_constant(ψ_xs[:, 0])
    ψ_real = torch.empty(samples.size(0), dtype=torch.float32)
    ψ_imag = torch.empty(samples.size(0), dtype=torch.float32)
    for i in range(samples.size(0)):
        t = cmath.exp(complex(ψ_xs[i, 0] + ψ_scale, ψ_xs[i, 1]))
        ψ_real[i] = t.real
        ψ_imag[i] = t.imag

    logging.info(
        "Negative log overlap: {}".format(
            negative_log_overlap((ψ_real, ψ_imag), (φ_real, φ_imag))
        )
    )

    with open("hello_world.dat", "wb") as f:
        for i in range(samples.size(0)):
            f.write(
                "{}\t{:.10e}\t{:.10e}\n".format(
                    Spin.from_array(samples[i].numpy()), ψ_real[i], ψ_imag[i]
                ).encode("utf-8")
            )
    return

    target = (lambda y: torch.atan2(y[:, 1], y[:, 0]))(φ(samples))
    target -= (torch.max(target) + torch.min(target)) / 2
    target = ((target.sign() + 1) // 2).long()
    print(target)

    # NOTE(twesterhout): This is a hack :)
    optimiser = getattr(torch.optim, optimiser)(
        filter(lambda p: p.requires_grad_, ψ.parameters()), lr=lr
    )

    def accuracy():
        with torch.no_grad():
            _, prediction = torch.max(ψ(samples), dim=1)
        return float(torch.sum(target == prediction)) / target.size(0)

    def save(i):
        out_file_name = "{}.{}.weights".format(out_file, i)
        torch.save(ψ.state_dict(), out_file_name)

    if epochs < 10:
        important_iterations = list(range(epochs))
    else:
        important_iterations = list(range(0, epochs, epochs // 10))
        if epochs - 1 not in important_iterations:
            important_iterations.append(epochs - 1)

    loss_fn = torch.nn.CrossEntropyLoss()

    logging.info("Initial accuracy: {:.2f}%".format(100 * accuracy()))

    if batch_size is not None:
        raise NotImplementedError()
        indices = torch.empty(samples.size(0), dtype=torch.int64)
        for i in range(epochs):
            torch.randperm(samples.size(0), out=indices)
            for batch in torch.split(indices, batch_size):
                optimiser.zero_grad()
                # ψ_ = _apply_normalised(ψ, samples[batch])
                # loss = torch.sum(torch.log(torch.cosh(φ[batch] - ψ_)))
                # logging.info("{}: loss = {}".format(100 * (i + 1) // epochs, loss))
                x = φ[batch] - torch.exp(ψ(samples[batch])).view(-1)
                loss = torch.dot(x, x)
                loss.backward()
                optimiser.step()
            if i in important_iterations:
                with torch.no_grad():
                    # ψ_ = _apply_normalised(ψ, samples)
                    # loss = torch.sum(torch.log(torch.cosh(φ - ψ_)))
                    x = φ - torch.exp(ψ(samples)).view(-1)
                    loss = torch.dot(x, x)
                logging.info("{}%: loss = {}".format(100 * (i + 1) // epochs, loss))
                save(i)
    else:
        for i in range(epochs):
            optimiser.zero_grad()
            # x = φ - torch.exp(ψ(samples)).view(-1)
            # loss = torch.dot(x, x)

            loss = loss_fn(ψ(samples), target)
            if i in important_iterations:
                logging.info("{}%: loss = {}".format(100 * (i + 1) // epochs, loss))
                save(i)
            loss.backward()
            optimiser.step()

    logging.info("Final accuracy: {:.2f}".format(100 * accuracy()))


@cli.command()
@click.argument(
    "amplitude-nn-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar="<amplitude torch.nn.Module>",
)
@click.argument(
    "phase-nn-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar="<phase torch.nn.Module>",
)
@click.option(
    "--in-amplitude",
    type=click.File(mode="rb"),
    help="File containing the Neural Network weights as a PyTorch `state_dict` "
    "serialised using `torch.save`. It is up to the user to ensure that "
    "the weights are compatible with the architecture read from "
    "<arch_file>.",
)
@click.option(
    "--in-phase",
    type=click.File(mode="rb"),
    help="File containing the Neural Network weights as a PyTorch `state_dict` "
    "serialised using `torch.save`. It is up to the user to ensure that "
    "the weights are compatible with the architecture read from "
    "<arch_file>.",
)
@click.option(
    "-o",
    "--out-file",
    type=str,
    help="Where to save the final state to. It will contain a "
    "PyTorch `state_dict` serialised using `torch.save`. If no file is "
    "specified, the result will be discarded.",
)
@click.option(
    "-H",
    "--hamiltonian",
    "hamiltonian_file",
    type=click.File(mode="r"),
    required=True,
    help="File containing the Heisenberg Hamiltonian specifications.",
)
@click.option(
    "--steps",
    type=click.IntRange(min=1),
    default=2000,
    show_default=True,
    help="Length of the Markov Chain.",
)
def energy(
    amplitude_nn_file,
    phase_nn_file,
    in_amplitude,
    in_phase,
    out_file,
    hamiltonian_file,
    steps,
):
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )
    H = read_hamiltonian(hamiltonian_file)
    number_spins = H.number_spins
    magnetisation = 0 if number_spins % 2 == 0 else 1

    thermalisation = steps // 10
    m_c_steps = (
        thermalisation * number_spins,
        (thermalisation + steps) * number_spins,
        number_spins,
    )

    ψ_amplitude = import_network(amplitude_nn_file)(number_spins)
    logging.info("Reading initial state...")
    ψ_amplitude.load_state_dict(torch.load(in_amplitude))

    ψ_phase = import_network(phase_nn_file)(number_spins)
    logging.info("Reading initial state...")
    ψ_phase.load_state_dict(torch.load(in_phase))

    ψ = CombiningState(ψ_amplitude, ψ_phase)
    machine = Machine(ψ)

    samples = sample_one(machine, m_c_steps, unique=False)

    E = np.empty((samples.size(0),), dtype=np.complex64)
    for i in range(samples.size(0)):
        E[i] = H(MonteCarloState(machine=machine, spin=samples[i].numpy(), weight=None))

    logging.info("E = {} ± {}".format(np.mean(E), np.std(E)))


if __name__ == "__main__":
    cli()
    # cProfile.run('main()')
