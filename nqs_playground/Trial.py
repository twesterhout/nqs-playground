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
import copy
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

# TODO(twesterhout): This function is left as a backup only is should be removed.
# @jit(uint8[:](float32[:]), nopython=True)
# def to_bytes(spin: np.ndarray) -> np.ndarray:
#     """
#     Converts a spin to a bit array. It is assumed that a spin-up corresponds to
#     1.0.
#
#     .. warning::
#
#        This function is deprecated and will be removed soon.
#     """
#     chunks, rest = divmod(spin.size, 8)
#     b = np.empty(chunks + int(rest > 0), dtype=np.uint8)
#     if rest != 0:
#         b[0] = spin[0] == 1.0
#         for i in range(1, rest):
#             b[0] = (b[0] << 1) | (spin[i] == 1.0)
#     for i in range(chunks):
#         j = 8 * i + rest
#         b[int(rest > 0) + i] = (
#             ((spin[j + 0] == 1.0) << 7)
#             | ((spin[j + 1] == 1.0) << 6)
#             | ((spin[j + 2] == 1.0) << 5)
#             | ((spin[j + 3] == 1.0) << 4)
#             | ((spin[j + 4] == 1.0) << 3)
#             | ((spin[j + 5] == 1.0) << 2)
#             | ((spin[j + 6] == 1.0) << 1)
#             | ((spin[j + 7] == 1.0) << 0)
#         )
#     return b


# TODO(twesterhout): This function is left as a backup only and should be remvoed.
# @jit(float32[:](Bytes(uint8, 1, "C"), int64), nopython=True)
# def from_bytes(b: bytes, n: int) -> np.ndarray:
#     """
#     .. warning::
#
#        This function is deprecated and will be removed soon.
#     """
#     chunks, rest = divmod(n, 8)
#     spin = np.empty(n, dtype=np.float32)
#     get = lambda _b, i: -1.0 + 2.0 * int((_b >> (7 - i)) & 0x01)
#     if rest != 0:
#         offset = 8 - rest
#         for i in range(rest):
#             spin[i] = get(b[0], offset + i)
#     for i in range(chunks):
#         for j in range(8):
#             spin[rest + 8 * i + j] = get(b[int(rest != 0) + i], j)
#     return spin


@jit(uint64(uint64), nopython=True)
def _hash_uint64(x: int) -> int:
    """
    Hashes an integer.

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
    Compact representation of a spin configuration.

    Spin quantum numbers are represented by bits:

      * unset (``0``) means *spin down* and
      * set (``1``) means *spin up*.

    Currently, maximum supported size is 192.
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
        Returns integer representation of the spin configuration.

        .. warning::

           Only spin configurations shorted than 64 support this function.
        """
        if self._size > 63:
            raise OverflowError(
                "Spin configuration is too long to be represented by an int"
            )
        return self._d2

    def equal_to(self, other) -> bool:
        """
        :return: ``self == other``
        """
        assert self._size == other._size
        return self._d0 == other._d0 and self._d1 == other._d1 and self._d2 == other._d2

    def less_than(self, other) -> bool:
        """
        :return: ``self < other``
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
        :return: ``self <= other``
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

    def as_str(self) -> np.ndarray:
        """
        Returns the string representation of the spin configuration.

        .. warning::

           This function is quite slow. Do not use it in performance-critical
           parts.
        """
        code_of_zero = 48
        code_of_one = 49
        s = np.empty(self._size, dtype=np.uint8)
        for i in range(self._size):
            s[i] = code_of_one if self.get(i) else code_of_zero
        return s


@jit(numba.types.uint32(numba.types.uint32), nopython=True)
def _popcount_32(v: int) -> int:
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    return (((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) & 0xFFFFFFFF) >> 24


@jit(uint64(uint64), nopython=True)
def _popcount(x: int) -> int:
    return _popcount_32(x & uint64(0x00000000FFFFFFFF)) + _popcount_32(x >> 32)


@jit(
    locals={
        "chunks": uint64,
        "rest": uint64,
        "d0": uint64,
        "d1": uint64,
        "d2": uint64,
        "i": uint64,
    }
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


@jit(locals={"size": uint64, "chunks": uint64, "rest": uint64, "t": uint64, "i": int64})
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
           Instead, have a look at ``from_str`` and ``from_array`` functions.
        """
        assert isinstance(x, _CompactSpin)
        self._jitted = x

    def __str__(self) -> str:
        """
        Returns the string representation of the spin configuration.

        .. warning::

           This function is quite slow, so don't use it in performance-critical
           parts.
        """
        return bytes(self._jitted.as_str().data).decode("utf-8")

    def __int__(self) -> int:
        """
        Returns the integer representation of the spin configuration.
        """
        return self._jitted.as_int()

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


def shuffle(spin: Spin, indices: np.ndarray):
    return Spin.from_array(
        np.fromiter(
            (spin._jitted.get(i) for i in indices), dtype=np.float32, count=spin.size
        )
    )


# CompactSpin = Spin.from_array
# class CompactSpin(bytes):
#     """
#     Compact representation of a spin.
#     """
#
#     def __new__(cls, spin: np.ndarray):
#         """
#         Creates a new ``CompactSpin`` given the spin (ℝⁿ).
#         """
#         return bytes.__new__(cls, to_bytes(spin).tobytes())
#
#     def __int__(self):
#         """
#         Returns an int representation of the spin.
#         """
#         return int.from_bytes(self, byteorder="big")


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

            def __init__(
                self, wave_function: complex, gradient: Optional[np.ndarray] = None
            ):
                self.log_wf = wave_function
                self.der_log_wf = gradient

        def __init__(self, *args, **kwargs):
            """
            Initialises the state with random values for the variational parameters.
            """
            super().__init__(*args, **kwargs)
            try:
                self._size = sum(
                    map(
                        lambda p: reduce(int.__mul__, p.size()),
                        filter(lambda p: p.requires_grad, self.parameters()),
                    )
                )
            except AttributeError:
                logging.warning(
                    "Could not determine the number of parameters in the Machine."
                )
                self._size = None
            # Hash-table mapping CompactSpin to Machine.Cell
            self._cache = {}
            # TODO(twesterhout): Move this to BaseNet
            # Automorphisms as a 2-dimensional array of indices
            self._automorphisms = None

        def _find_smallest(self, spin: Spin) -> Spin:
            """
            Given a compact spin configuration, returns the canonical representation of it.
            """
            if self._automorphisms is None:
                return spin

            # TODO(twesterhout): Use numba to optimise this as it really slows
            # down Monte Carlo loops.
            smallest = spin
            for permutation in self._automorphisms:
                brother = shuffle(spin, permutation)
                if brother < smallest:
                    smallest = brother
            return smallest

        def _log_wf(self, spin: np.ndarray, compact_spin: Spin) -> complex:
            cell = self._cache.get(compact_spin)
            if cell is not None:
                return cell.log_wf

            canonical_compact_spin = self._find_smallest(compact_spin)
            if canonical_compact_spin == compact_spin:
                (amplitude, phase) = self.forward(torch.from_numpy(spin))
            else:
                if False:
                    # Relatice phase is always 0
                    z = self._log_wf(
                        canonical_compact_spin.numpy(), canonical_compact_spin
                    )
                    amplitude = z.real
                    phase = z.imag
                else:
                    # Support for arbitrary phase
                    phase = self.forward(torch.from_numpy(spin))[1].item()
                    amplitude = self._log_wf(
                        canonical_compact_spin.numpy(), canonical_compact_spin
                    ).real
            log_wf = complex(amplitude, phase)
            self._cache[compact_spin] = Machine.Cell(log_wf)
            return log_wf

        def log_wf(self, spin: np.ndarray) -> complex:
            """
            Computes log(Ψ(x)).

            :param np.ndarray x: Spin configuration. Must be a numpy array of
                                 ``float32``.
            :return: log(Ψ(x))
            :rtype: complex
            """
            return self._log_wf(spin, Spin.from_array(spin))

        @property
        def size(self) -> int:
            """
            Returns the number of variational parameters.
            """
            return self._size

        def _copy_grad_to(self, out):
            assert out.shape == (self.size,)
            i = 0
            for p in map(
                lambda p_: p_.grad.view(-1).numpy(),
                filter(lambda p_: p_.requires_grad, self.parameters()),
            ):
                out[i : i + p.size] = p
                i += p.size

        def _der_log_wf(self, spin, compact_spin, out=None):
            if out is None:
                out = np.empty((self.size,), dtype=np.complex64)
            cell = self._cache.get(compact_spin)
            if cell is None:
                _ = self._log_wf(spin, compact_spin)
            if cell.der_log_wf is not None:
                out[:] = cell.der_log_wf
                return out

            canonical_compact_spin = self._find_smallest(compact_spin)
            if canonical_compact_spin == compact_spin:
                # Forward-propagation to construct the graph
                result = self.forward(torch.from_numpy(spin))
                (amplitude, phase) = result

                # Computes ∇Re[log(Ψ(x))]
                self.zero_grad()
                result.backward(
                    torch.tensor([1, 0], dtype=torch.float32), retain_graph=True
                )
                self._copy_grad_to(out.real)

                # Computes ∇Im[log(Ψ(x))]
                self.zero_grad()
                result.backward(torch.tensor([0, 1], dtype=torch.float32))
                self._copy_grad_to(out.imag)
            else:
                canonical_spin = canonical_compact_spin.numpy()
                if False:
                    # Relative phase always 0
                    out[:] = self._der_log_wf(canonical_spin, canonical_compact_spin)
                else:
                    # Support for arbitrary phase

                    # Computes ∇Re[log(Ψ(x))]
                    out[:].real = self._der_log_wf(
                        canonical_spin, canonical_compact_spin
                    ).real

                    # Forward-propagation to construct the graph
                    result = self.forward(torch.from_numpy(spin))

                    # Computes ∇Im[log(Ψ(x))]
                    self.zero_grad()
                    result.backward(torch.tensor([0, 1], dtype=torch.float32))
                    self._copy_grad_to(out.imag)

            # Save the results
            # TODO(twesterhout): Remove the copy when it's safe to do so.
            cell.der_log_wf = np.copy(out)
            # assert np.isclose(complex(amplitude, phase), cell.log_wf)
            return out

        def der_log_wf(self, spin: np.ndarray, out: np.ndarray = None) -> np.ndarray:
            """
            Computes ∇log(Ψ(x)).

            :param np.ndarray x:   Spin configuration. Must be a numpy array of ``float32``.
            :param np.ndarray out: Destination array. Must be a numpy array of ``complex64``.
            :param key: Precomputed ``CompactSpin``-representation of x.
            :type key: CompactSpin or None.
            :return: ∇log(Ψ(x)) as a numpy array of ``complex64``.
                     __Don't you dare modify it!__.
            """
            return self._der_log_wf(spin, Spin.from_array(spin), out)

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
                for dp in map(
                    lambda p_: p_.grad.data.view(-1),
                    filter(lambda p_: p_.requires_grad, self.parameters()),
                ):
                    (n,) = dp.size()
                    dp.copy_(gradients[i : i + n])
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
                    p.add_(-1, delta[i : i + n])
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


def all_spins(n: int, magnetisation: Optional[bool] = None):
    if n >= 64:
        raise OverflowError(
            "Brute-force iteration is not supported for such long spin chains."
        )
    if magnetisation is not None:

        def generate():
            for i in range(2 ** n):
                m = 2 * _popcount(i) - n
                if m == magnetisation:
                    yield Spin(_CompactSpin(0, 0, uint64(i), n))

        return generate()
    else:

        def generate():
            for i in range(2 ** n):
                yield Spin(_CompactSpin(0, 0, uint64(i), n))

        return generate()


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
                if min(
                    1.0, math.exp(self._state.log_quot_wf(flips).real) ** 2
                ) > np.random.uniform(0, 1):
                    self._accepted += 1
                    self._state.update(flips)
                    self._flipper.next(True)
                else:
                    self._flipper.next(False)

        return do_generate()

    @property
    def steps(self):
        return self._steps

    @property
    def accepted(self):
        return self._accepted


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
            ValueError(
                "Invalid graph: Counting from 0, but the minimal index "
                "present is {}.".format(smallest)
            )
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


def _load_hamiltonian(in_file):
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


def read_hamiltonian(in_file):
    return _load_hamiltonian(in_file)


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
        spin = Spin.from_array(state.spin)
        e_loc = energies_cache.get(spin)
        if e_loc is None:
            e_loc = hamiltonian(state)
            energies_cache[spin] = e_loc
        energies.append(e_loc)
    derivatives = np.array(derivatives, dtype=np.complex64)
    energies = np.array(energies, dtype=np.complex64)
    mean_O = np.mean(derivatives, axis=0)
    mean_E = np.mean(energies)
    std_E = np.std(energies)
    force = np.mean(energies * derivatives.conj().transpose(), axis=1)
    force -= mean_O.conj() * mean_E
    logging.info("Subspace dimension: {}".format(len(energies_cache)))
    logging.info(
        "Acceptance rate: {:.2f}%".format(chain._accepted / chain._steps * 100)
    )
    return derivatives, mean_O, mean_E, std_E ** 2, force


def monte_carlo_loop_for_lanczos(machine, hamiltonian, initial_spin, steps):
    logging.info("Running Monte Carlo...")
    energies = []
    energies_cache = {}
    wave_function = {}
    chain = MetropolisMC(machine, initial_spin)
    for state in islice(chain, *steps):
        spin = Spin.from_array(state.spin)
        e_loc = energies_cache.get(spin)
        if e_loc is None:
            e_loc = hamiltonian(state)
            energies_cache[spin] = e_loc
        energies.append(e_loc)
        wave_function[spin] = cmath.exp(state.log_wf())
        for s in hamiltonian.reachable_from(state.spin):
            wave_function[Spin.from_array(s)] = cmath.exp(state.machine.log_wf(s))
    energies = np.array(energies, dtype=np.complex64)
    mean_E = np.mean(energies)
    std_E = np.std(energies)
    logging.info("Subspace dimension: {}".format(len(wave_function)))
    logging.info(
        "Acceptance rate: {:.2f}%".format(chain._accepted / chain._steps * 100)
    )
    logging.info("E = {}, Var[E] = {}".format(mean_E, std_E ** 2))
    return mean_E, std_E ** 2, wave_function


def compute_l2_norm(machine, initial_spin, steps):
    _old_dps = mpmath.mp.dps
    mpmath.mp.dps = 50
    wave_function = {}
    chain = MetropolisMC(machine, initial_spin)
    for state in islice(chain, *steps):
        spin = Spin.from_array(state.spin)
        wave_function[spin] = mpmath.exp(mpmath.mpc(state.log_wf()))
    l2_norm = mpmath.sqrt(
        sum(map(lambda x: mpmath.fabs(x) ** 2, wave_function.values()), mpmath.mpf(0))
        / len(wave_function)
    )
    mpmath.mp.dps = _old_dps
    return float(l2_norm)


def monte_carlo(machine, hamiltonian, initial_spin, steps):
    logging.info("Running Monte-Carlo...")
    start = time.time()
    restarts = 5
    spin = np.copy(initial_spin)
    answer = None
    while answer is None:
        try:
            answer = monte_carlo_loop(machine, hamiltonian, spin, steps)
        except WorthlessConfiguration as err:
            if restarts > 0:
                logging.warning("Restarting the Monte-Carlo simulation...")
                restarts -= 1
                spin[err.suggestion] *= -1
            else:
                raise
    finish = time.time()
    logging.info("Done in {:.2f} seconds!".format(finish - start))
    return answer


#
# NOTE(twesterhout): This class is a work in progress: please, don't use it (yet).
#
class DenseCovariance(LinearOperator):
    """
    Sparse representation of the covariance matrix matrix S in Stochastic
    Reconfiguration method [1].
    """

    def __init__(self, gradients, mean_gradient, regulariser):
        """
        """
        (steps, n) = gradients.shape
        super().__init__(np.float32, (2 * n, n))
        gradients = gradients - mean_gradient
        S = 1 / steps * np.matmul(gradients.transpose().conj(), gradients)
        S[:n, :] += regulariser * np.eye(n, dtype=np.float32)
        self._matrix = np.empty((2 * n, n), dtype=np.float32)
        for i in range(n):
            self._matrix[i, :] = S[i].real
            self._matrix[n + i, :] = S[i].imag

    def _matvec(self, x):
        assert x.dtype == self.dtype
        return np.matmul(self._matrix, x)

    def _rmatvec(self, y):
        assert y.dtype == self.dtype
        return np.matmul(self._matrix.transpose(), y)

    def solve(self, b, x0=None):
        assert b.dtype == np.complex64
        start = time.time()
        logging.info("Calculating S⁻¹F...")
        (_, n) = self.shape
        b_ = np.empty((2 * n,), dtype=np.float32)
        b_[:n] = b.real
        b_[n:] = b.imag
        if x0 is None:
            x0 = np.zeros((n,), dtype=np.float32)
        else:
            # x0_norm = np.linalg.norm(x0)
            # x0 *= 1 / x0_norm
            pass
        x = scipy.linalg.lstsq(self._matrix, b_)[0]
        # x, istop, itn, r1norm, *_ = scipy.sparse.linalg.lsqr(self, b_, x0=x0)
        finish = time.time()
        # if istop == 1:
        #     logging.info("Done in {:.2f} seconds! Solved Sx = F in {} "
        #                  "iterations, |Sx - F| = {}".format(finish - start, itn, r1norm))
        # else:
        #     logging.info("Done in {:.2f} seconds! Solved min||Sx - F|| in {} "
        #                  "iterations, |Sx - F| = {}".format(finish - start, itn, r1norm))
        assert x.dtype == self.dtype
        return x


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
            self._S(self._S(np.ascontiguousarray(x, dtype=np.complex64))).real
            + self._lambda * x,
            dtype=np.float32,
        )

    def solve(self, b, x0=None):
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
        logging.info("Calculating S⁻¹F...")
        b_ = np.ascontiguousarray(self._S(b).real)
        x, info = scipy.sparse.linalg.lgmres(self, b_, x0)
        finish = time.time()
        if info == 0:
            logging.info("Done in {:.2f} seconds!".format(finish - start))
            return x
        if info > 0:
            logging.error("Failed to converge")
            return 0.1 * b_
        raise ValueError("The hell has just happened?")


def random_spin(n, magnetisation=None):
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
                self._machine.parameters(), lr=self._learning_rate
            )
        else:
            self._optimizer = torch.optim.Adam(
                self._machine.parameters(), lr=self._learning_rate
            )

    def learning_cycle(self, iteration):
        logging.info("==================== {} ====================".format(iteration))
        # Monte Carlo
        spin = random_spin(self._machine.number_spins, self._magnetisation)
        (Os, mean_O, E, var_E, F) = monte_carlo(
            self._machine, self._hamiltonian, spin, self._monte_carlo_steps
        )
        logging.info("E = {}, Var[E] = {}".format(E, var_E))
        # Calculate the "true" gradients
        if self._use_sr:
            # We also cache δ to use it as a guess the next time we're computing
            # S⁻¹F.
            self._delta = DenseCovariance(
                Os, mean_O, self._regulariser(iteration)
            ).solve(F, x0=self._delta)
            delta_norm = np.linalg.norm(self._delta)
            # if delta_norm > 10:
            #     self._delta /= (delta_norm / 10)
            self._machine.set_gradients(self._delta)
            logging.info("∥F∥₂ = {}, ∥δ∥₂ = {}".format(np.linalg.norm(F), delta_norm))
        else:
            self._machine.set_gradients(F.real)
            logging.info(
                "∥F∥₂ = {}, ∥Re[F]∥₂ = {}".format(
                    np.linalg.norm(F), np.linalg.norm(F.real)
                )
            )
        # Update the variational parameters
        self._optimizer.step()
        self._machine.clear_cache()

    def __call__(self):
        if self._model_file is not None:

            def save():
                # NOTE: This is important, because we want to overwrite the
                # previous weights
                self._model_file.seek(0)
                self._model_file.truncate()
                torch.save(self._machine.state_dict(), self._model_file)

        else:
            save = lambda: None
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


def _make_normalised_machine(BaseNet):
    class NormalisedMachine(_make_machine(BaseNet)):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._scale = complex(0.0, 0.0)

        @property
        def scale(self):
            return math.exp(self._scale.real)

        @scale.setter
        def scale(self, value):
            self._scale = complex(math.log(value), self._scale.imag)

        @property
        def phase(self):
            return self._scale.imag

        @phase.setter
        def phase(self, value):
            self._scale = complex(self._scale.real, value)

        def log_wf(self, x):
            return super().log_wf(x) + self._scale

        def forward(self, x):
            scale = torch.tensor(
                [self._scale.real, self._scale.imag],
                dtype=torch.float32,
                requires_grad=False,
            )
            return super().forward(x) + scale

        def backward(self, x):
            raise NotImplementedError("NormalisedMachine is not trainable")

        def der_log_wf(self, x):
            raise NotImplementedError("NormalisedMachine is not trainable")

        def set_gradients(self, x):
            raise NotImplementedError("NormalisedMachine is not trainable")

        def __isub__(self, x):
            raise NotImplementedError("NormalisedMachine is immutable")

    return NormalisedMachine


class AverageNet(nn.Module):
    def __init__(self, BaseNet, n, weight_files):
        self._machines = []
        self._number_spins = n
        Machine = _make_normalised_machine(BaseNet)
        for file_name in weight_files:
            with open(file_name, "rb") as in_file:
                psi = Machine(n)
                psi.load_state_dict(torch.load(in_file))
                self._machines.append(psi)

    @property
    def number_spins(self):
        return self._number_spins

    def normalise_(self, steps, magnetisation=None):
        for psi in self._machines:
            n_runs = 10
            l2_norms = np.array(
                [
                    compute_l2_norm(
                        psi, random_spin(psi.number_spins, magnetisation), steps
                    )
                    for _ in range(n_runs)
                ]
            )
            l2_mean = np.mean(l2_norms)
            logging.info(
                "After {} runs: ||ψ||₂ = {} ± {}".format(
                    n_runs, l2_mean, np.std(l2_norms)
                )
            )
            psi.scale = 1.0 / l2_mean
            psi.clear_cache()

    def align_(self, spin):
        for psi in self._machines:
            psi.phase = -psi.log_wf(spin).imag

    def forward(self, x):
        with torch.no_grad():

            def wf(psi, x):
                (a, b) = psi.forward(x)
                return cmath.exp(complex(a, b))

            avg_psi = np.mean(np.array([wf(psi, x) for psi in self._machines]))
            avg_log_psi = cmath.log(avg_psi)
            x = torch.tensor(
                [avg_log_psi.real, avg_log_psi.imag],
                dtype=torch.float32,
                requires_grad=False,
            )
            return x


def heisenberg6():
    hamiltonian = Heisenberg([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
    machine = Machine(6)
    return hamiltonian


def kagome12():
    # Generated using tipsi, Kagome(2, 2) with periodic boundary conditions.
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (0, 8),
        (1, 2),
        (1, 3),
        (1, 11),
        (2, 6),
        (2, 10),
        (3, 4),
        (3, 5),
        (3, 11),
        (4, 5),
        (4, 8),
        (5, 7),
        (5, 9),
        (6, 7),
        (6, 8),
        (6, 10),
        (7, 8),
        (7, 9),
        (9, 10),
        (9, 11),
        (10, 11),
    ]
    hamiltonian = Heisenberg(edges)
    return hamiltonian


def heisenberg3x3():
    hamiltonian = Heisenberg(
        [
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (4, 5),
            (5, 3),
            (6, 7),
            (7, 8),
            (8, 6),
            (0, 3),
            (3, 6),
            (6, 0),
            (1, 4),
            (4, 7),
            (7, 1),
            (2, 5),
            (5, 8),
            (8, 2),
        ]
    )
    return hamiltonian


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


def int_to_spin(spin: int, n: int) -> np.ndarray:
    return from_bytes(spin.to_bytes((n + 7) // 8, "big"), n)


def negative_log_overlap(
    machine: torch.nn.Module, target_wf: Dict[Spin, complex], offset=None
):
    """
    Calculates ``-log(|〈Ψ|Φ〉| / ‖Ψ‖₂)``, where Ψ is ``machine``
    and Φ is ``target_wf``. Φ is assumed to be normalised.
    """
    if offset is None:
        offset = 5.0
    new_offset = 0.0
    real_part = 0.0
    imag_part = 0.0
    l2_norm = 0.0
    for spin, coeff in target_wf.items():
        z = machine.forward(torch.from_numpy(spin.numpy()))
        temp = torch.exp(z[0] - offset)
        wf_real = temp * torch.cos(z[1])
        wf_imag = temp * torch.sin(z[1])
        real_part += wf_real * coeff.real - wf_imag * coeff.imag
        imag_part += wf_real * coeff.imag + wf_imag * coeff.real
        l2_norm += temp * temp
        if float(z[0]) > new_offset:
            new_offset = float(z[0])
    loss = -0.5 * torch.log((real_part ** 2 + imag_part ** 2) / l2_norm)
    return loss, new_offset


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


@click.group()
def cli():
    pass


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
    type=click.File(mode="wb"),
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
    "--epochs",
    type=click.IntRange(min=0),
    default=200,
    show_default=True,
    help="Number of learning steps to perform. Here step is defined as a single "
    "update of the parameters.",
)
@click.option(
    "--optimizer",
    type=str,
    default="SGD",
    help="Optimizer to use. Valid values are names of classes in torch.optim "
    "(i.e. 'SGD', 'Adam', 'Adamax', etc.)",
)
@click.option(
    "--time",
    "time_limit",
    type=click.FloatRange(min=1.0e-10),
    show_default=True,
    help="Time interval (in seconds) that specifies how often the model is written "
    "to the output file. If not specified, the weights are saved after every iteration.",
)
def train(nn_file, train_file, out_file, in_file, lr, optimizer, epochs, time_limit):
    """
    Supervised learning.
    """
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.DEBUG
    )
    Net = import_network(nn_file)
    target_wf, number_spins = load_explicit(train_file)
    psi = Net(number_spins)
    if in_file is not None:
        logging.info("Reading the initial weights...")
        psi.load_state_dict(torch.load(in_file))
    # psi._automorphisms = np.array([
    #     [0, 1, 2, 3, 4, 5],
    #     [1, 2, 3, 4, 5, 0],
    #     [2, 3, 4, 5, 0, 1],
    #     [3, 4, 5, 0, 1, 2],
    #     [4, 5, 0, 1, 2, 3],
    #     [5, 0, 1, 2, 3, 4],
    # ], dtype=np.int64)
    psi._automorphisms = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, 2, 1, 6, 8, 7, 3, 5, 4, 9, 11, 10],
            [0, 4, 8, 3, 1, 11, 6, 10, 2, 9, 7, 5],
            [0, 8, 4, 6, 2, 10, 3, 11, 1, 9, 5, 7],
            [1, 0, 2, 4, 3, 5, 10, 9, 11, 7, 6, 8],
            [1, 2, 0, 10, 11, 9, 4, 5, 3, 7, 8, 6],
            [1, 3, 11, 4, 0, 8, 10, 6, 2, 7, 9, 5],
            [1, 11, 3, 10, 2, 6, 4, 8, 0, 7, 5, 9],
            [2, 0, 1, 8, 6, 7, 11, 9, 10, 5, 3, 4],
            [2, 1, 0, 11, 10, 9, 8, 7, 6, 5, 4, 3],
            [2, 6, 10, 8, 0, 4, 11, 3, 1, 5, 9, 7],
            [2, 10, 6, 11, 1, 3, 8, 4, 0, 5, 7, 9],
            [3, 1, 11, 0, 4, 8, 9, 7, 5, 6, 10, 2],
            [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
            [3, 5, 4, 9, 11, 10, 0, 2, 1, 6, 8, 7],
            [3, 11, 1, 9, 5, 7, 0, 8, 4, 6, 2, 10],
            [4, 0, 8, 1, 3, 11, 7, 9, 5, 10, 6, 2],
            [4, 3, 5, 1, 0, 2, 7, 6, 8, 10, 9, 11],
            [4, 5, 3, 7, 8, 6, 1, 2, 0, 10, 11, 9],
            [4, 8, 0, 7, 5, 9, 1, 11, 3, 10, 2, 6],
            [5, 3, 4, 11, 9, 10, 8, 6, 7, 2, 0, 1],
            [5, 4, 3, 8, 7, 6, 11, 10, 9, 2, 1, 0],
            [5, 7, 9, 8, 4, 0, 11, 1, 3, 2, 10, 6],
            [5, 9, 7, 11, 3, 1, 8, 0, 4, 2, 6, 10],
            [6, 2, 10, 0, 8, 4, 9, 5, 7, 3, 11, 1],
            [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5],
            [6, 8, 7, 0, 2, 1, 9, 11, 10, 3, 5, 4],
            [6, 10, 2, 9, 7, 5, 0, 4, 8, 3, 1, 11],
            [7, 5, 9, 4, 8, 0, 10, 2, 6, 1, 11, 3],
            [7, 6, 8, 10, 9, 11, 4, 3, 5, 1, 0, 2],
            [7, 8, 6, 4, 5, 3, 10, 11, 9, 1, 2, 0],
            [7, 9, 5, 10, 6, 2, 4, 0, 8, 1, 3, 11],
            [8, 0, 4, 2, 6, 10, 5, 9, 7, 11, 3, 1],
            [8, 4, 0, 5, 7, 9, 2, 10, 6, 11, 1, 3],
            [8, 6, 7, 2, 0, 1, 5, 3, 4, 11, 9, 10],
            [8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 10, 9],
            [9, 5, 7, 3, 11, 1, 6, 2, 10, 0, 8, 4],
            [9, 7, 5, 6, 10, 2, 3, 1, 11, 0, 4, 8],
            [9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2],
            [9, 11, 10, 3, 5, 4, 6, 8, 7, 0, 2, 1],
            [10, 2, 6, 1, 11, 3, 7, 5, 9, 4, 8, 0],
            [10, 6, 2, 7, 9, 5, 1, 3, 11, 4, 0, 8],
            [10, 9, 11, 7, 6, 8, 1, 0, 2, 4, 3, 5],
            [10, 11, 9, 1, 2, 0, 7, 8, 6, 4, 5, 3],
            [11, 1, 3, 2, 10, 6, 5, 7, 9, 8, 4, 0],
            [11, 3, 1, 5, 9, 7, 2, 6, 10, 8, 0, 4],
            [11, 9, 10, 5, 3, 4, 2, 0, 1, 8, 6, 7],
            [11, 10, 9, 2, 1, 0, 5, 4, 3, 8, 7, 6],
        ],
        dtype=np.int64,
    )

    magnetisation = 0 if number_spins % 2 == 0 else 1
    # NOTE(twesterhout): This is a hack :)
    optimizer = getattr(torch.optim, optimizer)(psi.parameters(), lr=lr)

    # Function to overwrite the neural network weights
    def save():
        out_file.seek(0)
        out_file.truncate()
        torch.save(psi.state_dict(), out_file)

    def make_extented(wf):
        # target_wf_extented = copy.deepcopy(target_wf)
        # todo = len(target_wf)
        # max_count = 0
        # while todo > 0 and max_count < 10000:
        #     spin = int(CompactSpin(random_spin(number_spins, magnetisation)))
        #     if spin not in target_wf_extented:
        #         target_wf_extented[spin] = complex(0.0)
        #         todo -= 1
        #     max_count += 1
        return wf

    offset = 5.0  # None
    start = time.time()
    for i in range(epochs):
        target_wf_extented = make_extented(target_wf)
        optimizer.zero_grad()
        loss, _ = negative_log_overlap(psi, target_wf_extented, offset)
        logging.info("{}: Loss: {}".format(i + 1, loss))
        loss.backward()
        optimizer.step()
        if time_limit is None or time.time() - start > time_limit:
            save()
            start = time.time()


@cli.command()
@click.argument(
    "nn-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar="<arch_file>",
)
@click.option(
    "-i",
    "--in-file",
    type=click.File(mode="rb"),
    required=True,
    help="File containing the Neural Network weights as a PyTorch `state_dict` "
    "serialised using `torch.save`. It is up to the user to ensure that "
    "the weights are compatible with the architecture read from "
    "<arch_file>.",
)
@click.option(
    "-o",
    "--out-file",
    type=click.File(mode='w'),
    default=sys.stdout,
    show_default=True,
    help="Location where to save the sampled state.",
)
@click.option(
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
def sample(nn_file, in_file, out_file, hamiltonian_file, steps):
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
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.DEBUG
    )
    Machine = _make_machine(import_network(nn_file))
    H = read_hamiltonian(hamiltonian_file)
    psi = Machine(H.number_spins)
    psi.load_state_dict(torch.load(in_file))

    magnetisation = 0 if psi.number_spins % 2 == 0 else 1
    thermalisation = int(0.1 * steps)
    monte_carlo_steps = (
        thermalisation * psi.number_spins,
        (thermalisation + steps) * psi.number_spins,
        psi.number_spins,
    )
    E, var_E, wave_function = monte_carlo_loop_for_lanczos(
        psi, H, random_spin(psi.number_spins, magnetisation), monte_carlo_steps
    )
    # For normalisation
    scale = 1.0 / math.sqrt(sum(map(lambda x: abs(x) ** 2, wave_function.values())))
    out_file.write("# E = {} + {}\n".format(E.real, E.imag))
    out_file.write("# Var[E] = {} + {}".format(var_E.real, var_E.imag))
    fmt = "\n{:0" + str(psi.number_spins) + "b}\t{}\t{}"
    for (spin, coeff) in wave_function.items():
        out_file.write(fmt.format(int(spin), scale * coeff.real, scale * coeff.imag))


@cli.command()
@click.argument(
    "nn-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar="<arch_file>",
)
@click.option(
    "-i",
    "--in-file",
    type=click.File(mode="rb"),
    required=True,
    help="File containing the Neural Network weights as a PyTorch `state_dict` "
    "serialised using `torch.save`. It is up to the user to ensure that "
    "the weights are compatible with the architecture read from "
    "<arch_file>.",
)
@click.option(
    "-o",
    "--out-file",
    type=click.File(mode="w"),
    default=sys.stdout,
    show_default=True,
    help="Location where to save the statistics.",
)
@click.option("--spins", "n_spins", type=int)
@click.option("--exact", type=click.File(mode="rb"), help="Exact ground state")
def analyse(nn_file, in_file, out_file, n_spins, exact):
    """
    """
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.DEBUG
    )
    Machine = _make_machine(import_network(nn_file))
    if exact is not None:
        exact_psi, n_spins = load_explicit(exact)
    if n_spins is None:
        click.echo("Please, specify either `--exact` or `--spins`.")
        sys.exit(1)
    psi = Machine(n_spins)
    psi.load_state_dict(torch.load(in_file))
    magnetisation = 0 if psi.number_spins % 2 == 0 else 1

    l2_norm = mpmath.mpf(0)
    stats = np.empty(
        (int(scipy.special.comb(n_spins, (magnetisation + n_spins) // 2)),),
        dtype=[
            ("index", "i8"),
            ("spin", "U{}".format(n_spins)),
            ("real", "f4"),
            ("imag", "f4"),
            ("exact_real", "f4"),
            ("exact_imag", "f4"),
            ("key", "f4"),
        ],
    )
    for i, compact_spin in enumerate(all_spins(n_spins, magnetisation)):
        index = int(compact_spin)
        spin = compact_spin.numpy()
        wf = psi.log_wf(spin)
        wf_exact = (
            exact_psi.get(compact_spin, complex(0, 0))
            if exact_psi is not None
            else complex(float("nan"), float("nan"))
        )
        key = (
            wf_exact.real ** 2 + wf_exact.imag ** 2
            if exact_psi is not None
            else wf.real
        )
        stats[i] = (
            index,
            str(compact_spin),
            wf.real,
            wf.imag,
            wf_exact.real,
            wf_exact.imag,
            key,
        )
        l2_norm += mpmath.fabs(mpmath.exp(mpmath.mpc(wf))) ** 2
    scale = float(mpmath.log(mpmath.sqrt(l2_norm)))
    for i in range(stats.shape[0]):
        wf = cmath.exp(complex(stats[i]["real"] - scale, stats[i]["imag"]))
        stats[i]["real"] = wf.real
        stats[i]["imag"] = wf.imag
    stats.sort(order="key")
    np.savetxt(
        out_file,
        stats,
        fmt=["%i", "%s", "%.8e", "%.8e", "%.8e", "%.8e", "%.8e"],
        delimiter="\t",
    )


@cli.command()
@click.argument(
    "nn-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar="<arch_file>",
)
@click.option(
    "-i",
    "--in-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    required=True,
    help="",
)
# @click.option('-o', '--out-file',
#     type=click.File(mode='w'),
#     default=sys.stdout,
#     show_default=True,
#     help='Location where to save the sampled state.')
@click.option(
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
def sample_average(nn_file, in_file, hamiltonian_file, steps):
    """
    NOTE: DO NOT USE ME (YET).
    """
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.DEBUG
    )
    Net = import_network(nn_file)
    # Machine = _make_machine(Net)
    H = read_hamiltonian(hamiltonian_file)
    magnetisation = 0 if H.number_spins % 2 == 0 else 1
    AverageMachine = _make_machine(AverageNet)
    with open(in_file, "r") as f:
        input_files = map(lambda x: x.strip(), f.readlines())
    psi = AverageMachine(Net, H.number_spins, input_files)
    thermalisation = int(0.1 * steps)
    monte_carlo_steps = (
        thermalisation * psi.number_spins,
        (thermalisation + steps) * psi.number_spins,
        psi.number_spins,
    )
    psi.normalise_(monte_carlo_steps, magnetisation)

    monte_carlo_steps = (
        thermalisation * psi.number_spins,
        (thermalisation + 50 * steps) * psi.number_spins,
        psi.number_spins,
    )
    spin_fmt = "{:0" + str(psi.number_spins) + "b}"
    for magical_spin in [
        np.array([1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1], dtype=np.float32),
        np.array([1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1], dtype=np.float32),
        np.array([1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1], dtype=np.float32),
        np.array([1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1], dtype=np.float32),
    ]:
        logging.info(("S = " + spin_fmt).format(int(Spin.from_array(magical_spin))))
        psi.align_(magical_spin)
        E, var_E, _ = monte_carlo_loop_for_lanczos(
            psi, H, random_spin(psi.number_spins, magnetisation), monte_carlo_steps
        )
        logging.info("    E = {} + {}".format(E.real, E.imag))
        logging.info("    Var[E] = {} + {}".format(var_E.real, var_E.imag))


@cli.command()
@click.argument(
    "nn-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar="<arch_file>",
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
        format="[%(asctime)s] [%(levelname)s] %(message)s", level=logging.DEBUG
    )
    Machine = _make_machine(import_network(nn_file))
    H = read_hamiltonian(hamiltonian_file)
    psi = Machine(H.number_spins)
    if in_file is not None:
        logging.info("Reading the weights...")
        psi.load_state_dict(torch.load(in_file))
    # psi._automorphisms = np.array(
    #     [
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    #         [0, 2, 1, 6, 8, 7, 3, 5, 4, 9, 11, 10],
    #         [0, 4, 8, 3, 1, 11, 6, 10, 2, 9, 7, 5],
    #         [0, 8, 4, 6, 2, 10, 3, 11, 1, 9, 5, 7],
    #         [1, 0, 2, 4, 3, 5, 10, 9, 11, 7, 6, 8],
    #         [1, 2, 0, 10, 11, 9, 4, 5, 3, 7, 8, 6],
    #         [1, 3, 11, 4, 0, 8, 10, 6, 2, 7, 9, 5],
    #         [1, 11, 3, 10, 2, 6, 4, 8, 0, 7, 5, 9],
    #         [2, 0, 1, 8, 6, 7, 11, 9, 10, 5, 3, 4],
    #         [2, 1, 0, 11, 10, 9, 8, 7, 6, 5, 4, 3],
    #         [2, 6, 10, 8, 0, 4, 11, 3, 1, 5, 9, 7],
    #         [2, 10, 6, 11, 1, 3, 8, 4, 0, 5, 7, 9],
    #         [3, 1, 11, 0, 4, 8, 9, 7, 5, 6, 10, 2],
    #         [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
    #         [3, 5, 4, 9, 11, 10, 0, 2, 1, 6, 8, 7],
    #         [3, 11, 1, 9, 5, 7, 0, 8, 4, 6, 2, 10],
    #         [4, 0, 8, 1, 3, 11, 7, 9, 5, 10, 6, 2],
    #         [4, 3, 5, 1, 0, 2, 7, 6, 8, 10, 9, 11],
    #         [4, 5, 3, 7, 8, 6, 1, 2, 0, 10, 11, 9],
    #         [4, 8, 0, 7, 5, 9, 1, 11, 3, 10, 2, 6],
    #         [5, 3, 4, 11, 9, 10, 8, 6, 7, 2, 0, 1],
    #         [5, 4, 3, 8, 7, 6, 11, 10, 9, 2, 1, 0],
    #         [5, 7, 9, 8, 4, 0, 11, 1, 3, 2, 10, 6],
    #         [5, 9, 7, 11, 3, 1, 8, 0, 4, 2, 6, 10],
    #         [6, 2, 10, 0, 8, 4, 9, 5, 7, 3, 11, 1],
    #         [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5],
    #         [6, 8, 7, 0, 2, 1, 9, 11, 10, 3, 5, 4],
    #         [6, 10, 2, 9, 7, 5, 0, 4, 8, 3, 1, 11],
    #         [7, 5, 9, 4, 8, 0, 10, 2, 6, 1, 11, 3],
    #         [7, 6, 8, 10, 9, 11, 4, 3, 5, 1, 0, 2],
    #         [7, 8, 6, 4, 5, 3, 10, 11, 9, 1, 2, 0],
    #         [7, 9, 5, 10, 6, 2, 4, 0, 8, 1, 3, 11],
    #         [8, 0, 4, 2, 6, 10, 5, 9, 7, 11, 3, 1],
    #         [8, 4, 0, 5, 7, 9, 2, 10, 6, 11, 1, 3],
    #         [8, 6, 7, 2, 0, 1, 5, 3, 4, 11, 9, 10],
    #         [8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 10, 9],
    #         [9, 5, 7, 3, 11, 1, 6, 2, 10, 0, 8, 4],
    #         [9, 7, 5, 6, 10, 2, 3, 1, 11, 0, 4, 8],
    #         [9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2],
    #         [9, 11, 10, 3, 5, 4, 6, 8, 7, 0, 2, 1],
    #         [10, 2, 6, 1, 11, 3, 7, 5, 9, 4, 8, 0],
    #         [10, 6, 2, 7, 9, 5, 1, 3, 11, 4, 0, 8],
    #         [10, 9, 11, 7, 6, 8, 1, 0, 2, 4, 3, 5],
    #         [10, 11, 9, 1, 2, 0, 7, 8, 6, 4, 5, 3],
    #         [11, 1, 3, 2, 10, 6, 5, 7, 9, 8, 4, 0],
    #         [11, 3, 1, 5, 9, 7, 2, 6, 10, 8, 0, 4],
    #         [11, 9, 10, 5, 3, 4, 2, 0, 1, 8, 6, 7],
    #         [11, 10, 9, 2, 1, 0, 5, 4, 3, 8, 7, 6],
    #     ],
    #     dtype=np.int64,
    # )
    # psi._automorphisms = np.array([
    #     [0, 1, 2, 3, 4, 5],
    #     [1, 2, 3, 4, 5, 0],
    #     [2, 3, 4, 5, 0, 1],
    #     [3, 4, 5, 0, 1, 2],
    #     [4, 5, 0, 1, 2, 3],
    #     [5, 0, 1, 2, 3, 4],
    # ], dtype=np.int64)
    # psi._automorphisms = np.array(
    #     [
    #         [0, 1, 2, 3, 4, 5],
    #         [0, 5, 4, 3, 2, 1],
    #         [1, 0, 5, 4, 3, 2],
    #         [1, 2, 3, 4, 5, 0],
    #         [2, 1, 0, 5, 4, 3],
    #         [2, 3, 4, 5, 0, 1],
    #         [3, 2, 1, 0, 5, 4],
    #         [3, 4, 5, 0, 1, 2],
    #         [4, 3, 2, 1, 0, 5],
    #         [4, 5, 0, 1, 2, 3],
    #         [5, 0, 1, 2, 3, 4],
    #         [5, 4, 3, 2, 1, 0],
    #     ],
    #     dtype=np.int64,
    # )
    # psi._automorphisms = np.array([
    #     [0, 1, 2, 3, 4, 5, 6],
    #     [1, 2, 3, 4, 5, 6, 0],
    #     [2, 3, 4, 5, 6, 0, 1],
    #     [3, 4, 5, 6, 0, 1, 2],
    #     [4, 5, 6, 0, 1, 2, 3],
    #     [5, 6, 0, 1, 2, 3, 4],
    #     [6, 0, 1, 2, 3, 4, 5],
    # ], dtype=np.int64)
    # psi._automorphisms = np.array([
    #     [0, 1, 2, 3, 4, 5, 6, 7],
    #     [1, 2, 3, 4, 5, 6, 7, 0],
    #     [2, 3, 4, 5, 6, 7, 0, 1],
    #     [3, 4, 5, 6, 7, 0, 1, 2],
    #     [4, 5, 6, 7, 0, 1, 2, 3],
    #     [5, 6, 7, 0, 1, 2, 3, 4],
    #     [6, 7, 0, 1, 2, 3, 4, 5],
    #     [7, 0, 1, 2, 3, 4, 5, 6],
    # ], dtype=np.int64)

    magnetisation = 0 if psi.number_spins % 2 == 0 else 1
    thermalisation = int(0.1 * steps)
    opt = Optimiser(
        psi,
        H,
        magnetisation=magnetisation,
        epochs=epochs,
        monte_carlo_steps=(
            thermalisation * psi.number_spins,
            (thermalisation + steps) * psi.number_spins,
            psi.number_spins,
        ),
        learning_rate=lr,
        use_sr=use_sr,
        regulariser=lambda i: 100.0 * 0.9 ** i + 0.01,
        model_file=out_file,
        time_limit=time_limit,
    )
    opt()
    print(
        compute_l2_norm(
            psi,
            random_spin(psi.number_spins, magnetisation),
            (1000, 1000 + 10000 * psi.number_spins, psi.number_spins),
        )
    )


if __name__ == "__main__":
    cli()
    # cProfile.run('main()')
