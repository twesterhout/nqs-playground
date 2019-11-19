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

r"""This module defines routines for constructing symmetry operators and
building symmetry groups. It will work properly only if the group
representation is one-dimensional representation.
"""

__all__ = ["Symmetry", "make_group", "diagonalise"]

from cmath import exp
from cmath import pi as π
from fractions import Fraction
from typing import List

import numpy as np

from ._benes import make_perm_fn
from .core import _C


class Symmetry:
    r"""Class representing symmetry operators which are permutations.
    """

    def __init__(self, permutation: List[int], sector: int):
        r"""Constructs a symmetry from a permutation.

        :param permutation: a list of integers specifying the permutation.
        :param sector: index of the sector to which to restrict the Hilbert
            space.
        """
        self._map = np.asarray(permutation, dtype=np.int32, order="C")
        self._map.flags.writeable = False
        if np.any(np.sort(self._map) != np.arange(len(self._map), dtype=np.int32)):
            raise ValueError(
                "`permutation` is not a valid permutation: {}".format(permutation)
            )

        self._periodicity = Symmetry._compute_periodicity(self._map)
        if sector < 0 or sector >= self._periodicity:
            raise ValueError(
                "invalid `sector`: {}; permutation {} has periodicity {}"
                "".format(sector, list(self._map), self._periodicity)
            )
        self._sector = sector
        self._network = None

    @staticmethod
    def _compute_periodicity(p: np.ndarray) -> int:
        r"""Given a permutation ``p``, computes its periodicity, i.e. the
        smallest positive integer ``n`` such that ``p^n == id``.
        """
        identity = np.arange(len(p), dtype=np.int32)
        x = identity[p]
        count = 1
        while not np.array_equal(x, identity):
            count += 1
            x = x[p]
        # if count == 1:
        #     warnings.warn("`permutation` is an identity mapping")
        return count

    @property
    def sector(self) -> int:
        r"""Returns the sector to which we restricted the Hilbert space.
        """
        return self._sector

    @property
    def periodicity(self) -> int:
        r"""Returns the periodicity of the symmetry T, i.e. the smallest
        positive integer ``n`` such that ``T^n == id``.
        """
        return self._periodicity

    @property
    def phase(self) -> Fraction:
        return Fraction(self._sector, self._periodicity)

    @property
    def eigenvalue(self) -> complex:
        r"""Returns the eigenvalue of this operator corresponding to the ground
        state.
        """
        return exp(-2j * π * self.sector / self.periodicity)

    @property
    def permutation(self) -> np.ndarray:
        r"""Returns the permutation map."""
        return self._map

    def to_cxx(self):
        r"""Returns the C++ equivalent of this symmetry operator."""
        if self._network is None:
            self._network = make_perm_fn(self._map)
        return _C.v2.Symmetry(
            [self._network.left, self._network.right], self.sector, self.periodicity
        )

    def __mul__(self, other):
        r"""Group operation."""
        if len(self._map) != len(other._map):
            raise ValueError("symmetries are defined on different spaces")
        x = Fraction(self.sector, self.periodicity) + Fraction(
            other.sector, other.periodicity
        )
        periodicity = Symmetry._compute_periodicity(other._map[self._map])
        r = Symmetry(
            other._map[self._map],
            sector=(x.numerator % x.denominator) * (periodicity // x.denominator),
        )
        return r

    def __eq__(self, other):
        if len(self._map) != len(other._map):
            raise ValueError(
                "symmetries are defined on different spaces: {}, {}".format(self, other)
            )
        if np.array_equal(self._map, other._map):
            if self.sector != other.sector:
                raise ValueError(
                    "symmetries are defined on different sectors: {}, {}".format(
                        self, other
                    )
                )
            return True
        return False

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self._map.data.tobytes())

    def __str__(self):
        return "Symmetry({}, {}, {})".format(
            self.permutation, self.sector, self.periodicity
        )


def _make_cyclic_group(symmetry):
    group = []
    g = Symmetry(np.arange(len(symmetry.permutation)), sector=0)
    for i in range(symmetry.periodicity):
        group.append(g)
        g = symmetry * g
    return group


def make_group(symmetries: List[Symmetry]):
    r"""Given a list of symmetries, extends this list into a group."""
    group = set(sum((_make_cyclic_group(t) for t in symmetries), []))
    while True:
        extra = []
        for x in group:
            for y in group:
                g = x * y
                if g not in group:
                    extra.append(g)
        if len(extra) == 0:
            break
        for g in extra:
            group.add(g)
    return [g.to_cxx() for g in group]


def diagonalise(hamiltonian, k=2):
    import scipy.sparse.linalg

    hamiltonian.basis.build()
    n = hamiltonian.basis.number_states
    dtype = np.float64 if hamiltonian.is_real else np.complex128

    def matvec(x):
        y = np.empty(n, dtype=dtype)
        hamiltonian(x, y)
        return y

    op = scipy.sparse.linalg.LinearOperator(shape=(n, n), matvec=matvec, dtype=dtype)
    return scipy.sparse.linalg.eigsh(op, k=k, which="SA")
