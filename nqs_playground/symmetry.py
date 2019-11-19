import cmath
import warnings
from fractions import Fraction

import numpy as np

from ._benes import make_perm_fn
from .core import _C

__all__ = ["Symmetry", "make_group", "diagonalise"]


class Symmetry:
    r"""Class representing symmetry operators which are permutations.
    """

    def __init__(self, permutation, sector):
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
    def _compute_periodicity(p) -> int:
        r"""Given a permutation ``p``, computes its periodicity, i.e. the
        smallest positive integer ``n`` such that ``p^n == id``.
        """
        identity = np.arange(len(p), dtype=np.int32)
        x = identity[p]
        count = 1
        while not np.array_equal(x, identity):
            count += 1
            x = x[p]
        if count == 1:
            warnings.warn("`permutation` is an identity mapping")
        return count

    @property
    def sector(self):
        r"""Returns the sector to which we restricted the Hilbert space.
        """
        return self._sector

    @property
    def periodicity(self):
        r"""Returns the periodicity of the symmetry T, i.e. the smallest
        positive integer ``n`` such that ``T^n == id``.
        """
        return self._periodicity

    @property
    def phase(self):
        return Fraction(self._sector, self._periodicity)

    @property
    def eigenvalue(self):
        r"""Returns the eigenvalue of this operator corresponding to the ground
        state.
        """
        return cmath.exp(-2j * cmath.pi * self.sector / self.periodicity)

    @property
    def permutation(self):
        return self._map

    def to_cxx(self):
        if self._network is None:
            self._network = make_perm_fn(self._map)
        return _C.v2.Symmetry(
            [self._network.left, self._network.right], self.sector, self.periodicity
        )

    def __mul__(self, other):
        # print(self, "*", other, end=" ")
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
        # print("=", r)
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


def make_cyclic_group(symmetry):
    group = []
    g = Symmetry(np.arange(len(symmetry.permutation)), sector=0)
    for i in range(symmetry.periodicity):
        group.append(g)
        g = symmetry * g
    return group


def make_group(symmetries):
    group = set(sum((make_cyclic_group(t) for t in symmetries), []))
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
