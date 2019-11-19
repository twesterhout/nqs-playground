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

r"""Implementation of Benes network for bit permutation.

This module contains a ``make_perm_fn`` function which given a permutation
p constructs a Benes network to efficiently shuffle bits in an integer
according to p. ``BenesNetwork`` is a class representing such networks.
"""

__all__ = ["BenesNetwork", "make_perm_fn", "test_benes"]

from math import ceil, log2
from typing import Iterable, List, Tuple


def _btfly_step(x: int, m: int, d: int) -> int:
    r"""Performs the so called butterfly operation on ``x`` swapping pairs of
    bits distance ``d`` from each other according to mask ``m``.
    """
    y = (x ^ (x >> d)) & m
    return x ^ y ^ (y << d)


class BenesNetwork:
    r"""Class representing Benes networks."""

    def __init__(self, left: List[int], right: List[int]):
        self.left = left
        self.right = right

    def __call__(self, x: int) -> int:
        r"""Applies the permutation to x."""
        x = int(x)
        for i, m in enumerate(self.left):
            x = _btfly_step(x, m, 1 << i)
        for i, m in enumerate(self.right[::-1]):
            x = _btfly_step(x, m, 1 << (len(self.right) - 1 - i))
        return x


class _BenesBuilder:
    r"""A builder class for Benes networks. This is not a real class. All its
    methods are static, so it acts more like a namespace.
    """

    @staticmethod
    def is_pow_of_2(x: int) -> bool:
        r"""Returns whether an integer is a power of 2."""
        return x > 0 and (x & (x - 1)) == 0

    @staticmethod
    def round_up_pow_2(x: int) -> int:
        r"""Returns the number rounded up to the closest power of two. If ``x``
        is already a power of 2, then ``x`` itself is returned.
        """
        assert x > 0
        return 1 << ceil(log2(x))

    @staticmethod
    def pairs(n: int, d: int) -> Iterable[Tuple[int, int]]:
        r"""Iterates over all disjoint pairs of indices with distance ``d``.

        :param n: length of the sequence over which we iterate.
        :param d: distance between indices in each pair.
        
        Examples:

        >>> list(pairs(8, 1))
        [(0, 1), (2, 3), (4, 5), (6, 7)]
        >>> list(pairs(8, 2))
        [(0, 2), (1, 3), (4, 6), (5, 7)]
        """
        assert d > 0 and n // 2 >= d
        i = 0
        while i < n - d:
            for _ in range(d):
                yield i, i + d
                i += 1
            i += d

    @staticmethod
    def find_conn(x: int, ys: List[int], d: int) -> Tuple[int, int]:
        r"""Iterates over ``ys`` using :py:func:`pairs` function and returns
        indices of a pair which contains ``x``.
        """
        for i, j in _BenesBuilder.pairs(len(ys), d):
            if ys[i] == x or ys[j] == x:
                return i, j
        assert False

    @staticmethod
    def make_mask(src: List[int], tgt: List[int], d: int) -> int:
        r"""Returns a mask ``m`` which allows one to reorder ``src`` to get
        ``fwd`` by applying a "2^d--swap with mask" (see eq. (69) in section
        7.1.3 of "The Art of Computer Programming, Volume 4a" by D. Knuth)
        """
        mask = 0
        for i, j in _BenesBuilder.pairs(len(src), d):
            if src[i] != tgt[i]:
                assert (src[i], src[j]) == (tgt[j], tgt[i])
                mask |= 1 << i
            else:
                assert (src[i], src[j]) == (tgt[i], tgt[j])
        return mask

    @staticmethod
    def solve_cycle(
        node: Tuple[int, bool],
        left: Tuple[List[int], List[int]],
        right: Tuple[List[int], List[int]],
        end: int,
        d: int,
    ):
        a, is_left = node
        src, fwd = left
        tgt, bwd = right

        first, second = _BenesBuilder.find_conn(a, tgt, d)
        if not is_left:
            first, second = second, first

        bwd[first] = a
        bwd[second] = tgt[second if tgt[first] == a else first]
        a = bwd[second]

        if a != end:
            return _BenesBuilder.solve_cycle((a, not is_left), right, left, end, d)

    @staticmethod
    def solve_stage(
        src: List[int], tgt: List[int], d: int
    ) -> Tuple[List[int], List[int]]:
        fwd = [None for _ in src]
        bwd = [None for _ in tgt]
        for i, j in reversed(list(_BenesBuilder.pairs(len(src), d))):
            if bwd[i] is None:
                assert bwd[j] is None
                bwd[i], bwd[j] = tgt[i], tgt[j]
                _BenesBuilder.solve_cycle(
                    (bwd[i], True), (tgt, bwd), (src, fwd), bwd[j], d
                )
        return fwd, bwd

    @staticmethod
    def make(tgt: List[int]) -> BenesNetwork:
        src = list(range(len(tgt)))
        left, right = [], []
        for i in range(int(log2(len(tgt)))):
            d = 1 << i
            fwd, bwd = _BenesBuilder.solve_stage(src, tgt, d)
            left.append(_BenesBuilder.make_mask(src, fwd, d))
            right.append(_BenesBuilder.make_mask(tgt, bwd, d))
            src = fwd
            tgt = bwd
        return BenesNetwork(left, right)


def make_perm_fn(p: List[int], bits=None):
    n = len(p)
    if set(p) != set(range(n)):
        raise ValueError(
            "invalid p: {0}; must be a bijection {{0..{1}}}->{{0..{1}}}"
            "".format(p, n - 1)
        )
    if bits is not None:
        if not _BenesBuilder.is_pow_of_2(bits) or bits < n:
            raise ValueError(
                "invalid bits: {}; bits must be a positive power of 2 not smaller "
                "than {}".format(bits, _BenesBuilder.round_up_pow_2(n))
            )
    else:
        bits = _BenesBuilder.round_up_pow_2(n)
    if n < bits:
        p = list(p) + list(range(n, bits))
    return _BenesBuilder.make(p)


def test_benes(n, bits=None):
    import numpy as np

    def make_perm_fn_simple(p: List[int]):
        def fn(x):
            assert 0 <= x and x < (1 << len(p))
            s = "{1:0{0}b}".format(len(p), x)[::-1]
            s = "".join(s[i] for i in p)
            return int(s[::-1], base=2)

        return fn

    p = np.arange(n)
    for i in range(100):
        np.random.shuffle(p)
        benes = make_perm_fn(p, bits)
        simple = make_perm_fn_simple(p)
        if n <= 8:
            for x in range(1 << n):
                assert benes(x) == simple(x)
        else:
            for x in np.random.randint(low=0, high=1 << n, size=1000, dtype=np.uint64):
                assert benes(x) == simple(x)
