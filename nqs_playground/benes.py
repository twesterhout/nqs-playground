from collections import namedtuple
import math
from typing import Iterable, List, Tuple

__all__ = ["BenesNetwork", "make_perm_fn"]


def _btfly_step(x, m, d):
    print(x, m, d, end=" ")
    y = (x ^ (x >> d)) & m
    x = x ^ y ^ (y << d)
    print(x)
    return x


class BenesNetwork:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __call__(self, x):
        for i, m in enumerate(self.left):
            x = _btfly_step(x, m, 1 << i)
        for i, m in enumerate(self.right[::-1]):
            x = _btfly_step(x, m, 1 << (len(self.right) - 1 - i))
        return x


class _BenesBuilder:
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
        return 1 << math.ceil(math.log2(x))

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
        for i in range(int(math.log2(len(tgt)))):
            d = 1 << i
            fwd, bwd = _BenesBuilder.solve_stage(src, tgt, d)
            left.append(_BenesBuilder.make_mask(src, fwd, d))
            right.append(_BenesBuilder.make_mask(tgt, bwd, d))
            src = fwd
            tgt = bwd
        return BenesNetwork(left, right)


def make_perm_fn(p, backend: str = "python"):
    n = len(p)
    if not _BenesBuilder.is_pow_of_2(n):
        return make_perm_fn(
            list(p) + list(range(n, _BenesBuilder.round_up_pow_2(n))), backend=backend
        )
    if set(p) != set(range(n)):
        raise ValueError(
            "invalid p: {0}; must be a bijection {{0..{1}}}->{{0..{1}}}"
            "".format(p, n - 1)
        )
    if backend not in {"python"}:
        raise ValueError("invalid backend: {}; must be 'python'".format(mode))
    return _BenesBuilder.make(p)


def make_perm_fn_simple(p: List[int]):
    n = len(p)

    def fn(x):
        assert 0 <= x and x < (1 << n)
        s = "{1:0{0}b}".format(n, x)[::-1]
        s = "".join(s[i] for i in p)
        return int(s[::-1], base=2)

    return fn


def test_benes(n):
    import numpy as np

    p = np.arange(n)
    for i in range(100):
        np.random.shuffle(p)
        benes = make_perm_fn(p)
        simple = make_perm_fn_simple(p)
        if n <= 8:
            for x in range(1 << n):
                assert benes(x) == simple(x)
        else:
            for x in np.random.randint(low=0, high=1 << n, size=1000, dtype=np.uint64):
                assert benes(x) == simple(x)
