import numpy as np
from nqs_playground import *


def test_construction():
    _ = SpinBasis([], number_spins=10, hamming_weight=5)
    _ = SpinBasis([], number_spins=200, hamming_weight=100)
    _ = SpinBasis([], number_spins=20)
    _ = SpinBasis(
        make_group(
            [
                Symmetry([1, 2, 3, 4, 5, 0], sector=3),
                Symmetry([5, 4, 3, 2, 1, 0], sector=1),
            ]
        ),
        6,
    )
    extend = lambda x: x + list(range(len(x), 500))
    _ = SpinBasis(
        make_group(
            [
                Symmetry(extend([1, 2, 3, 4, 5, 0]), sector=3),
                Symmetry(extend([5, 4, 3, 2, 1, 0]), sector=1),
            ]
        ),
        500,
    )


def test_building():
    basis = SpinBasis([], number_spins=6, hamming_weight=3)
    basis.build()
    assert basis.number_spins == 6
    assert basis.hamming_weight == 3
    assert basis.number_states == 20
    assert basis.index(int("111000", base=2)) == 19
    assert basis.index(int("000111", base=2)) == 0
    assert basis.index(int("001011", base=2)) == 1
    assert basis.index(int("010101", base=2)) == 5


# def to_int(k) -> int:
#     acc = int(k[7])
#     for i in range(6, -1, -1):
#         acc <<= 64
#         acc |= k[i]
#     return acc


def test_full_info():
    L_x, L_y = 6, 4
    indices = np.arange(L_x * L_y).reshape(L_y, L_x)[::-1]
    x = indices % L_x
    y = indices // L_x
    T_x = (x + 1) % L_x + y * L_x
    T_y = x % L_x + ((y + 1) % L_y) * L_x

    T_x = T_x.reshape(-1)
    T_y = T_y.reshape(-1)

    indices = np.arange(24)
    np.random.shuffle(indices)
    def shuffle(xs):
        return [xs[i] for i in indices]

    basis_small = SpinBasis(
        shuffle(make_group([Symmetry(T_x, sector=1), Symmetry(T_y, sector=0)]))
        number_spins=L_x * L_y,
        hamming_weight=(L_x * L_y) // 2,
    )
    e = lambda p: p.tolist() + list(range(L_x * L_y, 100))
    basis_big = SpinBasis(
        shuffle(make_group([Symmetry(e(T_x), sector=1), Symmetry(e(T_y), sector=0)])),
        number_spins=L_x * L_y,
        hamming_weight=(L_x * L_y) // 2,
    )

    basis_small.build()
    for state in basis_small.states:
        r1, e1, n1 = basis_small.full_info(state)
        r2, e2, n2 = basis_big.full_info(state)
        assert r1 == r2
        assert e1 == e2
        assert n1 == n2
    
    basis_dummy = SpinBasis([], L_x * L_y, (L_x * L_y) // 2)
    basis_dummy.build()
    for state in basis_dummy.states:
        r1, e1, n1 = basis_small.full_info(state)
        r2, e2, n2 = basis_big.full_info(state)
        assert r1 == r2
        assert n1 == n2
        if n1 != 0.0:
            assert e1 == e2
        else:
            assert np.isnan(e2)
