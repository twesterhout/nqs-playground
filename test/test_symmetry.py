from nqs_playground import *
from nqs_playground import _C
import numpy as np

# Reference implementation
def _btfly_step(x: int, m: int, d: int) -> int:
    y = (x ^ (x >> d)) & m
    return x ^ y ^ (y << d)


def test_butterfly_step():
    def _to_int(k: np.ndarray) -> int:
        acc = int(k[7])
        for i in range(6, -1, -1):
            acc <<= 64
            acc |= k[i]
        return acc

    def _to_array(k: int) -> np.ndarray:
        return np.array([((k >> (64 * i)) & 0xFFFFFFFFFFFFFFFF) for i in range(8)], dtype=np.uint64)

    for i in range(9):
        d = 1 << i
        xs = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=(10000, 8), dtype=np.uint64)
        ms = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=(10000, 8), dtype=np.uint64)
        for x, m in zip(xs, ms):
            predicted = _C._btfly(x, m, d)
            expected = _to_array(_btfly_step(_to_int(x), _to_int(m), d)).tolist()
            assert predicted == expected

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

# def test_btfly():
#     x = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=8, dtype=np.uint64)
#     m = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, size=8, dtype=np.uint64)
#     d = 128
# 
#     def _to_int(k: np.ndarray) -> int:
#         acc = int(k[7])
#         for i in range(6, -1, -1):
#             acc <<= 64
#             acc |= k[i]
#         return acc
# 
#     def _to_array(k: int) -> np.ndarray:
#         return np.array([((k >> (64 * i)) & 0xFFFFFFFFFFFFFFFF) for i in range(8)], dtype=np.uint64)
# 
#     assert np.all(_to_array(_to_int(x)) == x)
#     assert np.all(_to_array(_to_int(m)) == m)
# 
#     print(x, bin(x[7]))
#     print(m)
#     return _to_array(_btfly_step(_to_int(x), _to_int(m), d)).tolist(), _C._btfly(x, m, d)

test_butterfly_step()
# a, b = test_btfly()
# print(a)
# print(b)

    
