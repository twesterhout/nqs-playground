# NOTE: We need to import PyTorch first, because we don't explicitly link
# against it in C++ code.
import torch as __torch
# TODO: Link against liblattice_symmetries properly
import lattice_symmetries as __ls

# from . import _C
# from .core import *
# from .symmetry import *


# def SpinBasis(symmetries, number_spins, hamming_weight=None):
#     small = lambda: _C.SmallSpinBasis(symmetries, number_spins, hamming_weight)
#     big = lambda: _C.BigSpinBasis(symmetries, number_spins, hamming_weight)
#     if len(symmetries) == 0:
#         return small() if number_spins <= 64 else big()
#     else:
#         t = next(iter(symmetries))
#         if isinstance(t, _C.Symmetry64):
#             return small()
#         if isinstance(t, _C.Symmetry512):
#             return big()
#         raise TypeError(
#             "symmetries has wrong type: List[{}]; expected either "
#             "List[_C.Symmetry64]or List[_C.Symmetry512]".format(type(t))
#         )


# from .hamiltonian import *
# from .monte_carlo import *
# from ._jacobian import *
# from ._C import manual_seed

# This operator becomes available only after loading _C
# unpack = __torch.ops.tcm.unpack
