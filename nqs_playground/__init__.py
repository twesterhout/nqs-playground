
__version__ = '1.0.1'

# NOTE: We need to import PyTorch first, because we don't explicitly link
# against it in C++ code.
# import torch as __torch

from .core import *
from .sampling import *
# from ._extension import PACKAGE_DIR, lib
# from .hamiltonian import *
# from ._jacobian import *
# from ._C import manual_seed, random_spin

# from .runner import *

# This operator becomes available only after loading _C
# unpack = __torch.ops.tcm.unpack
# hamming_weight = __torch.ops.tcm.hamming_weight
