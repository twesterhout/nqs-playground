
__version__ = '1.0.0'

# NOTE: We need to import PyTorch first, because we don't explicitly link
# against it in C++ code.
import torch as __torch

from .core import *
from .hamiltonian import *
from .sampling import *
from ._jacobian import *
from ._C import manual_seed, random_spin

# This operator becomes available only after loading _C
unpack = __torch.ops.tcm.unpack
hamming_weight = __torch.ops.tcm.hamming_weight
