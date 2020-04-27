import torch
from torch import Tensor

from nqs_playground import *



def unpack_simple(spins: Tensor, n: int) -> Tensor:

    def _unpack_one(bits: int) -> Tensor:
        out = torch.empty(n, dtype=torch.float32)
        for i in range(n):
            out[i] = float(2 * ((bits >> i) & 1) - 1)
        return out

    device = spins.device
    spins = spins.cpu()
    return torch.stack([_unpack_one(x) for x in spins]).to(device)


def test_unpack(device):
    ns = torch.randint(1, 64, (10,))
    for n in map(lambda x: x.item(), ns):
        spins = torch.randint(0, (1<<n), (587,)).to(device)
        predicted = unpack(spins, n)
        expected = unpack_simple(spins, n)
        assert (predicted == expected).all()

test_unpack('cuda')
test_unpack('cpu')
