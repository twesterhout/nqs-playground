import numpy as np
import torch
try:
    from nqs_playground import *
except ImportError:
    # For local development when we only compile the C++ extension, but don't
    # actually install the package using pip
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    from nqs_playground import *

np.random.seed(52339877)
torch.manual_seed(9218823294)
# manual_seed(3362121853)


@torch.no_grad()
def unpack_complete(spins):
    def unpack_one(bits, out):
        bits = bits.numpy().view(np.uint64)
        for i in range(8):
            word = int(bits[i])
            for j in range(64):
                out[64 * i + j] = float(2 * ((word >> j) & 1) - 1)

    device = spins.device
    spins = spins.cpu()
    out = torch.empty(spins.size(0), 512, dtype=torch.float32)
    for i in range(spins.size(0)):
        unpack_one(spins[i], out[i])
    return out.to(device)


@torch.no_grad()
def generate(batch_size, device):
    data = np.random.default_rng().integers(
        1 << 64 - 1, size=(batch_size, 8), dtype=np.uint64, endpoint=True
    )
    data = torch.from_numpy(data.view(np.int64)).clone().to(device)
    return data


@torch.no_grad()
def test_unpack(device):
    for batch_size in [1, 5, 128, 437]:
        packed = generate(batch_size, device)
        full = unpack_complete(packed)
        for number_spins in [1, 2, 32, 64, 65, 400]:
            predicted = unpack(packed, number_spins)
            expected = full[:, :number_spins]
            assert (predicted == expected).all()


test_unpack("cpu")
if torch.cuda.is_available():
    test_unpack("cuda")
