import numpy as np
import torch
import lattice_symmetries as ls
import nqs_playground as nqs

from nqs_playground._extension import lib as _C

def test_apply(use_jit=False):
    # ls.enable_logging()
    basis = ls.SpinBasis(ls.Group([]), number_spins=10, hamming_weight=None)
    basis.build()
    matrix = np.array([[1, 0, 0, 0], [0, -1, -2, 0], [0, -2, -1, 0], [0, 0, 0, 1]])
    edges = [(i, (i + 1) % basis.number_spins) for i in range(basis.number_spins)]
    op = ls.Operator(basis, [ls.Interaction(matrix, edges)])

    class MyModule(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.fn = torch.nn.Sequential(
                nqs.Unpack(n),
                torch.nn.Linear(n, 50),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(50, 2, bias=False),
            )

        def forward(self, x):
            y = self.fn(x)
            return torch.complex(y[:, 0], y[:, 1])

    log_psi = MyModule(basis.number_spins)
    if use_jit == True:
        log_psi = torch.jit.script(log_psi)

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device in devices:
        for batch_size in range(1, 10):
            for inference_batch_size in range(1, 20):
                states = basis.states[np.random.permutation(basis.number_states)[:batch_size]]
                states = torch.from_numpy(states.view(np.int64))
                states = nqs.pad_states(states)
                predicted = _C.log_apply(states.to(device), op, log_psi.to(device), batch_size)
                expected = nqs.reference_log_apply(states.cpu(), op, log_psi.cpu(), batch_size)
                assert torch.allclose(predicted.cpu(), expected)


test_apply()
