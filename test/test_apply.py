import numpy as np
import torch
import lattice_symmetries as ls

try:
    import nqs_playground as nqs
except ImportError:
    # For local development when we only compile the C++ extension, but don't
    # actually install the package using pip
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    import nqs_playground as nqs


def test_apply():
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

    log_psi = torch.jit.script(MyModule(basis.number_spins))

    for batch_size in range(1000):
        for inference_batch_size in range(1):
            states = basis.states[np.random.permutation(basis.number_states)[:batch_size]]
            states = torch.from_numpy(states.view(np.int64))
            states = torch.cat(
                [states.view(-1, 1), torch.zeros(states.size(0), 7, dtype=torch.int64)], dim=1
            )
            predicted = nqs._C.log_apply(states, op, log_psi, batch_size)
            # expected = nqs.reference_log_apply(states, op, log_psi, batch_size)
            # assert torch.allclose(predicted, expected)


test_apply()
