import numpy as np
import torch

import nqs_playground


class SlowPolynomialState:
    def __init__(self, hamiltonian, roots, log_psi):
        self.hamiltonian = hamiltonian
        self.basis = hamiltonian.basis
        self.basis.build()
        self.state, self.scale = self._make_state(log_psi)
        self.roots = roots

    def _make_state(self, log_psi):
        with torch.no_grad():
            spins = torch.from_numpy(self.basis.states.view(np.int64))
            out = log_psi(torch.ops.tcm.unpack(spins, self.basis.number_spins))
            scale = torch.max(out[:, 0]).item()
            out[:, 0] -= scale
            out = out.numpy().view(np.complex64).squeeze()
            out = np.exp(out)
            return out, scale

    def _forward_one(self, x):
        basis = self.hamiltonian.basis
        i = basis.index(x)
        vector = np.zeros((basis.number_states,), dtype=np.complex64)
        vector[i] = 1.0
        for r in self.roots:
            vector = self.hamiltonian(vector) - r * vector
        return self.scale + np.log(np.dot(self.state, vector))

    def __call__(self, spins):
        if isinstance(spins, torch.Tensor):
            assert spins.dtype == torch.int64
            spins = spins.numpy().view(np.uint64)
        return torch.from_numpy(
            np.array([self._forward_one(x) for x in spins], dtype=np.complex64)
            .view(np.float32)
            .reshape(-1, 2)
        )


basis = nqs_playground.SpinBasis([], 16, 8)
basis.build()
hamiltonian = nqs_playground.Heisenberg(
    [(1.0, i, (i + 1) % basis.number_spins) for i in range(basis.number_spins)], basis
)
m = torch.jit.script(
    torch.nn.Sequential(
        torch.nn.Linear(basis.number_spins, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )
)

spins = torch.from_numpy(basis.states.view(np.int64))

A = SlowPolynomialState(hamiltonian, [0.0], m)(spins)
B = nqs_playground._C.apply(spins, hamiltonian, m._c._get_method("forward"))
if not torch.allclose(A, B):
    for i in range(A.size(0)):
        if not torch.allclose(A[i], B[i]):
            print(i, A[i], B[i])
            print(A[i - 3 : i + 3])
            print(B[i - 3 : i + 3])
            break
else:
    print("[+] passed!")
