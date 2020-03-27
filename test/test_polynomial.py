import numpy as np
import torch

import nqs_playground
import nqs_playground._C as _C
from nqs_playground import *

def test_call():
    basis = SpinBasis([], 10, 5)
    basis.build()
    hamiltonian = Heisenberg([(1, i, (i + 1) % 10) for i in range(10)], basis)
    dense = hamiltonian.to_csr().todense()
    dense = (dense - 10 * np.eye(dense.shape[0])) @ (dense - (5.0 + 3j) * np.eye(dense.shape[0]))
    polynomial = _C.Polynomial(hamiltonian, [10.0, 5.0 + 3j])
    for i in range(basis.number_states):
        for j in range(basis.number_states):
            x = dense[i, j]
            _state = polynomial(int(basis.states[j]))
            y = _state[basis.states[i]] if int(basis.states[i]) in _state else 0.0
            assert x == y


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
            out = log_psi(spins)
            scale = torch.max(out[:, 0]).item()
            out[:, 0] -= scale
            out = out.numpy().view(np.complex64).squeeze().astype(np.complex128)
            out = np.exp(out)
            return out, scale

    def _forward_one(self, x):
        basis = self.hamiltonian.basis
        i = basis.index(x)
        vector = np.zeros((basis.number_states,), dtype=np.complex128)
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

def test_apply():
    basis = SpinBasis([], 14, 7)
    basis.build()
    hamiltonian = Heisenberg([(1, i, (i + 1) % 14) for i in range(14)], basis)
    log_psi = torch.jit.script(torch.nn.Sequential(
        nqs_playground.core.Unpack(14),
        torch.nn.Linear(14, 5),
        torch.nn.Tanh(),
        torch.nn.Linear(5, 2)
    ))
    roots = [8.0, 5.0 + 3j, 1.0 -0.5j]
    spins = torch.from_numpy(basis.states.view(np.int64))
    expected = SlowPolynomialState(hamiltonian, roots, log_psi)(spins)
    spins = torch.cat([spins.view(-1, 1), torch.zeros((spins.size(0), 7), dtype=torch.int64)], dim=1)
    predicted = _C.apply(spins, _C.Polynomial(hamiltonian, roots), log_psi._c._get_method("forward"))
    assert torch.allclose(predicted, expected, rtol=1e-4, atol=1e-6)
    if torch.cuda.is_available():
        spins = spins.cuda()
        log_psi.cuda()
        predicted = _C.apply(spins, _C.Polynomial(hamiltonian, roots), log_psi._c._get_method("forward"))
        predicted = predicted.cpu()
        assert torch.allclose(predicted, expected, rtol=1e-4, atol=1e-6)

test_apply()

