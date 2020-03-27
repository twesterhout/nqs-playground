import numpy as np
import torch
from nqs_playground import *
from nqs_playground import _C

try:
    from typing_extensions import Final
except ImportError:
    # If you don't have `typing_extensions` installed, you can use a
    # polyfill from `torch.jit`.
    from torch.jit import Final


def test_exact_diagonalisation():
    n = 10
    basis = SpinBasis([], number_spins=n, hamming_weight=n // 2)
    hamiltonian = Heisenberg([(1.0, i, (i + 1) % n) for i in range(n)], basis)
    energy, eigenvector = diagonalise(hamiltonian)
    assert np.isclose(energy, -18.06178542)
    basis = SpinBasis(
        make_group(
            [
                Symmetry(list(range(1, n)) + [0], sector=n // 2),
                Symmetry(list(range(n))[::-1], sector=1),
            ]
        ),
        number_spins=n,
        hamming_weight=n // 2,
    )
    hamiltonian = Heisenberg([(1.0, i, (i + 1) % n) for i in range(n)], basis)
    energy, eigenvector = diagonalise(hamiltonian)
    assert np.isclose(energy, -18.06178542)


def test_sparse_diagonalisation():
    import scipy.sparse
    import scipy.sparse.linalg

    n = 10
    basis = SpinBasis([], number_spins=n, hamming_weight=n // 2)
    basis.build()
    hamiltonian = Heisenberg([(1.0, i, (i + 1) % n) for i in range(n)], basis)
    m = hamiltonian.to_csr()
    energy, _ = scipy.sparse.linalg.eigsh(m, k=1)
    assert np.isclose(energy, -18.06178542)


def reference_apply(spins, hamiltonian, psi):
    with torch.no_grad():
        basis = hamiltonian.basis
        states = torch.from_numpy(basis.states.view(np.int64))
        psi = psi(states)
        scale = torch.max(psi[:, 0]).item()
        psi[:, 0] -= scale
        psi = np.exp(psi.numpy().view(np.complex64))
        hamiltonian = hamiltonian.to_csr()
        out = np.empty(len(spins), dtype=np.complex64)
        for i, s in enumerate(spins):
            state = np.zeros(basis.number_states, dtype=np.complex64)
            state[basis.index(s)] = 1.0
            state = hamiltonian @ state
            out[i] = scale + np.log(np.dot(state.conj(), psi))
        return out


def reference_diag(spins, hamiltonian):
    basis = hamiltonian.basis
    hamiltonian = hamiltonian.to_csr()
    out = np.empty(len(spins), dtype=np.complex64)
    for i, s in enumerate(spins):
        state = np.zeros(basis.number_states, dtype=np.complex64)
        state[basis.index(s)] = 1.0
        state = hamiltonian @ state
        out[i] = state[basis.index(s)]
    return out


class Unpack(torch.nn.Module):
    n: Final[int]

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x):
        return unpack(x, self.n)


def test_apply():
    n = 20
    basis = SpinBasis([], number_spins=n, hamming_weight=n // 2)
    basis.build()
    hamiltonian = Heisenberg([(1.0, i, (i + 1) % n) for i in range(n)], basis)
    for i in range(5):
        spins = torch.from_numpy(basis.states.view(np.int64))[
            torch.randperm(basis.number_states)[:500]
        ]
        psi = torch.jit.script(
            torch.nn.Sequential(
                Unpack(n),
                torch.nn.Linear(n, 5),
                torch.nn.Tanh(),
                torch.nn.Linear(5, 2, bias=False),
            )
        )
        expected = reference_apply(spins, hamiltonian, psi)
        spins_512 = torch.cat(
            [spins.unsqueeze(dim=1), torch.zeros(spins.size(0), 7, dtype=torch.int64)],
            dim=1,
        )
        predicted = (
            _C.apply(spins_512, hamiltonian, psi._c._get_method("forward"))
            .numpy()
            .view(np.complex64)
            .squeeze()
        )
        assert np.allclose(predicted, expected)


def test_diag():
    n = 20
    basis = SpinBasis([], number_spins=n, hamming_weight=n // 2)
    basis.build()
    hamiltonian = Heisenberg([(1.0, i, (i + 1) % n) for i in range(n)], basis)
    spins = torch.from_numpy(basis.states.view(np.int64))[
        torch.randperm(basis.number_states)[:1000]
    ]
    expected = reference_diag(spins, hamiltonian)
    spins_512 = torch.cat(
        [spins.unsqueeze(dim=1), torch.zeros(spins.size(0), 7, dtype=torch.int64)],
        dim=1,
    )
    predicted = _C.diag(spins_512, hamiltonian).numpy().view(np.complex64).squeeze()
    assert np.allclose(predicted, expected)
