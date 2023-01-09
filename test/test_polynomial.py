import numpy as np
import torch

from nqs_playground import *
from nqs_playground.core import _get_device
import nqs_playground._C as _C

np.random.seed(32974395)
torch.manual_seed(20245898)
manual_seed(3345724)


@torch.no_grad()
def buffers_to_explicit(spins, coeffs):
    assert spins.dtype == torch.int64
    assert coeffs.dtype == torch.float64
    spins = spins[:, 0].cpu().numpy().view(np.uint64)
    coeffs = coeffs.cpu().numpy().view(np.complex128)
    explicit = dict()
    for s, c in zip(spins, coeffs):
        s = int(s)
        if s in explicit:
            explicit[s] += c
        else:
            explicit[s] = c
    return explicit


def test_polynomial():
    basis = SpinBasis([], 10, 5)
    basis.build()
    hamiltonian = Heisenberg([(1, i, (i + 1) % 10) for i in range(10)], basis)

    dense = hamiltonian.to_csr().todense()
    roots = [10.0, 5.0 + 3j, -2.4 + 0.8j]
    P = np.eye(dense.shape[0])
    for r in roots:
        P = (dense - r * np.eye(dense.shape[0])) @ P

    polynomial = _C.Polynomial(hamiltonian, roots)
    for j in range(basis.number_states):
        explicit = buffers_to_explicit(*polynomial(int(basis.states[j]), 1.0))
        for i in range(basis.number_states):
            expected = P[i, j]
            predicted = explicit.get(basis.states[i], 0.0)
            assert np.isclose(predicted, expected)

@torch.no_grad()
def compute_log_target_state(
    spins: Tensor,
    hamiltonian,
    roots: List[complex],
    state: torch.nn.Module,
    batch_size: int,
    normalizing: bool = False,
) -> Tensor:
    logger.debug("Applying polynomial using batch_size={}...", batch_size)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("invalid batch_size: {}; expected a positive integer".format(batch_size))
    spins = as_spins_tensor(spins, force_width=True)
    original_shape = spins.size()[:-1]
    spins = spins.view(-1, spins.size(-1))
    if isinstance(state, torch.jit.ScriptModule):
        state = state._c._get_method("forward")
    log_target = _C.log_apply_polynomial(spins, hamiltonian, roots, state, batch_size, normalizing)
    return log_target.view(original_shape)

@torch.no_grad()
def module_to_explicit(state: torch.nn.Module, basis):
    device = _get_device(state)
    if device is None:
        device = torch.device("cpu")

    spins = torch.from_numpy(basis.states.view(np.int64)).to(device)
    out = forward_with_batches(log_psi, spins, batch_size=8192)
    scale = torch.max(out.real).item()
    out.real -= scale
    out = torch.exp_(out)
    return out, scale

@torch.no_grad()
def reference_log_apply_polynomial(
    spins: Tensor,
    hamiltonian,
    roots: List[complex],
    state: torch.nn.Module,
    normalizing: bool = False,
):
    basis = hamiltonian.basis
    basis.build()

    v, scale = module_to_explicit(state, basis)
    v = v.cpu().numpy()
    for r in roots:
        v = hamiltonian(v) - r * v
        if normalizing:
            v /= np.linalg.norm(v)

    device = spins.device
    if spins.dim() > 1:
        assert spins.size(1) == 8
        spins = spins[:, 0]
    spins = spins.cpu().numpy().view(np.uint64)
    indices = basis.batched_index(spins)
    v = v[indices]

    
    pass



class SlowPolynomialState:
    def __init__(self, hamiltonian, roots, log_psi):
        self.basis = hamiltonian.basis
        self.basis.build()
        self.state, self.scale = self._make_state(log_psi)

        dense = hamiltonian.to_csr().todense()
        self.polynomial = np.eye(dense.shape[0])
        for r in roots:
            self.polynomial = (dense - r * np.eye(dense.shape[0])) @ self.polynomial

    @torch.no_grad()
    def _make_state(self, log_psi):
        device = _get_device(log_psi)
        spins = torch.from_numpy(self.basis.states.view(np.int64)).to(device)
        out = forward_with_batches(log_psi, spins, 4096)
        scale = torch.max(out[:, 0]).item()
        out[:, 0] -= scale
        out = out.cpu().numpy().view(np.complex64).squeeze().astype(np.complex128)
        out = np.exp(out)
        return out, scale

    def _forward_one(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy().view(np.uint64)
        if isinstance(x, np.ndarray):
            x = x[0]
        x = int(x)

        i = self.basis.index(x)
        vector = np.zeros((self.basis.number_states,), dtype=np.complex128)
        vector[i] = 1.0
        return self.scale + np.log(np.dot(self.polynomial @ vector, self.state))

    def __call__(self, spins):
        device = spins.device
        spins = spins.cpu().numpy().view(np.uint64)
        out = np.empty(spins.shape[0], dtype=np.complex128)
        for i in range(spins.shape[0]):
            out[i] = self._forward_one(spins[i])
        out = out.astype(np.complex64).view(np.float32).reshape(-1, 2)
        return torch.from_numpy(out).to(device)


def test_apply(device):
    basis = SpinBasis([], 10, 5)
    basis.build()
    hamiltonian = Heisenberg(
        [(1, i, (i + 1) % basis.number_spins) for i in range(basis.number_spins)], basis
    )
    log_psi = torch.jit.script(
        torch.nn.Sequential(
            Unpack(basis.number_spins),
            torch.nn.Linear(basis.number_spins, 5),
            torch.nn.Tanh(),
            torch.nn.Linear(5, 2),
        )
    )
    log_psi = log_psi.to(device)
    roots = [1.0, 5.0 + 1.2j, 1.0 - 0.5j]
    spins = torch.from_numpy(basis.states.view(np.int64))
    spins = torch.cat(
        [spins.view(-1, 1), torch.zeros((spins.size(0), 7), dtype=torch.int64)], dim=1
    )
    spins = spins.to(device)

    expected = SlowPolynomialState(hamiltonian, roots, log_psi)(spins)
    for batch_size in [1, 2, 3, 128, 253, 254, 255, 256]:
        predicted = _C.apply_new(
            spins,
            _C.Polynomial(hamiltonian, roots),
            log_psi._c._get_method("forward"),
            batch_size,
            2,
        )
        # if not torch.allclose(predicted, expected, rtol=1e-4, atol=1e-6):
        #     for i in range(spins.size(0)):
        #         if not torch.allclose(predicted[i], expected[i], rtol=1e-4,
        #                 atol=1e-6):
        #             print(i, predicted[i], expected[i])
        assert torch.allclose(predicted, expected, rtol=1e-4, atol=1e-6)


test_polynomial()
test_apply("cpu")
if torch.cuda.is_available():
    test_apply("cuda")
