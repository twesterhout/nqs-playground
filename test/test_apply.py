import numpy as np
import torch
import lattice_symmetries as ls

try:
    from nqs_playground import _C
except ImportError:
    # For local development when we only compile the C++ extension, but don't
    # actually install the package using pip
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    from nqs_playground import _C


def _array_to_int(xs) -> int:
    if len(xs) == 0:
        return 0
    n = int(xs[-1])
    for i in reversed(range(0, len(xs) - 1)):
        n <<= 64
        n += int(xs[i])
    return n


def _reference_log_apply_one(spin, operator, log_psi, device):
    spins, coeffs = operator.apply(spin)
    spins = torch.from_numpy(spins.view(np.int64)).to(device)
    coeffs = torch.from_numpy(coeffs).to(device)
    output = log_psi(spins).to(coeffs.dtype)
    if output.dim() > 1:
        output.squeeze_(dim=1)
    scale = torch.max(output.real)
    output.real -= scale
    torch.exp_(output)
    return scale + torch.log(torch.dot(coeffs, output))


def reference_log_apply(spins, operator, log_psi, batch_size=None):
    device = spins.device
    result = torch.empty(spins.size(0), dtype=torch.complex128)
    for (i, spin) in enumerate(spins):
        result[i] = _reference_log_apply_one(_array_to_int(spin), operator, log_psi, device)
    return result


def batched_index(spins, basis):
    assert basis.number_spins <= 64
    device = None
    if isinstance(spins, torch.Tensor):
        device = spins.device
        spins = spins.detach().cpu().numpy()
    if spins.dtype == np.int64:
        spins = spins.view(np.uint64)
    if spins.ndim > 1:
        spins = spins[:, 0]
    r = np.empty(spins.shape[0], dtype=np.uint64)
    for i in range(spins.shape[0]):
        r[i] = basis.index(spins[i])
    if device is not None:
        r = torch.from_numpy(r.view(np.int64)).to(device)
    return r


def coefficients_to_log_amplitude(coeffs, basis):
    if not isinstance(coeffs, torch.Tensor):
        coeffs = torch.from_numpy(coeffs)
    coeffs = torch.log(coeffs.detach().to(torch.complex128))

    def log_psi(spins):
        indices = batched_index(spins, basis)
        return coeffs[indices]

    return log_psi


def test_apply():
    basis = ls.SpinBasis(ls.Group([]), number_spins=10, hamming_weight=5)
    basis.build()
    matrix = np.array([[1, 0, 0, 0], [0, -1, 2, 0], [0, 2, -1, 0], [0, 0, 0, 1]])
    edges = [(i, (i + 1) % basis.number_spins) for i in range(basis.number_spins)]
    op = ls.Operator(basis, [ls.Interaction(matrix, edges)])
    states = torch.from_numpy(basis.states.view(np.int64))
    coeffs = torch.rand(states.size(0), dtype=torch.float64) - 0.5
    log_psi = coefficients_to_log_amplitude(coeffs, basis)
    states = torch.cat(
        [states.view(-1, 1), torch.zeros((states.size(0), 7), dtype=torch.int64)], dim=1
    )
    predicted = _C.log_apply(states, op, log_psi, 1)
    expected = reference_log_apply(states, op, log_psi, 1)
    assert torch.allclose(predicted, expected)


test_apply()
