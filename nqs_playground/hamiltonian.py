#!/usr/bin/env python3

# Copyright Tom Westerhout (c) 2019
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of Tom Westerhout nor the names of other
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pathlib
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor

from . import _C
from ._C import Heisenberg as _Heisenberg
from ._C import SpinBasis
from .core import forward_with_batches

__all__ = ["Heisenberg", "read_hamiltonian", "diagonalise", "local_values", "local_values_slow"]


def Heisenberg(specs: List[Tuple[complex, int, int]], basis) -> _Heisenberg:
    smallest = min(map(lambda t: min(t[1:]), specs))
    largest = max(map(lambda t: max(t[1:]), specs))
    if smallest != 0:
        raise ValueError(
            "Invalid graph: counting from 0, but the minimal index "
            "present is {}.".format(smallest)
        )
    number_spins = largest + 1
    if basis is None:
        basis = SpinBasis([], number_spins, hamming_weight=number_spins // 2)
    elif number_spins > basis.number_spins:
        raise ValueError(
            "Invalid graph: there are {} spins in the system, but the "
            "largest index is {}".format(basis.number_spins, largest)
        )
    return _Heisenberg(specs, basis)


def read_hamiltonian(stream, basis) -> _Heisenberg:
    r"""Reads the Hamiltonian from ``stream``. ``stream`` could be either a
    file-like object or a ``str`` file name.
    """

    def _read_from_txt(stream, basis):
        specs = []
        for (coupling, edges) in map(
            lambda x: x.strip().split(maxsplit=1),
            filter(lambda x: not x.startswith(b"#") and len(x.strip()) > 0, stream),
        ):
            coupling = float(coupling)
            # TODO: Parse the edges properly, it's not that difficult...
            edges = eval(edges)
            for i, j in edges:
                specs.append((coupling, i, j))
        return Heisenberg(specs, basis)

    new_fd = False
    if isinstance(stream, str) or isinstance(stream, pathlib.Path):
        new_fd = True
        stream = open(stream, "r")
    try:
        return _read_from_txt(stream, basis)
    finally:
        if new_fd:
            stream.close()


def diagonalise(hamiltonian: _Heisenberg, k: int = 1, dtype=None):
    r"""Diagonalises the hamiltonian.

    :param hamiltonian: Heisenberg Hamiltonian to diagonalise.
    :param k: number eigenstates to calculate.
    :param dtype: which data type to use.
    """
    import numpy as np
    import scipy.sparse.linalg

    hamiltonian.basis.build()
    n = hamiltonian.basis.number_states
    if dtype is not None:
        if dtype not in {np.float32, np.float64, np.complex64, np.complex128}:
            raise ValueError(
                "invalid dtype: {}; expected float32, float64, complex64 or complex128"
                "".format(dtype)
            )
        if not hamiltonian.is_real and dtype in {np.float32, np.float64}:
            raise ValueError(
                "invalid dtype: {}; Hamiltonian is complex -- expected either complex64 "
                "or complex128".format(dtype)
            )
    else:
        dtype = np.float64 if hamiltonian.is_real else np.complex128

    op = scipy.sparse.linalg.LinearOperator(
        shape=(n, n), matvec=hamiltonian, dtype=dtype
    )
    return scipy.sparse.linalg.eigsh(op, k=k, which="SA")


def local_values(
    spins,
    hamiltonian: _Heisenberg,
    state: torch.jit.ScriptModule,
    log_psi: Optional[Tensor] = None,
    batch_size: int = 2048,
) -> np.ndarray:
    r"""Computes local values ``⟨s|H|ψ⟩/⟨s|ψ⟩`` for all ``s ∈ spins``.

    :param spins: Spin configurations ``{s}``. Must be either a
        ``numpy.ndarray`` of ``uint64`` or a ``torch.Tensor`` of ``int64``.
    :param hamiltonian: Heisenberg Hamiltonian.
    :param state: Quantum state ``ψ`` represented by a TorchScript module,
        which predicts ``log(ψ(s))``.
    :param log_psi: Pre-computed ``log(ψ(s))`` for all ``s`` in ``spins``.
    :param batch_size: Batch size to use internally for forward propagationn
        through ``state``.
    """
    if isinstance(spins, np.ndarray):
        if spins.dtype != np.uint64:
            raise TypeError(
                "spins must be either a numpy.ndarray of uint64 or a torch.Tensor "
                "of int64; got a numpy.ndarray of {}".format(spins.dtype)
            )
        spins = torch.from_numpy(spins.view(np.int64))
    with torch.no_grad():
        # Computes log(⟨s|H|ψ⟩) for all s.
        # TODO: add support for batch size
        # Computes log(⟨s|ψ⟩) for all s.
        if log_psi is None:
            log_psi = forward_with_batches(state, spins, batch_size).to(device='cpu', non_blocking=True)
        log_h_psi = _C.apply(spins, hamiltonian, state._c._get_method("forward"))
        log_h_psi -= log_psi
        log_h_psi = log_h_psi.numpy().view(np.complex64)
        return np.exp(log_h_psi, out=log_h_psi)


def local_values_slow(
    spins,
    hamiltonian: _Heisenberg,
    state: torch.jit.ScriptModule,
    log_psi: Optional[Tensor] = None,
    batch_size: int = 2048,
) -> np.ndarray:
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

    with torch.no_grad():
        # Computes log(⟨s|H|ψ⟩) for all s.
        # TODO: add support for batch size
        log_h_psi = SlowPolynomialState(hamiltonian, [0.0], state)(spins)
        # Computes log(⟨s|ψ⟩) for all s.
        log_psi = forward_with_batches(state, spins, batch_size)
        log_h_psi -= log_psi
        log_h_psi = log_h_psi.numpy().view(np.complex64)
        return np.exp(log_h_psi, out=log_h_psi)
