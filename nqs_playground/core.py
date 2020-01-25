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

import os
import sys
import tempfile
import pathlib
from typing import Optional, Tuple
from typing_extensions import Final

import numpy as np
import torch

from . import _C


# Taken from torch; all credit goes to PyTorch developers.
def with_file_like(f, mode, body):
    r"""Executes a 'body' function with a file object for 'f', opening
    it in 'mode' if it is a string filename.
    """
    new_fd = False
    if isinstance(f, str) or isinstance(f, pathlib.Path):
        new_fd = True
        f = open(f, mode)
    try:
        return body(f)
    finally:
        if new_fd:
            f.close()


class SetToTrain(object):
    """
    Temporary sets the module (i.e. a ``torch.nn.Module``) into training mode.

    We rely on the following: if ``m`` is a module, then

      * ``m.training`` returns whether ``m`` is currently in the training mode;
      * ``m.train()`` puts ``m`` into training mode;
      * ``m.eval()`` puts ``m`` into inference mode.

    This class is meant to be used in the ``with`` construct:

    .. code:: python

       with _Train(m):
           ...
    """

    def __init__(self, module: torch.nn.Module):
        """
        :param module:
            a ``torch.nn.Module`` or an intance of another class with the same
            interface.
        """
        self._module = module
        # Only need to update the mode if not already training
        self._update = module.training == False

    def __enter__(self):
        if self._update:
            self._module.train()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._update:
            self._module.eval()


class SetToEval(object):
    """
    Temporary sets the network (i.e. a ``torch.nn.Module``) into inference mode.

    We rely on the following: if ``m`` is a network, then

      * ``m.training`` returns whether ``m`` is currently in the training mode;
      * ``m.train()`` puts ``m`` into training mode;
      * ``m.eval()`` puts ``m`` into inference mode.

    This class is meant to be used in the ``with`` construct:

    .. code:: python

       with _Train(m):
           ...
    """

    def __init__(self, module: torch.nn.Module):
        """
        :param module:
            a ``torch.nn.Module`` or an intance of another class with the same
            interface.
        """
        self._module = module
        # Only need to update the mode if currently training
        self._update = module.training == True

    def __enter__(self):
        if self._update:
            self._module.eval()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._update:
            self._module.train()


def forward_with_batches(f, xs, batch_size: int) -> torch.Tensor:
    r"""Applies ``f`` to all ``xs`` propagating no more than ``batch_size``
    samples at a time. ``xs`` is split into batches along the first dimension
    (i.e. dim=0). ``f`` must return a torch.Tensor.
    """
    n = xs.shape[0]
    if n == 0:
        raise ValueError("invalid xs: {}; input should not be empty".format(xs))
    if batch_size <= 0:
        raise ValueError(
            "invalid batch_size: {}; expected a positive integer".format(batch_size)
        )
    i = 0
    out = []
    while i + batch_size <= n:
        out.append(f(xs[i : i + batch_size]))
        i += batch_size
    if i != n:  # Remaining part
        out.append(f(xs[i:]))
    return torch.cat(out, dim=0)




class SpinDataset(torch.utils.data.IterableDataset):
    r"""Dataset wrapping spin configurations and corresponding values.

    :param spins: either a ``numpy.ndarray`` of ``uint64`` or a
        ``torch.Tensor`` of ``int64`` containing compact spin configurations.
    :param values: a ``torch.Tensor``.
    :param number_spins: number of spins in the system. This is used to unpack
        spin configurations.
    :param batch_size: batch size.
    :param shuffle: whether to shuffle the samples.
    :param device: device where the batches will be used.
    """

    def __init__(
        self, spins, values, number_spins, batch_size, shuffle=False, device=None
    ):
        if isinstance(spins, np.ndarray):
            if spins.dtype != np.uint64:
                raise TypeError(
                    "spins must be a numpy.ndarray of uint64; got numpy.ndarray "
                    "of {}".format(spins.dtype.name)
                )
            # Use int64 because PyTorch doesn't support uint64
            self.spins = torch.from_numpy(spins.view(np.int64))
        elif isinstance(spins, torch.Tensor):
            if spins.dtype != torch.int64:
                raise TypeError(
                    "spins must be a torch.Tensor of int64; got torch.Tensor "
                    "of {}".format(spins.dtype)
                )
            self.spins = spins
        else:
            raise TypeError(
                "spins must be either a numpy.ndarray of uint64 or a "
                "torch.Tensor of int64; got {}".format(type(spins))
            )

        if isinstance(values, torch.Tensor):
            self.values = values
        else:
            raise TypeError(
                "values must be either a torch.Tensor; got {}".format(type(spins))
            )

        if self.spins.size(0) != self.values.size(0):
            raise ValueError(
                "spins and values must have the same size along the first "
                "dimension, but spins.shape={} != values.shape={}"
                "".format(spins.size(), values.size())
            )

        if number_spins <= 0:
            raise ValueError(
                "invalid number_spins: {}; expected a positive integer"
                "".format(number_spins)
            )
        self.number_spins = number_spins

        if batch_size <= 0:
            raise ValueError(
                "invalid batch_size: {}; expected a positive integer"
                "".format(batch_size)
            )
        self.batch_size = batch_size

        if device is None:
            device = self.values.device
        elif isinstance(device, str):
            device = torch.device(device)

        self.device = device
        if self.device.type == "cuda":
            self.values = self.values.pin_memory()

        self.dataset = _C.v2.ChunkLoader(
            _C.v2.SpinDataset(self.spins, self.values, self.number_spins),
            chunk_size=self.batch_size * max(81920 // self.batch_size, 1),
            shuffle=shuffle,
            device=self.device,
            pin_memory=self.device.type == "cuda",
        )

    def __len__(self) -> int:
        return self.spins.shape[0]

    def __iter__(self):
        for xs, ys in self.dataset:
            for mini_batch in zip(
                torch.split(xs, self.batch_size), torch.split(ys, self.batch_size)
            ):
                yield mini_batch


def random_split(dataset, k, replacement=False, weights=None):
    r"""Randomly splits dataset into two parts.

    :param dataset: a tuple of NumPy arrays or Torch tensors.
    :param k: either a ``float`` or an ``int`` specifying the size of the first
        part. If ``k`` is a ``float``, then it must lie in ``[0, 1]`` and is
        understood as a fraction of the whole dataset. If ``k`` is an ``int``,
        then it specifies the number of elements.
    :param weights: specifies how elements for the first part are chosen. If
        ``None``, uniform sampling is used. Otherwise elements are sampled from
        a multinomial distribution with probabilities proportional to ``weights``.
    """
    if not all(
        (isinstance(x, np.ndarray) or isinstance(x, torch.Tensor) for x in dataset)
    ):
        raise ValueError("dataset should be a tuple of NumPy arrays or Torch tensors")
    n = dataset[0].shape[0]
    if not all((x.shape[0] == n for x in dataset)):
        raise ValueError(
            "all elements of dataset should have the same size along the first dimension"
        )

    if isinstance(k, float):
        if k < 0.0 or k > 1.0:
            raise ValueError("k should be in [0, 1]; got k={}".format(k))
        k = round(k * n)
    elif isinstance(k, int):
        if k < 0 or k > n:
            raise ValueError("k should be in [0, {}]; got k={}".format(n, k))
    else:
        raise ValueError("k must be either an int or a float; got {}".format(type(k)))

    with torch.no_grad():
        if weights is None:
            # Uniform sampling
            if replacement:
                indices = torch.randint(n, size=k)
            else:
                indices = torch.randperm(n)[:k]
        else:
            # Sampling with specified weights
            if len(weights) <= (1 << 24):
                indices = torch.multinomial(
                    weights, num_samples=k, replacement=replacement
                )
            else:
                weights = weights.to(torch.float64)
                weights /= torch.sum(weights)
                indices = np.random.choice(
                    len(weights), size=k, replace=replacement, p=weights
                )

        remaining_indices = np.setdiff1d(
            np.arange(n), indices, assume_unique=not replacement
        )
        return [
            tuple(x[indices] for x in dataset),
            tuple(x[remaining_indices] for x in dataset),
        ]


def import_network(filename: str):
    r"""Loads ``Net`` class defined in Python source file ``filename``."""
    import importlib

    module_name, extension = os.path.splitext(os.path.basename(filename))
    module_dir = os.path.dirname(filename)
    if extension != ".py":
        raise ValueError(
            "Could not import the network from {!r}: ".format(filename)
            + "not a Python source file."
        )
    if not os.path.exists(filename):
        raise ValueError(
            "Could not import the network from {!r}: ".format(filename)
            + "no such file or directory"
        )
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    sys.path.pop(0)
    return module.Net


def combine_amplitude_and_sign(
    *modules, apply_log: bool = False, out_dim: int = 1, use_jit: bool = True
) -> torch.nn.Module:
    r"""Combines two torch.nn.Modules representing amplitudes and signs of the
    wavefunction coefficients into one model.
    """
    if out_dim != 1 and out_dim != 2:
        raise ValueError("invalid out_dim: {}; expected either 1 or 2".format(out_dim))
    if out_dim == 1 and apply_log:
        raise ValueError("apply_log is incompatible with out_dim=1")

    class CombiningState(torch.nn.Module):
        __constants__ = ["apply_log", "out_dim"]

        def __init__(self, amplitude, phase):
            super().__init__()
            self.apply_log = apply_log
            self.out_dim = out_dim
            self.amplitude = amplitude
            self.phase = phase

        def forward(self, x):
            a = torch.log(self.amplitude(x)) if self.apply_log else self.amplitude(x)
            if self.out_dim == 1:
                b = (
                    (1 - 2 * torch.argmax(self.phase(x), dim=1))
                    .to(torch.float32)
                    .view([-1, 1])
                )
                a *= b
                return a
            else:
                b = 3.141592653589793 * torch.argmax(self.phase(x), dim=1).to(
                    torch.float32
                ).view([-1, 1])
                return torch.cat([a, b], dim=1)

    m = CombiningState(*modules)
    if use_jit:
        m = torch.jit.script(m)
    return m


def combine_amplitude_and_phase(
        *modules, number_spins: Optional[int], apply_log: bool = False, use_jit: bool = True
) -> torch.nn.Module:
    r"""Combines PyTorch modules representing amplitude (or logarithm thereof)
    and phase into a single module representing the logarithm of the
    wavefunction.

    :param modules: a tuple of two modules: ``(amplitude, phase)``. Both
        modules have ``(batch_size, in_features)`` as input shape and
        ``(batch_size, 1)`` as output shape.
    :param apply_log: if ``True``, logarithm is applied to the output of
        ``amplitude`` module.
    :param use_jit: if ``True``, the returned module is a
        ``torch.jit.ScriptModule``.
    """
    if number_spins is None:
        number_spins = -1

    class CombiningState(torch.nn.Module):
        apply_log: Final[bool]
        number_spins: Final[int]

        def __init__(self, amplitude: torch.nn.Module, phase: torch.nn.Module):
            super().__init__()
            self.apply_log = apply_log
            self.number_spins = number_spins
            self.amplitude = amplitude
            self.phase = phase

        def forward(self, x: torch.Tensor):
            if self.number_spins > 0:
                x = torch.ops.tcm.unpack(x, self.number_spins)
            a = self.amplitude(x)
            if self.apply_log:
                a = torch.log_(a)
            b = self.phase(x)
            return torch.cat([a, b], dim=1)

    m = CombiningState(*modules)
    if use_jit:
        m = torch.jit.script(m)
    return m


# def _forward_with_batches(state, input, batch_size):
#     n = input.shape[0]
#     if n == 0:
#         raise ValueError("input should not be empty")
#     system_size = len(_C.unsafe_get(input, 0))
#     i = 0
#     out = []
#     while i + batch_size <= n:
#         out.append(state(_C.unpack(input[i : i + batch_size])))
#         i += batch_size
#     if i != n:  # Remaining part
#         out.append(state(_C.unpack(input[i:]).view(-1, system_size)))
#     # r = torch.cat(out, dim=0)
#     # assert torch.all(r == state(_C.unpack(input)))
#     # return r
#     return torch.cat(out, dim=0)
# 
# 
# def local_energy(
#     state: torch.jit.ScriptModule,
#     hamiltonian: _C.v2.Heisenberg,
#     spins: np.ndarray,
#     log_values: Optional[np.ndarray] = None,
#     batch_size: int = 128,
# ) -> np.ndarray:
#     r"""Computes local estimators ⟨σ|H|ψ⟩/⟨σ|ψ⟩ for all σ.
# 
#     :param state: wavefunction ``ψ``. ``state`` should be a function
#         mapping ``R^{batch_size x in_features}`` to ``R^{batch_size x 2}``.
#         Columns of the output are interpreted as real and imaginary parts of
#         ``log(⟨σ|ψ⟩)``.
#     :param hamiltonian: Hamiltonian ``H``.
#     :param spins: spin configurations ``σ``. Should be a non-empty NumPy array
#         of compact spin configurations.
#     :param log_values: pre-computed ``log(⟨σ|ψ⟩)``. Should be a NumPy array of
#         ``complex64``.
#     :param batch_size: batch size to use for forward propagation through
#         ``state``.
# 
#     :return: local energies ⟨σ|H|ψ⟩/⟨σ|ψ⟩ as a NumPy array of ``complex64``.
#     """
#     assert isinstance(state, torch.jit.ScriptModule)
#     assert isinstance(hamiltonian, _C.v2.Heisenberg)
#     if len(spins) == 0:
#         return numpy.array([], dtype=np.complex64)
#     with torch.no_grad(), torch.jit.optimized_execution(True):
#         device = next(state.parameters()).device
#         if log_values is None:
#             number_spins = hamiltonian.basis.number_spins
#             log_values = forward_with_batches(
#                 lambda x: state(_C.v2.unpack(x, number_spins).to(device)).cpu(),
#                 spins,
#                 batch_size,
#             )
#             if log_values.dim() != 2:
#                 raise ValueError(
#                     "state should return the logarithm of the wavefunction, but"
#                     "output tensor has dimension {}; did you by accident forget"
#                     "to combine amplitude and phase networks?".format(log_values.dim())
#                 )
#             log_values = log_values.numpy().view(np.complex64)
# 
#         log_H_values = (
#             forward_with_batches(
#                 _C.v2.PolynomialState(
#                     _C.v2.Polynomial(hamiltonian, [0.0]),
#                     state._c._get_method("forward"),
#                     batch_size,
#                     device,
#                 ),
#                 spins,
#                 batch_size=batch_size // 4,
#             )
#             .numpy()
#             .view(np.complex64)
#         )
#         return np.exp(log_H_values - log_values).squeeze(axis=1)


# def make_monte_carlo_options(config, number_spins: int) -> _C._Options:
#     if number_spins <= 0:
#         raise ValueError(
#             "invalid number spins: {}; expected a positive integer".format(number_spins)
#         )
#     sweep_size = config.sweep_size if config.sweep_size is not None else number_spins
#     number_discarded = (
#         config.number_discarded
#         if config.number_discarded is not None
#         else config.number_samples // 10
#     )
#     magnetisation = (
#         config.magnetisation if config.magnetisation is not None else number_spins % 2
#     )
#     return _C._Options(
#         number_spins=number_spins,
#         magnetisation=magnetisation,
#         number_chains=config.number_chains,
#         number_samples=config.number_samples,
#         sweep_size=sweep_size,
#         number_discarded=number_discarded,
#     )


# def sample_exact(
#     state: torch.jit.ScriptModule, options: _C._Options, batch_size: int = 256
# ) -> Tuple[np.ndarray, torch.Tensor]:
#     spins = _C.all_spins(options.number_spins, options.magnetisation)
#     num_samples = options.number_samples * options.number_chains
#     with torch.no_grad(), torch.jit.optimized_execution(True):
#         values = _forward_with_batches(state, spins, batch_size)
#         weights = _log_amplitudes_to_probabilities(values).squeeze(dim=1)
#         indices = torch.multinomial(weights, num_samples=num_samples, replacement=True)
#         return spins[indices.numpy()], values[indices]
