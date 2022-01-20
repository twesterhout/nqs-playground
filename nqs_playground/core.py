# Copyright Tom Westerhout (c) 2019-2021
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

__all__ = [
    # "with_file_like",
    "setup_random_generators",
    "Unpack",
    "pack",
    "pad_states",
    "split_into_batches",
    "forward_with_batches",
    "as_spins_tensor",
    # "SpinDataset",
    "combine_amplitude_and_sign",
    "combine_amplitude_and_phase",
    "safe_exp",
    "load_ground_state",
    "load_hamiltonian",
    "get_device",
    "get_dtype",
]

import os
import sys
import tempfile
import pathlib
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from loguru import logger

# from typing_extensions import Final
# try:
#     from typing_extensions import Final
# except ImportError:
#     # If you don't have `typing_extensions` installed, you can use a
#     # polyfill from `torch.jit`.
#     from torch.jit import Final
import lattice_symmetries as ls
import unpack_bits

# from . import _C


# Taken from torch; all credit goes to PyTorch developers.
# def with_file_like(f, mode, body):
#     r"""Executes a 'body' function with a file object for 'f', opening
#     it in 'mode' if it is a string filename.
#     """
#     new_fd = False
#     if isinstance(f, str) or isinstance(f, pathlib.Path):
#         new_fd = True
#         f = open(f, mode)
#     try:
#         return body(f)
#     finally:
#         if new_fd:
#             f.close()


def setup_random_generators(*args):
    """Set seeds for NumPy, Torch, and nqs_playground based on args.

    We convert `args` to strings, compute SHA1 of them, and use it as seed.
    """
    import hashlib

    def get_bytes(x):
        if isinstance(x, bytes):
            return x
        if isinstance(x, str):
            return x.encode()
        return get_bytes(str(x))

    # NOTE(twesterhout): A cool hack. We hash together input files and use it as seed. That way
    # using the same input files will produce the same seeds, but we still get different seeds for
    # different systems.
    sha1 = hashlib.sha1()
    for x in args:
        sha1.update(get_bytes(x))
    sha1 = sha1.hexdigest()
    seeds = int(sha1[:8], base=16), int(sha1[8:16], base=16)  # , int(sha1[16:24], base=16)
    logger.info("Seeding NumPy with {}, PyTorch with {} ...", *seeds)
    np.random.seed(seeds[0])
    torch.manual_seed(seeds[1])
    # _C.manual_seed(seeds[2])


class Unpack(torch.nn.Module):
    r"""Unpacks spin configurations represented as bits (`uint64_t` or `ls_bits512`) into a
    2D-tensor of `float32`.
    """
    __constants__ = ["number_spins"]
    number_spins: int

    def __init__(self, number_spins: int):
        super().__init__()
        self.number_spins = number_spins

    def forward(self, x: Tensor) -> Tensor:
        return unpack_bits.unpack(x, self.number_spins)

    def extra_repr(self) -> str:
        return "number_spins={}".format(self.number_spins)


def pack(xs: torch.Tensor) -> torch.Tensor:
    assert xs.dim() == 2
    assert xs.size(1) < 64
    r = torch.zeros(xs.size(0), dtype=torch.int64, device=xs.device)
    for i in range(xs.size(1)):
        r |= (xs[:, i] == 1).long() << i
    # print(xs, torch.ops.tcm.unpack(r, xs.size(1)))
    assert torch.all(torch.ops.tcm.unpack(r, xs.size(1)) == xs)
    return r


def pad_states(states: Tensor) -> Tensor:
    """Pad states with zeros to get a Tensor of bits512 instead of int64."""
    if states.ndim == 0 or states.ndim == 1:
        states = states.view(-1, 1)
    if states.ndim != 2:
        raise ValueError("'states' has invalid shape: {}".format(states.size()))
    if states.size(1) == 1:
        padding = torch.zeros(states.size(0), 7, device=states.device, dtype=torch.int64)
        states = torch.cat([states, padding], dim=1)
    elif spins.size(1) != 8:
        raise ValueError(
            "'states' has invalid shape: {}; size along the last dimension "
            "must be either 1 or 8".format(states.size())
        )
    return states


def as_spins_tensor(spins: Tensor, force_width: bool = True) -> Tensor:
    r"""Convert `spins` to a PyTorch Tensor representing spin configurations."""
    if isinstance(spins, np.ndarray):
        if not spins.dtype == np.uint64:
            raise TypeError("'spins' has invalid datatype: {}; expected uint64".format(spins.dtype))
        spins = torch.from_numpy(spins.view(np.int64))
    if not isinstance(spins, Tensor):
        raise TypeError("'spins' has invalid type: {}; expected a torch.Tensor".format(type(spins)))
    if spins.dtype != torch.int64:
        raise TypeError("'spins' has invalid datatype: {}; expected int64".format(spins.dtype))
    if spins.ndim == 0 or spins.ndim == 1:
        spins = spins.view(-1, 1)
    if force_width:
        if spins.size(-1) == 1:
            padding = spins.new_zeros(spins.size()[:-1] + (7,))
            spins = torch.cat([spins, padding], dim=spins.ndim - 1)
        if spins.size(-1) != 8:
            raise ValueError(
                "'spins' has invalid shape: {}; size along the last dimension "
                "must be 8".format(spins.size())
            )
    if spins.stride(-1) != 1:
        raise ValueError("'spins' must be contiguous along the last dimension")
    return spins


def split_into_batches(xs: Tensor, batch_size: int, device=None):
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("invalid batch_size: {}; expected a positive integer".format(batch_size))

    expanded = False
    if isinstance(xs, (np.ndarray, Tensor)):
        xs = (xs,)
        expanded = True
    else:
        assert isinstance(xs, (tuple, list))
    n = xs[0].shape[0]
    if any(filter(lambda x: x.shape[0] != n, xs)):
        raise ValueError("tensors 'xs' must all have the same batch dimension")
    if n == 0:
        return None

    i = 0
    while i + batch_size <= n:
        chunks = tuple(x[i : i + batch_size] for x in xs)
        if device is not None:
            chunks = tuple(chunk.to(device) for chunk in chunks)
        if expanded:
            chunks = chunks[0]
        yield chunks
        i += batch_size
    if i != n:  # Remaining part
        chunks = tuple(x[i:] for x in xs)
        if device is not None:
            chunks = tuple(chunk.to(device) for chunk in chunks)
        if expanded:
            chunks = chunks[0]
        yield chunks


def forward_with_batches(f, xs, batch_size: int, device=None) -> Tensor:
    r"""Applies ``f`` to all ``xs`` propagating no more than ``batch_size``
    samples at a time. ``xs`` is split into batches along the first dimension
    (i.e. dim=0). ``f`` must return a torch.Tensor.
    """
    if xs.shape[0] == 0:
        raise ValueError("invalid xs: {}; input should not be empty".format(xs))
    out = []
    for chunk in split_into_batches(xs, batch_size, device):
        out.append(f(chunk))
    return torch.cat(out, dim=0)


class SpinDataset(torch.utils.data.IterableDataset):
    r"""Dataset wrapping spin configurations and corresponding values.

    :param spins: either a ``numpy.ndarray`` of ``uint64`` or a
        ``torch.Tensor`` of ``int64`` containing compact spin configurations.
    :param values: a ``torch.Tensor``.
    :param batch_size: batch size.
    :param shuffle: whether to shuffle the samples.
    :param device: device where the batches will be used.
    """

    def __init__(self, spins, values, batch_size, shuffle=False, device=None):
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
            raise TypeError("values must be either a torch.Tensor; got {}".format(type(spins)))

        if self.spins.size(0) != self.values.size(0):
            raise ValueError(
                "spins and values must have the same size along the first "
                "dimension, but spins.shape={} != values.shape={}"
                "".format(spins.size(), values.size())
            )

        if batch_size <= 0:
            raise ValueError(
                "invalid batch_size: {}; expected a positive integer" "".format(batch_size)
            )
        self.batch_size = batch_size

        if device is None:
            device = self.values.device
        elif isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.spins = self.spins.to(self.device)
        self.values = self.values.to(self.device)
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (self.spins.size(0) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.spins.size(0), device=self.device)
            spins = self.spins[indices]
            values = self.values[indices]
        else:
            spins = self.spins
            values = self.values
        return zip(
            torch.split(self.spins, self.batch_size),
            torch.split(self.values, self.batch_size),
        )


def combine_amplitude_and_sign(*modules, use_jit: bool = True) -> torch.nn.Module:
    r"""Combines two torch.nn.Modules representing log amplitudes and signs of the
    wavefunction coefficients into one model.
    """
    (log_amplitude, sign) = modules

    class CombiningState(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_amplitude = log_amplitude
            self.sign = sign

        @torch.no_grad()
        def forward(self, x: Tensor) -> Tensor:
            a = self.log_amplitude(x)
            b = 3.141592653589793 * torch.argmax(self.sign(x), dim=1).to(a.dtype).view([-1, 1])
            return torch.complex(a, b)

    m = CombiningState()
    if use_jit:
        m = torch.jit.script(m)
    return m


def combine_amplitude_and_phase(*modules, use_jit: bool = True) -> torch.nn.Module:
    r"""Combines PyTorch modules representing log amplitude and phase into a
    single module representing the logarithm of the wavefunction.

    :param modules: a tuple of two modules: ``(amplitude, phase)``. Both
        modules receive ``(batch_size, in_features)`` as input shape produce
        ``(batch_size, 1)`` as output shape.
    :param use_jit: if ``True``, the returned module is a
        ``torch.jit.ScriptModule``.
    """

    (_amplitude, phase) = modules

    class LogProbState(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.amplitude = _amplitude

        def forward(self, x: Tensor) -> Tensor:
            return 0.5 * self.amplitude.log_prob(x).view(-1, 1)

    if hasattr(_amplitude, "log_prob"):
        amplitude = LogProbState()
    else:
        amplitude = _amplitude

    class CombiningState(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.amplitude = amplitude
            self.phase = phase

        def forward(self, x: Tensor) -> Tensor:
            a = self.amplitude(x)
            b = self.phase(x)
            # print(a.size(), b.size())
            # print(torch.complex(a, b).size())
            return torch.complex(a, b)

    m = CombiningState()
    if use_jit:
        m = torch.jit.script(m)
    return m


# def combine_amplitude_and_sign_classifier(
#     *modules, number_spins: int, use_jit: bool = True
# ) -> torch.nn.Module:
#     r"""Combines two torch.nn.Modules representing amplitudes and signs of the
#     wavefunction coefficients into one model.
#     """
#
#     class CombiningState(torch.nn.Module):
#         def __init__(self, amplitude, phase, number_spins):
#             super().__init__()
#             self.amplitude = amplitude
#             self.phase = phase
#             self.number_spins = number_spins
#
#         def forward(self, x):
#             x = torch.ops.tcm.unpack(x, self.number_spins)
#             log_phi = self.amplitude(x)
#             p = 2 * self.phase(x) - 1
#
#             a = log_phi + torch.log(torch.abs(p))
#             b = 3.141592653589793 * (1.0 - torch.sign(p)) / 2.0
#             return torch.cat([a, b], dim=1)
#
#     m = CombiningState(*modules, number_spins)
#     if use_jit:
#         m = torch.jit.script(m)
#     return m


def load_ground_state(
    filename: str, representatives: bool = True
) -> Tuple[Tensor, float, Optional[np.ndarray]]:
    """Load ground state from 'filename'.

    Parameters
    ----------
    filename: str
        Filename from which to load the ground state. Currently, only HDF5
        files are supported. Support for other file formats (such as those
        generated by `numpy.save` or `pickle.dump` may be added in the future).
    representatives: bool
        Whether to load basis representatives as well.
    """
    _, extension = os.path.splitext(filename)
    if extension == ".hdf5" or extension == ".h5":
        import h5py

        logger.info("Loading ground state from '{}'...", filename)
        with h5py.File(filename, "r") as f:
            ground_state = f["/hamiltonian/eigenvectors"][:]
            ground_state = torch.from_numpy(ground_state).squeeze()
            if ground_state.dim() > 1:
                logger.info(
                    "/hamiltonian/eigenvectors has shape {}, "
                    "slicing along the 0th dimension ...",
                    ground_state.shape,
                )
                ground_state = ground_state[0]
            energy = f["/hamiltonian/eigenvalues"][0]
            basis_representatives = None
            if representatives:
                basis_representatives = f["/basis/representatives"][:]
            return ground_state, energy, basis_representatives
    raise ValueError(
        "Do not know how to read the ground state from files with extension '{}'..."
        "".format(extension)
    )


def load_hamiltonian(
    filename: str,
    keep_hamming_weight: bool = True,
    keep_spin_inversion: bool = True,
    keep_symmetries: bool = True,
):
    _, extension = os.path.splitext(filename)
    if extension != ".yaml" and extension != ".yml":
        raise ValueError(
            "Do not know how to read the basis from files with extension '{}'..."
            " Try specifying a path to a YAML file instead.".format(extension)
        )

    import yaml

    logger.info("Loading basis from '{}'...", filename)
    with open(filename, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    basis = ls.SpinBasis.load_from_yaml(config["basis"])
    if not (keep_hamming_weight and keep_spin_inversion and keep_symmetries):
        logger.error("keep_* arguments are not implemented yet...")
    logger.info("Loading Hamiltonian from '{}'...", filename)
    return ls.Operator.load_from_yaml(config["hamiltonian"], basis)


@torch.no_grad()
def compute_overlap(combined_state, basis, ground_state: Tensor, batch_size: int) -> float:
    if ground_state is None:
        logger.debug("Skipping overlap computation, because ground state not provided...")
        return None
    if isinstance(ground_state, np.ndarray):
        ground_state = torch.from_numpy(ground_state)
    ground_state.squeeze_()
    if ground_state.dim() != 1:
        raise ValueError(
            "'ground_state' has wrong shape: {}; expected a vector".format(ground_state.size())
        )
    device = get_device(combined_state)
    spins = torch.from_numpy(basis.states.view(np.int64)).view(-1, 1)
    state = forward_with_batches(lambda x: combined_state(x.to(device)).cpu(), spins, batch_size)
    if state.size() != (spins.size(0), 1):
        raise ValueError(
            "'combined_state' returned a Tensor of wrong shape: {}; expected {}"
            "".format(state.size(), (spins.size(0), 1))
        )
    if not state.dtype.is_complex:
        raise ValueError(
            "'combined_state' returned a Tensor of wrong dtype: {}; 'combined_state' predicts "
            "log(ψ(σ)) and is complex because ψ(σ) can be negative".format(state.dtype)
        )
    state.squeeze_(dim=1)

    state.real -= torch.max(state.real)
    state.exp_()
    overlap = torch.abs(torch.dot(state.conj(), ground_state))
    overlap /= torch.linalg.norm(state)
    overlap /= torch.linalg.norm(ground_state)
    return overlap.item()


@torch.jit.script
def safe_exp(x: Tensor, normalise: bool = True) -> Tensor:
    r"""Calculate ``exp(x)`` avoiding overflows. Result is not equal to
    ``exp(x)``, but rather proportional to it. If ``normalise==True``, then
    this function makes sure that output tensor elements sum up to 1.
    """
    x = x - torch.max(x)
    torch.exp_(x)
    if normalise:
        x /= torch.sum(x)
    return x


def get_device(obj) -> Optional[torch.device]:
    x = _get_a_var(obj)
    return x.device if x is not None else None


def get_dtype(obj) -> Optional[torch.dtype]:
    x = _get_a_var(obj)
    return x.device if x is not None else None


def _get_a_var(obj):
    if isinstance(obj, Tensor):
        return obj
    if isinstance(obj, torch.nn.Module):
        for result in obj.parameters(recurse=True):
            if isinstance(result, Tensor):
                return result
        for result in obj.buffers(recurse=True):
            if isinstance(result, Tensor):
                return result
    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, Tensor):
                return result
    return None
