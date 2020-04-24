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

__all__ = [
    "with_file_like",
    "Unpack",
    "forward_with_batches",
    "SpinDataset",
    "combine_amplitude_and_sign",
    "combine_amplitude_and_sign_classifier",
    "combine_amplitude_and_phase",
    "load_model",
    "load_device",
    "load_optimiser",
    "_get_device"
]

import os
import sys
import tempfile
import pathlib
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

try:
    from typing_extensions import Final
except ImportError:
    # If you don't have `typing_extensions` installed, you can use a
    # polyfill from `torch.jit`.
    from torch.jit import Final

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


class Unpack(torch.nn.Module):
    n: Final[int]

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x):
        return torch.ops.tcm.unpack(x, self.n)


def forward_with_batches(f, xs, batch_size: int) -> Tensor:
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
            raise TypeError(
                "values must be either a torch.Tensor; got {}".format(type(spins))
            )

        if self.spins.size(0) != self.values.size(0):
            raise ValueError(
                "spins and values must have the same size along the first "
                "dimension, but spins.shape={} != values.shape={}"
                "".format(spins.size(), values.size())
            )

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
        self.spins = self.spins.to(self.device)
        self.values = self.values.to(self.device)
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (self.spins.size(0) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        print("__iter__: {}, {}".format(self.spins.size(), self.values.size()))
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


def _import_network(filename: str):
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


def combine_amplitude_and_phase(*modules, use_jit: bool = True) -> torch.nn.Module:
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

    class CombiningState(torch.nn.Module):
        def __init__(self, amplitude: torch.nn.Module, phase: torch.nn.Module):
            super().__init__()
            self.amplitude = amplitude
            self.phase = phase

        def forward(self, x: Tensor) -> Tensor:
            a = self.amplitude(x)
            b = self.phase(x)
            return torch.cat([a, b], dim=1)

    m = CombiningState(*modules)
    if use_jit:
        m = torch.jit.script(m)
    return m


def combine_amplitude_and_sign_classifier(
        *modules, number_spins: int,
    use_jit: bool = True
) -> torch.nn.Module:
    r"""Combines two torch.nn.Modules representing amplitudes and signs of the
    wavefunction coefficients into one model.
    """
    
    class CombiningState(torch.nn.Module):
        def __init__(self, amplitude, phase, number_spins):
            super().__init__()
            self.amplitude = amplitude
            self.phase = phase
            self.number_spins = number_spins

        def forward(self, x):
            x = torch.ops.tcm.unpack(x, self.number_spins)
            log_phi = self.amplitude(x)
            p = 2 * self.phase(x) - 1

            a = log_phi + torch.log(torch.abs(p))
            b = 3.141592653589793 * (1. - torch.sign(p)) / 2.
            return torch.cat([a, b], dim=1)

    m = CombiningState(*modules, number_spins)
    if use_jit:
        m = torch.jit.script(m)
    return m


def load_model(model, number_spins=None) -> torch.jit.ScriptModule:
    r"""Loads a model for a simulation. If ``model`` is already a
    ``torch.jit.ScriptModule`` nothing is done. If ``model`` is just a
    ``torch.nn.Module`` we compile it to TorchScript. Otherwise ``model`` is
    assumed to be a path to TorchScript archive or a Python module.
    """
    # model is already a ScriptModule, nothing to be done.
    if isinstance(model, torch.jit.ScriptModule):
        return model
    # model is a Module, so we just compile it.
    elif isinstance(model, torch.nn.Module):
        return torch.jit.script(model)
    # model is a string
    # If model is a Python script, we import the Net class from it,
    # construct the model, and JIT-compile it. Otherwise, we assume
    # that the user wants to continue the simulation and has provided a
    # path to serialised TorchScript module. We simply load it.
    _, extension = os.path.splitext(os.path.basename(model))
    if extension == ".py":
        if number_spins is None:
            raise ValueError(
                "cannot construct the network imported from {}, because "
                "the number of spins is not given".format(model)
            )
        return torch.jit.script(_import_network(name)(number_spins))
    return torch.jit.load(model)


def load_optimiser(optimiser, parameters) -> torch.optim.Optimizer:
    if isinstance(optimiser, str):
        # NOTE: Yes, this is unsafe, but terribly convenient!
        optimiser = eval(optimiser)
    if not isinstance(optimiser, torch.optim.Optimizer):
        # assume that optimiser is a lambda
        optimiser = optimiser(parameters)
    return optimiser


def load_device(config) -> torch.device:
    r"""Determines which device to use for the simulation. We support
    specifying the device as either 'torch.device' (for cases when you e.g.
    have multiple GPUs and want to use some particular one) or as a string with
    device type (i.e. either "gpu" or "cpu").
    """
    device = config.device
    if isinstance(device, (str, bytes)):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        raise TypeError(
            "config.device has wrong type: {}; must be either a "
            "'torch.device' or a 'str'".format(type(device))
        )
    return device

def _get_device(obj) -> Optional[torch.device]:
    return __get_a_var(obj).device

def __get_a_var(obj):
    if isinstance(obj, Tensor):
        return obj
    if isinstance(obj, torch.nn.Module):
        for result in obj.parameters():
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
