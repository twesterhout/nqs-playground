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
from typing import Optional, Tuple

import numpy as np
import torch

# NOTE(twesterhout): Yes, it's not nice to depend on internal functions, but
# it's so tiring to reimplement _with_file_like every time...
from torch.serialization import _with_file_like as with_file_like

from . import _C_nqs as _C


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


def SetNumThreads(object):
    """
    Temporary changes the number of threads used by PyTorch.

    This is useful when we, for example, want to run multiple different neural
    networks in parallel rather than use multiple threads within a single
    network.
    """

    def __init__(self, num_threads: int):
        if not isinstance(num_threads, int) or num_threads <= 0:
            raise ValueError(
                "num_threads should be a positive integer, but got {}"
                "".format(num_threads)
            )
        self._new = num_threads
        self._old = torch.get_num_threads()

    def __enter__(self):
        if self._new != self._old:
            torch.set_num_threads(self._new)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._new != self._old:
            torch.set_num_threads(self._old)


class SamplingOptions:
    r"""Options for Monte Carlo sampling spin configurations."""

    def __init__(
        self,
        number_samples: int,
        number_chains: int = 1,
        number_discarded: Optional[int] = None,
    ):
        r"""Initialises the options.

        :param number_samples: specifies the number of samples per Markov
            chain. Must be a positive integer.
        :param number_chains: specifies the number of independent Markov
            chains. Must be a positive integer.
        :param number_discarded: specifies the number of samples to discard
            in the beginning of each Markov chain (i.e. how long should the
            thermalisation procedure be). If specified, must be a positive
            integer. Otherwise, 10% of ``number_samples`` is used.
        """
        self.number_samples = int(number_samples)
        if self.number_samples <= 0:
            raise ValueError(
                "invalid number_samples: {}; expected a positive integer"
                "".format(number_samples)
            )
        self.number_chains = int(number_chains)
        if self.number_chains <= 0:
            raise ValueError(
                "invalid number_chains: {}; expected a positive integer"
                "".format(number_chains)
            )
        if number_discarded is not None:
            self.number_discarded = int(number_discarded)
            if self.number_discarded <= 0:
                raise ValueError(
                    "invalid number_discarded: {}; expected either a positive "
                    "integer or None".format(number_chains)
                )
        else:
            self.number_discarded = self.number_samples // 10


class SpinDataset(torch.utils.data.Dataset):
    r"""Dataset wrapping spin configurations and corresponding values.

    Each sample will be retrieved by indexing along the first dimension.

    .. note:: This class is very similar to :py:class:`torch.utils.data.TensorDataset`
              except that ``spins`` is a NumPy array of structured type and is
              thus incompatible with :py:class:`torch.utils.data.TensorDataset`.

    :param spins: a NumPy array of :py:class:`CompactSpin`.
    :param values: a NumPy array or a Torch tensor.
    :param bool unpack: if ``True``, ``spins`` will be unpacked into Torch tensors. 
    """

    def __init__(self, spins, values, unpack=None):
        if spins.shape[0] != values.shape[0]:
            raise ValueError(
                "spins and values must have the same size of the first dimension, but "
                "spins.shape={} != values.shape={}".format(spins.shape, values.shape)
            )
        self.spins = spins
        self.values = values
        self.unpack = unpack

    def __len__(self) -> int:
        return self.spins.shape[0]

    def __getitem__(self, index: torch.Tensor):
        if self.unpack is not None:
            return self.unpack(self.spins, index.numpy()), self.values[index]
        return self.spins[index.numpy()], self.values[index]


class IterableSpinDataset(torch.utils.data.IterableDataset):
    r"""Dataset wrapping spin configurations and corresponding values.

    Each sample will be retrieved by indexing along the first dimension.

    .. note:: This class is very similar to :py:class:`torch.utils.data.TensorDataset`
              except that ``spins`` is a NumPy array of structured type and is
              thus incompatible with :py:class:`torch.utils.data.TensorDataset`.

    :param spins: a NumPy array of :py:class:`CompactSpin`.
    :param values: a NumPy array or a Torch tensor.
    :param bool unpack: if ``True``, ``spins`` will be unpacked into Torch tensors. 
    """

    def __init__(self, spins, values, batch_size, drop_last=False, unpack=None):
        if spins.shape[0] != values.shape[0]:
            raise ValueError(
                "spins and values must have the same size along the first dimension, but"
                " spins.shape={} != values.shape={}".format(spins.shape, values.shape)
            )
        if batch_size <= 0:
            raise ValueError(
                "invalid batch_size: {}; expected a positive integer".format(batch_size)
            )
        self.spins = spins
        self.values = values
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.unpack = unpack

    def __len__(self) -> int:
        return self.spins.shape[0]

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError(
                "{} does not play well with parallel data loading"
                "".format(self.__class__.__name__)
            )

        i = 0
        n = self.spins.shape[0]
        while i + self.batch_size <= n:
            xs = self.spins[i : i + self.batch_size]
            if self.unpack is not None:
                xs = self.unpack(xs)
            yield xs, self.values[i : i + self.batch_size]
            i += self.batch_size
        if i < n and not self.drop_last:
            xs = self.spins[i:]
            if self.unpack is not None:
                xs = self.unpack(xs)
            yield xs, self.values[i:]


class BatchedRandomSampler(torch.utils.data.Sampler):
    r"""Samples batches of elements randomly.

    :param int num_samples: number of elements in the dataset
    :param int batch_size: size of mini-batch
    :param bool drop_last: if ``True``, the sampler will ignore the last batch
        if its size would be less than ``batch_size``.

    Example:
        >>> list(BatchedRandomSampler(10, batch_size=3, drop_last=True))
        [tensor([6, 8, 0]), tensor([2, 3, 1]), tensor([4, 7, 9])]
        >>> list(BatchedRandomSampler(10, batch_size=3, drop_last=False))
        [tensor([3, 4, 7]), tensor([0, 2, 8]), tensor([1, 5, 9]), tensor([6])]
    """

    def __init__(self, num_samples: int, batch_size: int, drop_last: bool):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._indices = None

        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer, but got num_samples={}".format(
                    num_samples
                )
            )
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integer, but got batch_size={}".format(
                    batch_size
                )
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean, but got drop_last={}".format(drop_last)
            )

    def __iter__(self):
        if self._indices is None:
            self._indices = torch.randperm(self.num_samples)
        else:
            torch.randperm(self.num_samples, out=self._indices)
        i = 0
        while i + self.batch_size <= self.num_samples:
            yield self._indices[i : i + self.batch_size]
            i += self.batch_size
        if i < self.num_samples and not self.drop_last:
            yield self._indices[i:]

    def __len__(self) -> int:
        """
        Returns the number of batches that will be produced by :py:func:`__iter__`.
        """
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def make_spin_dataloader(
    spins,
    values,
    batch_size: int,
    drop_last: bool = False,
    unpack: bool = _C.unpack,
    shuffle: bool = True,
):
    r"""Creates a new dataloader from packed spin configurations and corresponding
    values.
    """
    if shuffle:
        dataset = SpinDataset(spins, values, unpack)
        sampler = BatchedRandomSampler(len(dataset), batch_size, drop_last)
        # This is a bit of a hack. We want to use our custom batch sampler, but don't
        # want PyTorch to use auto_collate. So we tell PyTorch that our sampler is
        # a plain sampler (i.e. not batch_sampler), but set batch_size to 1 to
        # disable automatic batching
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=sampler,
            collate_fn=lambda x: x[0],
        )
    else:
        dataset = IterableSpinDataset(spins, values, batch_size, drop_last, unpack)
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
        )


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
    r"""
    """
    if out_dim != 1 and out_dim != 2:
        raise ValueError("invalid out_dim: {}; expected either 1 or 2".format(out_dim))

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
    *modules, apply_log: bool = False, use_jit: bool = True, use_classifier: bool = True
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
    if use_classifier:

        class CombiningState(torch.nn.Module):
            __constants__ = ["apply_log"]

            def __init__(self, amplitude: torch.nn.Module, sign: torch.nn.Module):
                super().__init__()
                self.apply_log = apply_log
                self.amplitude = amplitude
                self.sign = sign

            def forward(self, x):
                a = (
                    torch.log(self.amplitude(x))
                    if self.apply_log
                    else self.amplitude(x)
                )
                b = 3.141592653589793 * torch.argmax(self.sign.forward(x), dim=1).to(
                    dtype=torch.float32
                ).view([-1, 1])
                return torch.cat([a, b], dim=1)

    else:

        class CombiningState(torch.nn.Module):
            __constants__ = ["apply_log"]

            def __init__(self, amplitude: torch.nn.Module, phase: torch.nn.Module):
                super().__init__()
                self.apply_log = apply_log
                self.amplitude = amplitude
                self.phase = phase

            def forward(self, x: torch.Tensor):
                a = (
                    torch.log(self.amplitude(x))
                    if self.apply_log
                    else self.amplitude(x)
                )
                b = self.phase(x)
                return torch.cat([a, b], dim=1)

    m = CombiningState(*modules)
    if use_jit:
        m = torch.jit.script(m)
    return m


# def CombiningState(
#     amplitude: torch.nn.Module, sign: torch.nn.Module, use_jit=True, use_log=False
# ) -> torch.nn.Module:
#
#     if use_log:
#
#         class CombiningState(torch.nn.Module):
#             def __init__(self, amplitude, sign):
#                 super().__init__()
#                 self.amplitude = amplitude
#                 self.sign = sign
#
#             def forward(self, x):
#                 A = torch.log(self.amplitude.forward(x))
#                 phase = 3.141592653589793 * torch.argmax(
#                     self.sign.forward(x), dim=1
#                 ).to(dtype=torch.float32).view([-1, 1])
#                 return torch.cat([A, phase], dim=1)
#
#     else:
#
#         class CombiningState(torch.nn.Module):
#             def __init__(self, amplitude, sign):
#                 super().__init__()
#                 self.amplitude = amplitude
#                 self.sign = sign
#
#             def forward(self, x):
#                 y = self.amplitude.forward(x).squeeze()
#                 y *= (1 - 2 * torch.argmax(self.sign.forward(x), dim=1)).to(
#                     dtype=torch.float32
#                 )
#                 return y
#
#     m = CombiningState(amplitude, sign)
#     if use_jit:
#         m = torch.jit.script(m)
#     return m


def _forward_with_batches(state, input, batch_size):
    n = input.shape[0]
    if n == 0:
        raise ValueError("input should not be empty")
    system_size = len(_C.unsafe_get(input, 0))
    i = 0
    out = []
    while i + batch_size <= n:
        out.append(state(_C.unpack(input[i : i + batch_size])))
        i += batch_size
    if i != n:  # Remaining part
        out.append(state(_C.unpack(input[i:]).view(-1, system_size)))
    # r = torch.cat(out, dim=0)
    # assert torch.all(r == state(_C.unpack(input)))
    # return r
    return torch.cat(out, dim=0)


def local_energy(
    state: torch.jit.ScriptModule,
    hamiltonian: _C.Heisenberg,
    spins: np.ndarray,
    log_values: Optional[np.ndarray] = None,
    batch_size: int = 128,
) -> np.ndarray:
    r"""Computes local estimators ⟨σ|H|ψ⟩/⟨σ|ψ⟩ for all σ.

    :param state: wavefunction ``ψ``. ``state`` should be a function
        mapping ``R^{batch_size x in_features}`` to ``R^{batch_size x 2}``.
        Columns of the output are interpreted as real and imaginary parts of
        ``log(⟨σ|ψ⟩)``.
    :param hamiltonian: Hamiltonian ``H``.
    :param spins: spin configurations ``σ``. Should be a non-empty NumPy array
        of py:class:`CompactSpin`.
    :param log_values: pre-computed ``log(⟨σ|ψ⟩)``. Should be a NumPy array of
        ``complex64``.
    :param batch_size: batch size to use for forward propagation through
        ``state``.

    :return: local energies ⟨σ|H|ψ⟩/⟨σ|ψ⟩ as a NumPy array of ``complex64``.
    """
    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            if log_values is None:
                log_values = _forward_with_batches(state, spins, batch_size)
                log_values = log_values.numpy().view(np.complex64)

            # Since torch.jit.ScriptModules can't be directly passed to C++
            # code as torch::jit::script::Modules, we first save ψ to a
            # temporary file and then load it back in C++ code.
            with tempfile.NamedTemporaryFile(delete=False) as f:
                filename = f.name
            try:
                state.save(filename)
                log_H_values = (
                    _C.PolynomialState(
                        _C.Polynomial(hamiltonian, [0.0]),
                        filename,
                        (batch_size, len(_C.unsafe_get(spins, 0))),
                    )(spins)
                    .numpy()
                    .view(np.complex64)
                )
            finally:
                os.remove(filename)
            return np.exp(log_H_values - log_values).squeeze(axis=1)


@torch.jit.script
def _log_amplitudes_to_probabilities(values: torch.Tensor) -> torch.Tensor:
    prob = values - torch.max(values)
    prob *= 2
    prob = torch.exp_(prob)
    prob /= torch.sum(prob)
    return prob


def make_monte_carlo_options(config, number_spins: int) -> _C._Options:
    if number_spins <= 0:
        raise ValueError(
            "invalid number spins: {}; expected a positive integer".format(number_spins)
        )
    sweep_size = config.sweep_size if config.sweep_size is not None else number_spins
    number_discarded = (
        config.number_discarded
        if config.number_discarded is not None
        else config.number_samples // 10
    )
    magnetisation = (
        config.magnetisation if config.magnetisation is not None else number_spins % 2
    )
    return _C._Options(
        number_spins=number_spins,
        magnetisation=magnetisation,
        number_chains=config.number_chains,
        number_samples=config.number_samples,
        sweep_size=sweep_size,
        number_discarded=number_discarded,
    )


def sample_exact(
    state: torch.jit.ScriptModule, options: _C._Options, batch_size: int = 256
) -> Tuple[np.ndarray, torch.Tensor]:
    spins = _C.all_spins(options.number_spins, options.magnetisation)
    num_samples = options.number_samples * options.number_chains
    with torch.no_grad(), torch.jit.optimized_execution(True):
        values = _forward_with_batches(state, spins, batch_size)
        weights = _log_amplitudes_to_probabilities(values).squeeze(dim=1)
        indices = torch.multinomial(weights, num_samples=num_samples, replacement=True)
        return spins[indices.numpy()], values[indices]
