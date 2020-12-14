# Copyright Tom Westerhout (c) 2020
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

from collections import namedtuple
import os
import time
from loguru import logger
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

if torch.has_cuda:
    import threading
    from torch._utils import ExceptionWrapper

from nqs_playground import *

Config = namedtuple(
    "Config",
    [
        #
        "model",
        "output",
        "hamiltonian",
        "epochs",
        "sampling_options",
        "optimiser",
        ## OPTIONAL
        "device",
        "exact",
        "sampling_mode",
        "inference_batch_size",
    ],
    defaults=["cpu", None, "exact", 8192],
)


class Runner:
    def __init__(self, config):
        # "const" stuff
        self.config = config
        self.device = load_device(self.config)
        self.hamiltonian = self.config.hamiltonian
        self.basis = self.hamiltonian.basis
        self.amplitude, self.phase = self._load_models()
        self.optimiser = self._load_optimiser()
        self.tb_writer = self._load_loggers()
        self.sampling_options = self.config.sampling_options
        self.ground_state = load_exact(self.config.exact)
        self.inference_batch_size = self.config.inference_batch_size
        # Mutable attributes
        self._iteration = 0

    def _load_models(self):
        r"""Constructs amplitude and phase networks."""
        model = self.config.model
        if not isinstance(self.config.model, (tuple, list)):
            raise TypeError(
                "config.model has wrong type: {}; expected either a pair of "
                "torch.nn.Modules or a pair of filenames".format(type(model))
            )
        amplitude, phase = model
        load = lambda m: load_model(m, number_spins=self.basis.number_spins).to(self.device)
        amplitude, phase = load(amplitude), load(phase)
        return amplitude, phase

    def _load_optimiser(self):
        return load_optimiser(
            self.config.optimiser,
            list(self.amplitude.parameters()) + list(self.phase.parameters()),
        )

    def _load_loggers(self):
        return SummaryWriter(log_dir=self.config.output)

    def _using_full_sum(self):
        return self.config.sampling_mode == "full"

    @property
    def combined_state(self):
        if not hasattr(self, "__combined_state"):
            self.__combined_state = combine_amplitude_and_phase(self.amplitude, self.phase)
        return self.__combined_state

    @torch.no_grad()
    def compute_overlap(self):
        if self.ground_state is None:
            return None
        spins = torch.from_numpy(self.basis.states.view(np.int64)).to(self.device)
        state = forward_with_batches(self.combined_state, spins, self.inference_batch_size)
        state.real -= torch.max(state.real)
        state.exp_()
        state = state.cpu().numpy()
        overlap = abs(np.dot(state, self.ground_state)) / np.linalg.norm(state)
        return overlap

    def _energy_and_variance(self, local_values, weights) -> Tuple[complex, float]:
        r"""Given local energies, computes an energy estimate and variance
        estimate.
        """
        if weights is not None:
            energy = np.dot(weights, local_values)
            variance = np.dot(weights, np.abs(local_values - energy) ** 2)
        else:
            energy = np.mean(local_values)
            variance = np.var(local_values)
        return energy, variance

    @torch.no_grad()
    def sample(self):
        spins, log_probs, info = sample_some(
            self.amplitude,
            self.basis,
            self.sampling_options,
            mode=self.config.sampling_mode,
        )
        weights = info["weights"] # torch.ones_like(log_probs)
        assert not torch.any(torch.isnan(weights))
        return spins, weights

    def _set_gradient(self, grad):
        def run(m, i):
            for p in filter(lambda x: x.requires_grad, m.parameters()):
                assert p.is_leaf
                n = p.numel()
                if p.grad is not None:
                    p.grad.view(-1).copy_(grad[i : i + n])
                else:
                    p.grad = grad[i : i + n].view(p.size())
                i += n
            return i

        with torch.no_grad():
            i = run(self.amplitude, 0)
            _ = run(self.phase, i)

    def calculate_gradient(self, spins, local_energies, weights):
        spins = spins.view(-1, spins.size(-1))
        # jacobian = jacobian_simple(self.amplitude, spins)
        # print(jacobian)
        # weights = weights.view(1, -1)
        # jacobian -= weights @ jacobian
        # local_energies = local_energies.view(1, -1).real.to(weights.dtype)
        # grad = local_energies @ jacobian
        # self._set_gradient(grad.view(-1))
        grad = (local_energies - torch.mean(local_energies)) * weights
        # grad.requires_grad = True
        # print(local_energies.dtype, weights.dtype)
        self.optimiser.zero_grad()
        self.amplitude(spins).backward(grad.real.view(-1, 1))
        # self.phase(spins).backward(grad)

    def step(self):
        spins, weights = self.sample()
        local_energies = local_values(
            spins, self.hamiltonian, self.combined_state, batch_size=self.inference_batch_size
        )
        assert not torch.any(torch.isnan(local_energies))
        logger.info("local_energies: {}", local_energies.view(-1))
        logger.info("{} Energy: {}", self._iteration, torch.sum(weights * local_energies).item())
        self.calculate_gradient(spins, local_energies, weights)
        self.optimiser.step()
        self._iteration += 1
