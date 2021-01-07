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
from loguru import logger
import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from . import *

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
        load = lambda m: load_model(m, number_spins=self.basis.number_spins, jit=False).to(self.device)
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
            self.__combined_state = combine_amplitude_and_phase(self.amplitude, self.phase, use_jit=False)
        return self.__combined_state

    @torch.no_grad()
    def compute_overlap(self):
        if self.ground_state is None:
            return None
        spins = torch.from_numpy(self.basis.states.view(np.int64)).to(self.device)
        state = forward_with_batches(self.combined_state, spins, self.inference_batch_size)
        if state.dim() > 1:
            state.squeeze_(dim=1)
        state.real -= torch.max(state.real)
        state.exp_()
        state = state.cpu().numpy()
        overlap = abs(np.dot(state, self.ground_state)) / np.linalg.norm(state)
        return overlap

    @torch.no_grad()
    def sample(self):
        logger.info("Sampling from |ψ(σ)|² with sampling_mode={}...", self.config.sampling_mode)
        spins, log_probs, info = sample_some(
            self.amplitude,
            self.basis,
            self.sampling_options,
            mode=self.config.sampling_mode,
        )
        if log_probs is not None:
            # Compute autocorrelation time based on log(|ψ(s)|²)
            tau = integrated_autocorr_time(log_probs)
            logger.info("Autocorrelation time of log(|ψ(σ)|²: {}", tau)

        if "weights" not in info:
            logger.debug(
                "Sampler did not return 'weights'. We assume that no importance sampling "
                "is used and initialize all weights with 1..."
            )
            if log_probs is not None:
                weights = torch.ones_like(log_probs)
            else:
                weights = torch.ones(spins.size()[:-1], device=self.device)
            weights /= torch.sum(weights)
        else:
            weights = info["weights"]
        return spins, weights

    def calculate_gradient(self, spins, local_loss, weights):
        logger.info("Calculating gradients...")
        if local_loss.dtype == torch.complex64 or local_loss.dtype == torch.complex128:
            local_loss = local_loss.real
        mean_loss = torch.dot(local_loss, weights)
        grad = (local_loss - mean_loss) * weights
        grad = grad.view(-1, 1)

        self.optimiser.zero_grad()
        if any(map(lambda p: p.requires_grad, self.amplitude.parameters())):
            if hasattr(self.amplitude, "log_prob"):
                output = 0.5 * self.amplitude.log_prob(spins).view(-1, 1)
            else:
                output = self.amplitude(spins)
            output.backward(grad)
        if any(map(lambda p: p.requires_grad, self.phase.parameters())):
            output = self.phase(spins)
            output.backward(grad)

    def step(self):
        logger.info("Iteration {}...", self._iteration)
        overlap = self.compute_overlap()
        if overlap is not None:
            logger.info("Overlap with ground state: {}", overlap)
        spins, weights = self.sample()
        local_energies = local_values(
            spins,
            self.hamiltonian,
            self.combined_state,
            batch_size=self.inference_batch_size,
            debug=False,
        )
        # Compute autocorrelation time based on E_loc(σ)
        if self.config.sampling_mode != "full":
            tau = integrated_autocorr_time(local_energies)
            logger.info("Autocorrelation time of local energies: {}", tau)
        # Reshape spins to flatten dimensions representing Markov chains
        spins = spins.view(-1, spins.size(-1))
        local_energies = local_energies.view(-1)
        weights = weights.view(-1)

        energy = torch.dot(local_energies, weights.to(local_energies.dtype))
        logger.info("Energy: {}", energy)
        variance = torch.dot(torch.abs(local_energies - energy)**2, weights)
        logger.info("Energy variance: {}", variance)
        self.calculate_gradient(spins, local_energies, weights)
        self.optimiser.step()
        self._iteration += 1
