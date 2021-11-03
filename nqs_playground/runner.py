# Copyright Tom Westerhout (c) 2021
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
    "RunnerBase",
]

import os
from loguru import logger
import numpy as np
import time
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Optional, Tuple

from .sampling import *
from .hamiltonian import *
from .core import forward_with_batches, get_dtype, get_device


def _determine_initial_weights(
    states: Tensor, log_probs: Tensor, info: Dict[str, Any], dtype=None
) -> Tensor:
    weights = info.get("weights")
    if weights is None:
        logger.debug(
            "Sampler did not return 'weights'. We assume that no importance sampling "
            "is used and initialize all weights with 1..."
        )
        value = 1.0 / (states.size(0) * states.size(1))
        device = states.device
        if dtype is None:
            dtype = log_probs.dtype
        weights = torch.full(states.size()[:2], value, device=device, dtype=dtype)
    else:
        assert torch.isclose(torch.sum(weights), torch.scalar_tensor(1, dtype=weights.dtype))
        del info["weights"]
    return weights


class RunnerBase:
    def __init__(self, config):
        self.config = config
        self.global_index = 0
        self.tb_writer = SummaryWriter(log_dir=self.config.output)

    @property
    def basis(self):
        if hasattr(self.config, "basis"):
            return self.config.basis
        return self.config.hamiltonian.basis

    @property
    def amplitude_forward_fn(self):
        if hasattr(self.config.amplitude, "log_prob"):
            return lambda x: 0.5 * self.config.amplitude.log_prob(x).view(-1, 1)
        return self.config.amplitude

    @property
    def log_prob_fn(self):
        if hasattr(self.config.amplitude, "log_prob"):
            return self.config.amplitude.log_prob
        return lambda x: 2 * self.config.amplitude(x)

    @property
    def device(self):
        if hasattr(self.config, "device"):
            return self.config.device
        return get_device(self.config.amplitude)

    @property
    def dtype(self):
        return get_dtype(self.config.amplitude)

    @torch.no_grad()
    def perform_sampling(self) -> Tuple[Tensor, Tensor, Tensor]:
        logger.info("Sampling from |ψ(σ)|²...")
        self.config.amplitude.eval()
        device = self.device
        dtype = self.dtype
        options = self.config.sampling_options._replace(device=device)
        states, log_probs, info = sample_some(self.config.amplitude, self.basis, options)
        weights = _determine_initial_weights(states, log_probs, info, dtype=torch.float64)
        if len(info) > 0:
            logger.info("Additional info from the sampler: {}", info)
            for (k, v) in info.items():
                self.tb_writer.add_scalar("sampling/{}".format(k), v, self.global_index)
        if log_probs is not None and options.mode != "full":
            assert not torch.any(torch.isnan(log_probs))
            # Compute autocorrelation time based on log(|ψ(s)|²)
            tau = integrated_autocorr_time(log_probs)
            logger.info("Autocorrelation time of log(|ψ(σ)|²): {}", tau)
            self.tb_writer.add_scalar("sampling/tau_log_prob", tau, self.global_index)
        return states, log_probs, weights

    @torch.no_grad()
    def compute_local_energies(self, states: Tensor, weights: Tensor) -> Tensor:
        self.combined_state.eval()
        local_energies = local_values(
            states,
            self.config.hamiltonian,
            self.combined_state,
            batch_size=self.config.inference_batch_size // self.basis.number_spins,
            debug=False,
        )
        assert not torch.any(torch.isnan(local_energies))
        tau = integrated_autocorr_time(local_energies)
        logger.info("Autocorrelation time of Eₗ(σ): {}", tau)
        self.tb_writer.add_scalar("sampling/tau_energy", tau, self.global_index)

        scale = torch.reciprocal(weights.sum(dim=0))
        weighted_energies = scale * torch.complex(
            weights * local_energies.real, weights * local_energies.imag
        )
        energy_per_chain = weighted_energies.sum(dim=0, keepdim=True)
        variance_per_chain = weights * torch.abs((local_energies - energy_per_chain) ** 2)
        variance_per_chain = scale * variance_per_chain.sum(dim=0, keepdim=True)

        energy_err, energy = torch.std_mean(energy_per_chain)
        energy_err, energy = energy_err.item().real, energy.item()
        variance_err, variance = torch.std_mean(variance_per_chain)
        variance_err, variance = variance_err.item(), variance.item()

        logger.info("  Energy: {} ± {:.2e}", energy, energy_err)
        logger.info("Variance: {} ± {:.2e}", variance, variance_err)
        self.tb_writer.add_scalar("loss/energy_real", energy.real, self.global_index)
        self.tb_writer.add_scalar("loss/energy_imag", energy.imag, self.global_index)
        self.tb_writer.add_scalar("loss/energy_err", energy_err, self.global_index)
        self.tb_writer.add_scalar("loss/variance", variance, self.global_index)
        self.tb_writer.add_scalar("loss/variance_err", variance_err, self.global_index)
        return local_energies.view(-1)

    def run(self, number_inner: int = 1):
        for i in range(self.config.epochs):
            logger.info("Outer iteration №{}...", i)
            tick = time.time()
            self.outer_iteration(number_inner)
            tock = time.time()
            logger.info("Completed outer iteration in {:.1f} seconds!", tock - tick)

    @torch.no_grad()
    def compute_log_probs(self, states: Tensor) -> Tensor:
        return forward_with_batches(
            self.log_prob_fn,
            states.view(-1, states.size(-1)),
            batch_size=self.config.inference_batch_size,
        ).view(states.size()[:2])

    def outer_iteration(self, number_inner: int):
        (states, log_probs, weights) = self.perform_sampling()
        if log_probs is None:
            log_probs = self.compute_log_probs(states)
            assert not torch.any(torch.isnan(log_probs))
        weights = weights.to(torch.float64)
        log_probs = log_probs.to(torch.float64)

        original_log_probs = log_probs
        log_original_weights = weights.to(torch.float64, copy=True).log_()
        self.inner_iteration(states, log_probs, weights)
        self.global_index += 1
        for i in range(1, number_inner):
            log_probs = self.compute_log_probs(states)
            assert not torch.any(torch.isnan(log_probs))
            log_probs = log_probs.to(torch.float64)
            with torch.no_grad():
                weights = log_original_weights + log_probs - original_log_probs
                weights -= torch.max(weights, dim=0, keepdim=True)[0] - 5.0
                weights = torch.exp_(weights)
                weights /= torch.sum(weights)
                assert not torch.any(torch.isnan(weights))
            self.inner_iteration(states, log_probs, weights)
            self.global_index += 1

    def checkpoint(self, init=None):
        folder = os.path.join(self.config.output, "checkpoints")
        os.makedirs(folder, exist_ok=True)
        state = init
        if state is None:
            state = {}
        state["amplitude"] = self.config.amplitude.state_dict()
        # We do not know whether phase of sign network is used, so we check both
        if hasattr(self.config, "phase"):
            state["phase"] = self.config.phase.state_dict()
        if hasattr(self.config, "sign"):
            state["sign"] = self.config.sign.state_dict()
        torch.save(state, os.path.join(folder, "state_dict_{:05d}.pt".format(self.global_index)))
