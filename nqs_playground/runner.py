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
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Optional, Tuple

from .sampling import *
from .hamiltonian import *
from .core import forward_with_batches, _get_dtype, _get_device


class RunnerBase:
    def __init__(self, config):
        self.config = config
        self._epoch = 0
        self._tb_writer = SummaryWriter(log_dir=self.config.output)
        os.makedirs(os.path.join(self.config.output, "epochs"), exist_ok=True)

    @property
    def basis(self):
        if hasattr(self.config, "basis"):
            return self.config.basis
        return self.config.hamiltonian.basis

    @torch.no_grad()
    def do_sampling(self):
        logger.info(
            "Sampling from |ψ(σ)|² with sampling_mode='{}'...", self.config.sampling_mode,
        )
        states, log_probs, info = sample_some(
            self.config.amplitude,
            self.basis,
            self.config.sampling_options._replace(device=_get_device(self.config.amplitude)),
            mode=self.config.sampling_mode,
        )
        if states.dim() == 2:
            states.unsqueeze_(dim=2)
        (number_samples, number_chains, *extra) = states.size()
        if log_probs is not None:
            assert log_probs.size() == (number_samples, number_chains)
            # Compute autocorrelation time based on log(|ψ(s)|²)
            tau = integrated_autocorr_time(log_probs)
            logger.info("Autocorrelation time of log(|ψ(σ)|²): {}", tau)
            self._tb_writer.add_scalar("sampling/tau_log_prob", tau, self._epoch)
        if info is None or "weights" not in info:
            logger.debug(
                "Sampler did not return 'weights'. We assume that no importance sampling "
                "is used and initialize all weights with 1..."
            )
            value = 1.0 / (number_samples * number_chains)
            weights = torch.full(
                (number_samples, number_chains),
                value,
                device=states.device,
                dtype=_get_dtype(self.config.amplitude),
            )
        else:
            weights = info["weights"]
            del info["weights"]
            assert torch.isclose(torch.sum(weights), torch.scalar_tensor(1, dtype=weights.dtype))
        if info is not None and len(info) > 0:
            logger.info("Additional info from the sampler: {}", info)
            for (k, v) in info.items():
                self._tb_writer.add_scalar("sampling/{}".format(k), v, self._epoch)
        return states, log_probs, weights

    def checkpoint(self, init=None):
        if hasattr(self.config, "checkpoint_every"):
            checkpoint_every = self.config.checkpoint_every
        else:
            checkpoint_every = 1
        if self._epoch % checkpoint_every != 0:
            return

        logger.info("Saving weights...")
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

        torch.save(state, os.path.join(folder, "state_dict_{:04d}.pt".format(self._epoch)))

    @torch.no_grad()
    def compute_local_loss(self, states, weights):
        local_energies = local_values(
            states,
            self.config.hamiltonian,
            self.combined_state,
            batch_size=self.config.inference_batch_size,
        )
        tau = integrated_autocorr_time(local_energies)
        logger.info("Autocorrelation time of Eₗ(σ): {}", tau)
        self._tb_writer.add_scalar("sampling/tau_energy", tau, self._epoch)

        # TODO(twesterhout): Add more statistics based on multiple chains
        local_energies = local_energies.view(-1)
        weights = weights.view(-1)
        energy = torch.complex(
            torch.dot(weights, local_energies.real), torch.dot(weights, local_energies.imag),
        )
        logger.info("Energy: {}", energy.item())
        self._tb_writer.add_scalar("loss/energy_real", energy.real.item(), self._epoch)
        self._tb_writer.add_scalar("loss/energy_imag", energy.imag.item(), self._epoch)
        variance = torch.dot(weights, torch.abs(local_energies - energy) ** 2)
        logger.info("Variance: {}", variance.item())
        self._tb_writer.add_scalar("loss/variance", variance.item(), self._epoch)
        return local_energies


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


class _RunnerBase:
    def __init__(self, config):
        self.config = config
        self._epoch = 0
        self._tb_writer = SummaryWriter(log_dir=self.config.output)
        os.makedirs(os.path.join(self.config.output, "epochs"), exist_ok=True)

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
        return _get_device(self.config.amplitude)

    @torch.no_grad()
    def perform_sampling(self) -> Tuple[Tensor, Tensor, Tensor]:
        logger.info("Sampling from |ψ(σ)|²...")
        device = self.device
        dtype = _get_dtype(self.config.amplitude)
        options = self.config.sampling_options._replace(device=device)
        if hasattr(self.config, "sampling_mode"):
            options = options._replace(mode=self.config.sampling_mode)
        states, log_probs, info = sample_some(self.config.amplitude, self.basis, options)
        weights = _determine_initial_weights(states, log_probs, info, dtype)
        if len(info) > 0:
            logger.info("Additional info from the sampler: {}", info)
            for (k, v) in info.items():
                self._tb_writer.add_scalar("sampling/{}".format(k), v, self._epoch)
        if log_probs is not None and options.mode != "full":
            # Compute autocorrelation time based on log(|ψ(s)|²)
            tau = integrated_autocorr_time(log_probs)
            logger.info("Autocorrelation time of log(|ψ(σ)|²): {}", tau)
            self._tb_writer.add_scalar("sampling/tau_log_prob", tau, self._epoch)
        return states, log_probs, weights

    @torch.no_grad()
    def compute_local_energies(self, states: Tensor, weights: Tensor):
        local_energies = local_values(
            states,
            self.config.hamiltonian,
            self.combined_state,
            batch_size=self.config.inference_batch_size,
        )
        tau = integrated_autocorr_time(local_energies)
        logger.info("Autocorrelation time of Eₗ(σ): {}", tau)
        self._tb_writer.add_scalar("sampling/tau_energy", tau, self._epoch)

        scale = 1 / weights.sum(dim=0)
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
        self._tb_writer.add_scalar("loss/energy_real", energy.real, self._epoch)
        self._tb_writer.add_scalar("loss/energy_imag", energy.imag, self._epoch)
        self._tb_writer.add_scalar("loss/energy_err", energy_err, self._epoch)
        self._tb_writer.add_scalar("loss/variance", variance, self._epoch)
        self._tb_writer.add_scalar("loss/variance_err", variance_err, self._epoch)
        return local_energies.view(-1)

    def run(self, number_inner=1):
        while self._epoch < self.config.epochs:
            logger.info("Outer iteration №{}...", self._epoch)
            self.outer_iteration(number_inner)
            self._epoch += 1

    @torch.no_grad()
    def compute_log_probs(self, states: Tensor) -> Tensor:
        return forward_with_batches(
            self.log_prob_fn,
            states.view(-1, states.size(-1)),
            batch_size=self.config.inference_batch_size,
        ).view(states.size()[:2])

    def outer_iteration(self, number_inner: int):
        (states, original_log_probs, original_weights) = self.perform_sampling()
        if original_log_probs is None and number_inner > 1:
            original_log_probs = self.compute_log_probs(states)
        self._inner = 0
        self.inner_iteration(states, original_log_probs, original_weights)
        for i in range(1, number_inner):
            log_probs = self.compute_log_probs(states)
            weights = original_weights * torch.exp_(log_probs - original_log_probs)
            self._inner = i
            self.inner_iteration(states, log_probs, weights)

    def checkpoint(self, epoch: Optional[int] = None, inner: Optional[int] = None, init=None):
        logger.info("Saving weights...")
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

        if epoch is None:
            epoch = self._epoch
        if inner is None:
            inner = self._inner
        torch.save(state, os.path.join(folder, "state_dict_{:04d}_{:03d}.pt".format(epoch, inner)))
