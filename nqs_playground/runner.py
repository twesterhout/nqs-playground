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

from .sampling import *
from .hamiltonian import *
from .core import _get_dtype, _get_device


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
        logger.info("Sampling from |ψ(σ)|² with sampling_mode='{}'...", self.config.sampling_mode)
        states, log_probs, info = sample_some(
            self.config.amplitude,
            self.basis,
            self.config.sampling_options,
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

    def checkpoint(self):
        logger.info("Saving weights...")
        folder = os.path.join(self.config.output, "epochs", "{:04d}".format(self._epoch))
        os.makedirs(folder, exist_ok=True)
        torch.save(
            self.config.amplitude.state_dict(),
            os.path.join(folder, "amplitude_weights.pt"),
        )
        # We do not know whether phase of sign network is used, so we check both
        if hasattr(self.config, "phase"):
            torch.save(
                self.config.phase.state_dict(),
                os.path.join(folder, "phase_weights.pt"),
            )
        if hasattr(self.config, "sign"):
            torch.save(
                self.config.sign.state_dict(),
                os.path.join(folder, "sign_weights.pt"),
            )

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
            torch.dot(weights, local_energies.real), torch.dot(weights, local_energies.imag)
        )
        logger.info("Energy: {}", energy.item())
        self._tb_writer.add_scalar("loss/energy_real", energy.real.item(), self._epoch)
        self._tb_writer.add_scalar("loss/energy_imag", energy.imag.item(), self._epoch)
        variance = torch.dot(weights, torch.abs(local_energies - energy) ** 2)
        logger.info("Variance: {}", variance.item())
        self._tb_writer.add_scalar("loss/variance", variance.item(), self._epoch)
        return local_energies
