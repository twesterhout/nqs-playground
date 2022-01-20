# Copyright Tom Westerhout (c) 2020-2021
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
import time
import os
from loguru import logger
import numpy as np
import torch
from torch import Tensor
import torch.utils.tensorboard.summary
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any

from . import *

Config = namedtuple(
    "Config",
    [
        "amplitude",
        "phase",
        "hamiltonian",
        "output",
        "epochs",
        "sampling_options",
        "optimizer",
        "scheduler",
        "exact",
        "constraints",
        "inference_batch_size",
        "checkpoint_every",
    ],
    defaults=[],
)


def _should_optimize(module):
    return any(map(lambda p: p.requires_grad, module.parameters()))


# class Runner(RunnerBase):
#     def __init__(self, config):
#         super().__init__(config)
#         self.combined_state = combine_amplitude_and_phase(
#             self.config.amplitude, self.config.phase, use_jit=False
#         )
#
#     def hamming_weight_loss(self, states, output, target=None):
#         number_spins = self.basis.number_spins
#         if target is None:
#             target = number_spins // 2
#         device = states.device
#         loss = hamming_weight(states.cpu(), number_spins).to(device)
#         loss = torch.abs(loss - target)
#         loss = loss.view(output.size())
#         r = torch.logsumexp(torch.log(loss) + 2 * output, dim=0)
#         r = r - np.log(states.size(0))
#         return r, loss.sum().item()
#
#     def step(self):
#         self._epoch += 1
#         logger.info("Starting epoch {}...", self._epoch)
#         tick = time.time()
#         states, _, weights = self.do_sampling()
#         E = self.compute_local_loss(states, weights)
#         states = states.view(-1, states.size(-1))
#         weights = weights.view(-1)
#
#         # Compute output gradient
#         with torch.no_grad():
#             grad = E.real - torch.dot(E.real, weights)
#             grad *= 2 * weights
#             grad = grad.view(-1, 1)
#
#         self.config.optimizer.zero_grad()
#         batch_size = self.config.inference_batch_size
#         # Computing gradients for the amplitude network
#         logger.info("Computing gradients...")
#         with_hamming_weight_constraint = "hamming_weight" in self.config.constraints
#         if _should_optimize(self.config.amplitude):
#             if hasattr(self.config.amplitude, "log_prob"):
#                 forward_fn = lambda x: 0.5 * self.config.amplitude.log_prob(x).view(-1, 1)
#             else:
#                 forward_fn = self.config.amplitude
#
#             if with_hamming_weight_constraint:
#                 hamming_weight_loss = 0.0
#             for (states_chunk, grad_chunk) in split_into_batches((states, grad), batch_size):
#                 output = forward_fn(states_chunk)
#                 output.backward(grad_chunk, retain_graph=True)
#                 if with_hamming_weight_constraint:
#                     loss, count = self.hamming_weight_loss(states_chunk, output)
#                     hamming_weight_loss += count
#                     if count > 0:
#                         loss = self.config.constraints["hamming_weight"](self._epoch) * loss
#                         loss.backward()
#             if with_hamming_weight_constraint:
#                 hamming_weight_loss /= states.size(0)
#                 logger.info("Hamming weight loss: {}".format(hamming_weight_loss))
#                 self._tb_writer.add_scalar("loss/hamming_weight", hamming_weight_loss, self._epoch)
#         # Computing gradients for the phase network
#         if _should_optimize(self.config.phase):
#             forward_fn = self.config.phase
#             for (states_chunk, grad_chunk) in split_into_batches((states, grad), batch_size):
#                 output = forward_fn(states_chunk)
#                 output.backward(grad_chunk)
#
#         self.config.optimizer.step()
#         if self.config.scheduler is not None:
#             self.config.scheduler.step()
#         self.checkpoint({"optimizer": self.config.optimizer.state_dict()})
#         tock = time.time()
#         logger.info("Completed epoch {}! It took {:.1f} seconds...", self._epoch, tock - tick)


def optimizer_hparams(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    hparams = {"optimizer": optimizer.__class__.__name__}
    for g in optimizer.state_dict()["param_groups"]:
        for key, value in g.items():
            if key in {"lr", "momentum", "weight_decay"}:
                hparams[key] = value
    return hparams


def sampling_hparams(options: SamplingOptions) -> Dict[str, Any]:
    return options.hparams()


class Runner(RunnerBase):
    def __init__(self, config):
        super().__init__(config)
        self.combined_state = combine_amplitude_and_phase(
            self.config.amplitude, self.config.phase, use_jit=False
        )
        self.tb_writer.add_hparams(
            {
                **optimizer_hparams(self.config.optimizer),
                **sampling_hparams(self.config.sampling_options),
            },
            {"dummy": 0}, # REQUIED! Otherwise, Tensorboard won't display hparams at all
        )
        self.tb_writer.flush()

    def inner_iteration(self, states, log_probs, weights):
        assert weights.dtype == torch.float64
        assert log_probs.dtype == torch.float64
        logger.info("Inner iteration...")
        tick = time.time()
        E = self.compute_local_energies(states, weights)
        E = E.real.to(dtype=weights.dtype)
        states = states.view(-1, states.size(-1))
        log_probs = log_probs.view(-1)
        weights = weights.view(-1)

        # Compute output gradient
        with torch.no_grad():
            grad = E - torch.dot(E, weights)
            grad *= 2 * weights
            grad = grad.view(-1, 1)
            grad_norm = torch.linalg.norm(grad)
            logger.info("‖∇E‖₂ = {}", grad_norm)
            self.tb_writer.add_scalar("loss/grad", grad_norm, self.global_index)

        # checkpoint_name = "checkpoint_{}.pt".format(self.global_index)
        # if not os.path.exists(checkpoint_name):
        #     logger.debug("Saving...")
        #     obj = {
        #         "states": states,
        #         "log_probs": log_probs,
        #         "weights": weights,
        #         "grad": grad,
        #     }
        #     torch.save(obj, checkpoint_name)
        # else:
        #     logger.debug("Loading...")
        #     obj = torch.load(checkpoint_name)
        #     if not torch.all(states == obj["states"]):
        #         other_states = obj["states"]
        #         for i in range(states.size(0)):
        #             if not torch.all(states[i] == other_states[i]):
        #                 logger.error("{}: {} != {}", i, states[i, 0], other_states[i, 0])
        #     assert torch.all(log_probs == obj["log_probs"])
        #     assert torch.all(weights == obj["weights"])
        #     assert torch.all(grad == obj["grad"])

        self.config.optimizer.zero_grad()
        batch_size = self.config.inference_batch_size

        # Computing gradients for the amplitude network
        logger.info("Computing gradients...")
        if _should_optimize(self.config.amplitude):
            self.config.amplitude.train()
            forward_fn = self.amplitude_forward_fn
            for (states_chunk, grad_chunk) in split_into_batches((states, grad), batch_size):
                output = forward_fn(states_chunk)
                output.backward(grad_chunk)  # , retain_graph=True)

        # Computing gradients for the phase network
        if _should_optimize(self.config.phase):
            self.config.phase.train()
            forward_fn = self.config.phase
            for (states_chunk, grad_chunk) in split_into_batches((states, grad), batch_size):
                output = forward_fn(states_chunk)
                output.backward(grad_chunk)

        self.config.optimizer.step()
        if self.config.scheduler is not None:
            self.config.scheduler.step()
        if self.global_index % self.config.checkpoint_every == 0:
            self.checkpoint(init={"optimizer": self.config.optimizer.state_dict()})
        tock = time.time()
        logger.info("Completed inner iteration in {:.1f} seconds!", tock - tick)
