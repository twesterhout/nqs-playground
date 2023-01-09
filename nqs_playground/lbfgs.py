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

from LBFGS import LBFGS

from collections import namedtuple
import time
import os
from loguru import logger
import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

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


class Runner(RunnerBase):
    def __init__(self, config):
        super().__init__(config)
        self.combined_state = combine_amplitude_and_phase(
            self.config.amplitude, self.config.phase, use_jit=False
        )

    def outer_iteration(self, number_inner: int):
        (states, log_probs, weights) = self.perform_sampling()
        if log_probs is None:
            log_probs = self.compute_log_probs(states)
            assert not torch.any(torch.isnan(log_probs))
        weights = weights.to(torch.float64)
        log_probs = log_probs.to(torch.float64)

        original_log_probs = log_probs
        log_original_weights = weights.to(torch.float64, copy=True).log_()

        def closure():
            nonlocal states
            log_probs = self.compute_log_probs(states).to(torch.float64)
            weights = recompute_weights(log_probs, original_log_probs, log_original_weights)
            local_energies, energy, _, info = local_values_with_extras(
                (states, None, weights),
                self.config.hamiltonian,
                self.combined_state,
                self.config.inference_batch_size
            )
            log_stuff_to_tensorboard(info, self.global_index, self.tb_writer)

            weights = weights.view(-1)
            local_energies = local_energies.real.view(-1)
            grad = 2 * weights * (local_energies - energy.real)
            grad = grad.view(-1, 1)

            self.config.optimizer.zero_grad()
            self.config.amplitude.train()
            output = torch.sum(self.config.amplitude(states.view(-1, states.size(-1))) * grad)
            print(output)
            assert output == energy.real
            return output
            # forward_fn = self.amplitude_forward_fn
            # for (states_chunk, grad_chunk) in split_into_batches((, grad), 1024):
            #     output = forward_fn(states_chunk)
            #     output.backward(grad_chunk)

            # grad = self.config.optimizer._gather_flat_grad()
            # return grad, energy.real


        for i in range(number_inner):

            grad, energy = compute_gradient(should_recompute_weights=i > 0)
            p = self.config.optimizer.two_loop_recursion(-grad)

            # @torch.no_grad()
            # def closure():
            #     log_probs = self.compute_log_probs(states).to(torch.float64)
            #     weights = recompute_weights(log_probs, original_log_probs, log_original_weights)
            #     _, energy, _, info = local_values_with_extras(
            #         (states, None, weights),
            #         self.config.hamiltonian,
            #         self.combined_state,
            #         self.config.inference_batch_size
            #     )
            #     log_stuff_to_tensorboard(info, self.global_index, self.tb_writer)
            #     return energy.real

            options = {'closure': closure, 'current_loss': energy}
            energy, grad, lr, _, _, _, _, _ = self.config.optimizer.step(p, grad, options=options)
            self.config.optimizer.curvature_update(grad)

            self.global_index += 1
