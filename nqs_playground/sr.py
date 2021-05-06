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

from collections import namedtuple
import os
import time
from typing import Dict, List, Tuple, Optional, Union

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
        "linear_system_kwargs",
        "inference_batch_size",
    ],
    defaults=[],
)


@torch.no_grad()
def solve_linear_problem(A: Tensor, b: Tensor, rcond: float) -> Tensor:
    r"""Solve linear problem `A · x = b` where `A` is approximately positive-definite."""
    logger.debug("Computing SVD of {} matrix...", A.size())
    u, s, v = (t.to(A.device) for t in torch.svd(A.cpu()))
    logger.debug("Computing inverse...")
    s_inv = torch.where(
        s > rcond * torch.max(s),
        torch.reciprocal(s),
        torch.scalar_tensor(0, dtype=s.dtype, device=s.device),
    )
    return v.mv(s_inv.mul_(u.t().mv(b)))


def _compute_centered_jacobian(module, inputs, weights):
    parameters = list(filter(lambda p: p.requires_grad, module.parameters()))
    if len(parameters) > 0:
        logger.info("Calculating jacobian...")
        if hasattr(module, "log_prob"):
            f = lambda x: module.log_prob(x)
        else:
            f = module
        gradients = jacobian(f, parameters, inputs)
        with torch.no_grad():
            gradients -= weights @ gradients
        return gradients
    else:
        logger.debug("Module contains no trainable parameters")
        return None


def compute_centered_jacobian(amplitude, phase, inputs, weights):
    Ore = _compute_centered_jacobian(amplitude, inputs, weights)
    Oim = _compute_centered_jacobian(phase, inputs, weights)
    return Ore, Oim


def _compute_gradient_with_curvature(O: Tensor, E: Tensor, weights: Tensor, **kwargs) -> Tensor:
    if O is None:
        return None
    logger.debug("Computing simple gradient...")
    f = torch.mv(O.t(), E)
    logger.debug("Computing covariance matrix...")
    S = torch.einsum("ij,j,jk", O.t(), weights, O)
    logger.info("Computing true gradient δ by solving S⁻¹·δ = f...")
    δ = solve_linear_problem(S, f, **kwargs)
    return δ


def compute_gradient_with_curvature(
    Ore: Tensor, Oim: Tensor, E: Tensor, weights: Tensor, **kwargs
) -> Tensor:
    # The following is an implementation of Eq. (6.22) (without the minus sign) in "Quantum Monte
    # Carlo Approaches for Correlated Systems" by F.Becca & S.Sorella.
    #
    # Basically, what we need to compute is `2/N · Re[E*·(O - ⟨O⟩)]`, where `E` is a `Nx1` vector
    # of complex-numbered local energies, `O` is a `NxM` matrix of logarithmic derivatives, `⟨O⟩` is
    # a `1xM` vector of mean logarithmic derivatives, and `N` is the number of Monte Carlo samples.
    #
    # Now, `Re[a*·b] = Re[a]·Re[b] - Im[a*]·Im[b] = Re[a]·Re[b] +
    # Im[a]·Im[b]` that's why there are no conjugates or minus signs in
    # the code.
    E = 2 * weights * E
    δre = _compute_gradient_with_curvature(Ore, E.real, weights, **kwargs)
    δim = _compute_gradient_with_curvature(Oim, E.imag, weights, **kwargs)
    return δre, δim


class Runner(RunnerBase):
    def __init__(self, config):
        super().__init__(config)
        self.combined_state = combine_amplitude_and_phase(
            self.config.amplitude, self.config.phase, use_jit=False
        )

    @torch.no_grad()
    def _set_gradient(self, grads):
        def run(m, grad):
            i = 0
            for p in filter(lambda x: x.requires_grad, m.parameters()):
                assert p.is_leaf
                n = p.numel()
                if p.grad is not None:
                    p.grad.copy_(grad[i : i + n].view(p.size()))
                else:
                    p.grad = grad[i : i + n].view(p.size())
                i += n
            assert grad is None or i == grad.numel()

        run(self.config.amplitude, grads[0])
        run(self.config.phase, grads[1])

    def inner_iteration(self, states, log_probs, weights):
        logger.info("Inner iteration...")
        tick = time.time()
        E = self.compute_local_energies(states, weights)
        states = states.view(-1, states.size(-1))
        log_probs = states.view(-1)
        weights = weights.view(-1)

        Os = compute_centered_jacobian(self.config.amplitude, self.config.phase, states, weights)
        grads = compute_gradient_with_curvature(*Os, E, weights, **self.config.linear_system_kwargs)
        self._set_gradient(grads)
        self.config.optimizer.step()
        if self.config.scheduler is not None:
            self.config.scheduler.step()
        self.checkpoint(init={"optimizer": self.config.optimizer.state_dict()})
        tock = time.time()
        logger.info("Completed inner iteration in {:.1f} seconds!", tock - tick)
