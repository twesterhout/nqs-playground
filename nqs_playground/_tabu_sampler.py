#!/usr/bin/env python3

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

from typing import Callable, List, Tuple, Optional

import torch
from torch import Tensor

from . import _C
from ._C import (
    zanella_next_state_index,
    zanella_waiting_time,
    zanella_jump_rates,
    zanella_choose_samples,
)
from .monte_carlo import SamplingOptions, _prepare_initial_state


# @torch.no_grad()
# def zanella_jump_rates(current_log_prob, proposed_log_prob, counts):
#     rates = torch.empty_like(proposed_log_prob)
#     rates.copy_(proposed_log_prob)
#     rates_batches = torch.split(rates, counts)
#     for current, r in zip(current_log_prob, rates_batches):
#         r -= current
#     torch.exp_(rates)
#     torch.min(
#         rates, torch.scalar_tensor(1, dtype=rates.dtype, device=rates.device), out=rates
#     )
#     rates_sum = torch.tensor([r.sum() for r in rates_batches])
#     return rates, rates_sum


# @torch.jit.script
# @torch.no_grad()
# def pick_next_state_index(jump_rates: Tensor, counts: List[int]) -> Tensor:
#     indices = torch.empty(len(counts), dtype=torch.int64, device=jump_rates.device)
#     offset = 0
#     for i, n in enumerate(counts):
#         indices[i] = (
#             offset
#             + torch.multinomial(jump_rates[offset : offset + n], num_samples=1).item()
#         )
#         offset += n
#     return indices

# return torch.tensor([
#     torch.multinomial(r, num_samples=1).item()
#         for r in torch.split(jump_rates, counts)
# ])


# @torch.jit.script
# @torch.no_grad()
# def zanella_waiting_time(rates: Tensor, out: Optional[Tensor]) -> Tensor:
#     u = torch.rand(rates.size(), dtype=rates.dtype, device=rates.device)
#     u.log_()
#     u *= -1
#     u /= rates
#     if out is not None:
#         out.copy_(u)
#         return out
#     return u


@torch.no_grad()
def zanella_process(
    current_state: Tensor,
    log_prob_fn: Callable[[Tensor], Tensor],
    generator_fn: Callable[[Tensor], Tuple[Tensor, List[int]]],
    number_samples: int,
    number_discarded: int,
):
    r"""

    :param current_state: State from which to start the process. The first
        dimension is the batch dimension. It corresponds to multiple chains.
    :param log_prob_fn: Function returning the logarithmic probability of a
        state. It must support batching, i.e. work with tensors similar to
        `current_state`.
    :param generator_fn: 
    :param number_samples:
    """
    assert number_samples >= 1
    cpu = torch.device("cpu")
    # Device is determined by the location of initial state
    device = current_state.device
    current_log_prob = log_prob_fn(current_state)
    if current_log_prob.dim() > 1:
        current_log_prob.squeeze_(dim=1)
    # Number of chains is also deduced from current_state. It is simply
    # current_state.size(0). In the following we pre-allocate storage for
    # states and log probabilities.
    states = current_state.new_empty((number_samples,) + current_state.size())
    log_prob = current_log_prob.new_empty((number_samples,) + current_log_prob.size())
    # Weights stores weights of samples, i.e. time we spend sitting there
    weights = torch.empty(number_samples, current_state.size(0))
    # Store results of the first iteration. Note that current_weight is not yet
    # computed! It will be done inside the loop
    states[0] = current_state
    log_prob[0] = current_log_prob
    current_weight = weights[0]
    # Main loop. We keep track of the iteration manually since we want to stop
    # in the middle of the loop body rather than at the end. We also keep a
    # flag which indicated whether we are still in the thermalisation phase and
    # that samples should be discarded
    iteration = 0
    discard = True
    while True:
        # Generates all states to which we could jump
        possible_state, counts = generator_fn(current_state)
        possible_log_prob = log_prob_fn(possible_state)
        if possible_log_prob.dim() > 1:
            possible_log_prob.squeeze_(dim=1)
        jump_rates, jump_rates_sum = zanella_jump_rates(
            current_log_prob, possible_log_prob, counts
        )
        # Calculate for how long we have to sit in the current state
        # Note that only now have we computed all quantities for `iteration`.
        zanella_waiting_time(jump_rates_sum, out=current_weight)

        iteration += 1
        if iteration % 1000 == 0:
            print(
                "Information :: Patience! At {:.2f}%...".format(
                    100 * iteration / number_samples
                )
            )
        if discard:
            if iteration == number_discarded:
                iteration = 0
                discard = False
        else:
            if iteration == number_samples:
                break
            current_state = states[iteration]
            current_log_prob = log_prob[iteration]
            current_weight = weights[iteration]

        # Pick the next state
        indices = zanella_next_state_index(jump_rates, jump_rates_sum, counts)
        torch.index_select(possible_state, dim=0, index=indices, out=current_state)
        torch.index_select(
            possible_log_prob, dim=0, index=indices, out=current_log_prob
        )

    return states, log_prob, weights


@torch.no_grad()
def sample_using_zanella(log_prob_fn, basis, options):
    current_state = _prepare_initial_state(basis, options.number_chains)
    current_state = current_state.to(options.device)
    generator_fn = _C._ProposalGenerator(basis)
    sweep_size = options.sweep_size if options.sweep_size is not None else 1
    states, log_prob, weights = zanella_process(
        current_state,
        log_prob_fn,
        generator_fn,
        options.number_samples * sweep_size,
        options.number_discarded * sweep_size,
    )
    final_states = states.new_empty((options.number_samples,) + states.size()[1:])
    final_log_prob = log_prob.new_empty((options.number_samples,) + log_prob.size()[1:])
    device = final_states.device
    time_step = torch.sum(weights, dim=0)
    time_step /= options.number_samples
    for chain in range(weights.size(1)):
        indices = zanella_choose_samples(
            weights[:, chain], options.number_samples, time_step[chain].item(), device,
        )
        torch.index_select(
            states[:, chain], dim=0, index=indices, out=final_states[:, chain]
        )
        torch.index_select(
            log_prob[:, chain], dim=0, index=indices, out=final_log_prob[:, chain]
        )
    return final_states, final_log_prob
