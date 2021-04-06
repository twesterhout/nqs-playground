#!/usr/bin/env python3

from collections import namedtuple
from enum import Enum
import inspect
import math
import time
from typing import Dict, List, Tuple, Optional

from loguru import logger
import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from nqs_playground import *
import nqs_playground._C as _C
from nqs_playground.core import _get_device


TrainingOptions = namedtuple(
    "TrainingOptions", ["epochs", "batch_size", "optimizer", "scheduler"], defaults=[None],
)

Config = namedtuple(
    "Config",
    [
        "amplitude",
        "phase",
        "output",
        "hamiltonian",
        "evolution_operator",
        "epochs",
        "sampling_options",
        "inference_batch_size",
        "training_options",
    ],
)


class TensorIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, *tensors, batch_size=1, shuffle=False):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        assert all(tensors[0].device == tensor.device for tensor in tensors)
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle

    @property
    def device(self):
        return self.tensors[0].device

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.tensors[0].size(0), device=self.device)
            tensors = tuple(tensor[indices] for tensor in self.tensors)
        else:
            tensors = self.tensors
        return zip(*(torch.split(tensor, self.batch_size) for tensor in tensors))


def negative_log_overlap(log_ψ: Tensor, log_φ: Tensor, log_weights: Tensor) -> Tensor:
    log_ψ, log_φ, log_weights = log_ψ.squeeze(), log_φ.squeeze(), log_weights.squeeze()
    dot_part = torch.logsumexp(log_weights + log_φ - log_ψ, dim=0)
    norm_ψ_part = torch.logsumexp(log_weights, dim=0)
    norm_φ_part = torch.logsumexp(log_weights + 2 * (log_φ - log_ψ), dim=0)
    return -dot_part + 0.5 * (norm_ψ_part + norm_φ_part)


def supervised_loop_once(
    dataset, model, optimizer, scheduler, loss_fn,
):
    tick = time.time()
    model.train()
    total_loss = 0
    total_count = 0
    for batch in dataset:
        (x, y, w) = batch
        optimizer.zero_grad()
        ŷ = model(x)
        loss = loss_fn(ŷ, y, w)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += x.size(0) * loss.item()
        total_count += x.size(0)
    tock = time.time()
    return {"loss": total_loss / total_count, "time": tock - tick}


@torch.no_grad()
def compute_average_loss(dataset, model, loss_fn):
    tick = time.time()
    model.eval()
    total_loss = 0
    total_count = 0
    for batch in dataset:
        (x, y, w) = batch
        ŷ = model(x)
        loss = loss_fn(ŷ, y, w)
        total_loss += x.size(0) * loss.item()
        total_count += x.size(0)
    tock = time.time()
    return {"loss": total_loss / total_count, "time": tock - tick}


# @torch.no_grad()
# def compute_log_target_state(
#     spins: Tensor,
#     hamiltonian,
#     roots: List[complex],
#     state: torch.nn.Module,
#     batch_size: int,
#     normalizing: bool = False,
# ) -> Tensor:
#     logger.debug("Applying polynomial using batch_size={}...", batch_size)
#     batch_size = int(batch_size)
#     if batch_size <= 0:
#         raise ValueError("invalid batch_size: {}; expected a positive integer".format(batch_size))
#     spins = as_spins_tensor(spins, force_width=True)
#     original_shape = spins.size()[:-1]
#     spins = spins.view(-1, spins.size(-1))
#     if isinstance(state, torch.jit.ScriptModule):
#         state = state._c._get_method("forward")
#     log_target = _C.log_apply_polynomial(spins, hamiltonian, roots, state, batch_size, normalizing)
#     return log_target.view(original_shape)
@torch.no_grad()
def compute_log_target_state(
    spins: Tensor, evolution_operator, state: torch.nn.Module, batch_size: int,
) -> Tensor:
    logger.debug("Applying evolution operator using batch_size={}...", batch_size)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("invalid batch_size: {}; expected a positive integer".format(batch_size))
    spins = as_spins_tensor(spins, force_width=True)
    original_shape = spins.size()[:-1]
    spins = spins.view(-1, spins.size(-1))
    if isinstance(state, torch.jit.ScriptModule):
        state = state._c._get_method("forward")
    log_target = _C.log_apply(spins, evolution_operator, state, batch_size)
    return log_target.view(original_shape)


class Runner(RunnerBase):
    def __init__(self, config):
        super().__init__(config)
        self.combined_state = combine_amplitude_and_phase(
            self.config.amplitude, self.config.phase, use_jit=False
        )

    def inner_iteration(self, states, _, weights):
        logger.info("Inner iteration...")
        tick = time.time()

        _ = self.compute_local_energies(states, weights)
        states = states.view(-1, states.size(-1))
        weights = weights.view(-1)
        log_target = compute_log_target_state(
            states,
            self.config.evolution_operator,
            self.combined_state,
            self.config.inference_batch_size,
        )

        self.train_amplitude(states, log_target, weights)
        self.checkpoint()
        tock = time.time()
        logger.info("Completed inner iteration in {:.1f} seconds!", tock - tick)

    def train_amplitude(self, states, log_target, weights):
        logger.info("Supervised training...")
        dataset = TensorIterableDataset(
            states,
            log_target.real,
            torch.log(weights),
            batch_size=self.config.training_options.batch_size,
            shuffle=True,
        )
        model = self.config.amplitude
        optimizer = self.config.training_options.optimizer
        scheduler = self.config.training_options.scheduler
        loss_fn = negative_log_overlap

        info = compute_average_loss(dataset, model, loss_fn)
        logger.debug("Initial loss: -log(⟨ψ|ϕ⟩) = {}", info["loss"])
        self.tb_writer.add_scalar("loss/negative_log_overlap", info["loss"], self.global_index)
        for epoch in range(self.config.training_options.epochs):
            info = supervised_loop_once(dataset, model, optimizer, scheduler, loss_fn)
            self.global_index += 1
            self.tb_writer.add_scalar("loss/negative_log_overlap", info["loss"], self.global_index)
        logger.debug("Final loss:   -log(⟨ψ|ϕ⟩) = {}", info["loss"])
