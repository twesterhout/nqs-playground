#!/usr/bin/env python3

# Copyright Tom Westerhout (c) 2019
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

import cmath
from collections import namedtuple
from copy import deepcopy
# import cProfile
# import importlib
# from itertools import islice
# from functools import reduce
import logging
import math
# import os
# import sys
import tempfile
import time
from typing import Dict, List, Tuple, Optional

# import click
# import mpmath  # Just to be safe: for accurate computation of L2 norms
# from numba import jit, jitclass, uint8, int64, uint64, float32
# from numba.types import Bytes
# import numba.extending
import numpy as np
# import scipy
# from scipy.sparse.linalg import lgmres, LinearOperator
import torch
import torch.utils.data
# import torch.nn as nn
# import torch.nn.functional as F

# from .core import Machine, MonteCarloState, CombiningState, CompactSpin, normalisation_constant, negative_log_overlap_real
from .monte_carlo import all_spins, sample_some
from . import _C_nqs as _C

_TrainConfig = namedtuple("_TrainConfig", ["optimiser", "loss", "epochs", "use_log", "batch_size"])

SWOConfig = namedtuple(
    "SWOConfig",
    [
        "H",
        "τ",
        "steps",
        "magnetisation",
        "lr_amplitude",
        "lr_phase",
        "epochs_amplitude",
        "epochs_phase",
    ],
)

def _make_checkpoints_for(n: int):
    if n <= 10:
        return list(range(0, n))
    important_iterations = list(range(0, n, n // 10))
    if important_iterations[-1] != n - 1:
        important_iterations.append(n - 1)
    return important_iterations

# 
# class TargetState(torch.nn.Module):
#     """
#     Represents (1 - τH)|ψ⟩.
#     """
# 
#     def __init__(self, ψ, H, τ):
#         super().__init__()
#         self._ψ = ψ if isinstance(ψ, Machine) else Machine(ψ)
#         self._ψ.clear_cache()
#         self._H = H
#         self._τ = τ
#         for p in self._ψ.parameters():
#             p.requires_grad = False
# 
#     def _forward_single(self, σ: torch.FloatTensor):
#         with torch.no_grad():
#             σ = σ.numpy()
#             E_loc = self._H(
#                 MonteCarloState(spin=σ, machine=self._ψ, weight=None), cutoff=None
#             )
#             log_wf = self._ψ.log_wf(σ) + cmath.log(1.0 - self._τ * E_loc)
#             return torch.tensor([log_wf.real, log_wf.imag], dtype=torch.float32)
# 
#     @property
#     def number_spins(self):
#         return self._ψ.number_spins
# 
#     def forward(self, x):
#         if x.dim() == 1:
#             return self._forward_single(x)
#         else:
#             assert x.dim() == 2
#             out = torch.empty((x.size(0), 2), dtype=torch.float32, requires_grad=False)
#             for i in range(x.size(0)):
#                 out[i] = self._forward_single(x[i])
#             return out
# 

class CombiningState(torch.jit.ScriptModule):
    def __init__(self, amplitude, phase):
        super().__init__()
        self._amplitude = amplitude
        self._phase = phase

    @property
    def number_spins(self):
        return self._amplitude.number_spins

    @torch.jit.script_method
    def forward(self, x):
        A = self._amplitude(x)
        _, phi = torch.max(self._phase(x), dim=1, keepdim=True)
        phi = 1 - 2 * phi
        return A * phi.float()

FastCombiningState = CombiningState

# 
# class FastTargetState(torch.nn.Module):
#     """
#     Represents (H - c0)(H - c1)...(H - cN-1)|ψ⟩.
#     """
# 
#     def __init__(self, amplitude, phase, poly):
#         super().__init__()
#         self._amplitude = amplitude
#         self._phase = phase
#         self._poly = poly
#         for p in self._amplitude.parameters():
#             p.requires_grad = False
#         for p in self._phase.parameters():
#             p.requires_grad = False
# 
#     def _sum_logs(self, coeffs, xs):
#         assert not torch.any(torch.isnan(coeffs))
#         assert not torch.any(torch.isnan(xs))
#         scale = -torch.max(xs).item() + 3.0
#         xs = torch.dot(coeffs, torch.exp(xs + scale)).item()
#         if xs > 0:
#             return torch.tensor([math.log(xs) - scale, 0.0],
#                     dtype=torch.float32)
#         else:
#             return torch.tensor([math.log(-xs) - scale, math.pi],
#                     dtype=torch.float32)
# 
#     def _from_items(self, keys, values):
#         values[torch.max(self._phase(keys), dim=1)[1].byte()] *= -1
#         return self._sum_logs(values, self._amplitude(keys).view(-1))
# 
#     def _forward_single(self, σ: torch.FloatTensor):
#         with torch.no_grad():
#             σ = σ.numpy()
#             self._poly(1, CompactSpin(σ))
#             keys = torch.empty((self._poly.size, σ.size), dtype=torch.float32)
#             values = torch.empty((self._poly.size,), dtype=torch.float32)
#             self._poly.keys(keys.numpy())
#             self._poly.values(values.numpy())
#             # temp = np.empty((self._poly.size,), dtype=np.complex64)
#             # self._poly.values(temp)
#             # print(temp)
#             # print(values)
#             return self._from_items(keys, values)
# 
#     @property
#     def number_spins(self):
#         return self._amplitude.number_spins
# 
#     def forward(self, x):
#         if x.dim() == 1:
#             return self._forward_single(x)
#         else:
#             assert x.dim() == 2
#             out = torch.empty((x.size(0), 2), dtype=torch.float32, requires_grad=False)
#             for i in range(x.size(0)):
#                 out[i] = self._forward_single(x[i])
#             return out
# 
# class NewTargetState(torch.nn.Module):
#     """
#     Represents (H - c0)(H - c1)...(H - cN-1)|ψ⟩.
#     """
# 
#     def __init__(self, amplitude, phase, poly):
#         super().__init__()
#         self._amplitude = amplitude
#         self._phase = phase
#         self._poly = poly
# 
#     def _from_items(self, keys, values):
#         A = self._amplitude(keys).view([-1])
#         _, phi = torch.max(self._phase(keys), dim=1)
#         return torch.dot(values, torch.mul(A, 1 - 2 * phi.float()))
#         # values[torch.max(self._phase(keys), dim=1)[1].byte()] *= -1
#         # return torch.dot(values, self._amplitude(keys).view(-1))
# 
#     def _forward_single(self, σ: torch.FloatTensor):
#         with torch.no_grad():
#             σ = σ.numpy()
#             self._poly(1, CompactSpin(σ))
#             keys = self._poly.keys()
#             values = self._poly.values()
#             return self._from_items(keys, values)
# 
#     @property
#     def number_spins(self):
#         return self._amplitude.number_spins
# 
#     def forward(self, x):
#         if x.dim() == 1:
#             return self._forward_single(x)
#         else:
#             assert x.dim() == 2
#             out = torch.empty(x.size(0), dtype=torch.float32, requires_grad=False)
#             for i in range(x.size(0)):
#                 out[i] = self._forward_single(x[i])
#             return out
# 
# class NewerTargetState(torch.nn.Module):
#     """
#     Represents (H - c0)(H - c1)...(H - cN-1)|ψ⟩.
#     """
# 
#     def __init__(self, psi, poly):
#         super().__init__()
#         self._psi = psi
#         self._poly = poly
# 
#     def _from_items(self, keys, values):
#         return torch.dot(values, self._psi(keys).view([-1]))
# 
#     def _forward_single(self, σ: torch.FloatTensor):
#         with torch.no_grad():
#             σ = σ.numpy()
#             self._poly(1, CompactSpin(σ))
#             keys = self._poly.keys()
#             values = self._poly.values()
#             return self._from_items(keys, values)
# 
#     @property
#     def number_spins(self):
#         return self._amplitude.number_spins
# 
#     def forward(self, x):
#         if x.dim() == 1:
#             return self._forward_single(x)
#         else:
#             assert x.dim() == 2
#             out = torch.empty(x.size(0), dtype=torch.float32, requires_grad=False)
#             for i in range(x.size(0)):
#                 out[i] = self._forward_single(x[i])
#             return out
# 

def _train_amplitude(ψ: torch.nn.Module, dataset: torch.utils.data.Dataset, config):
    logging.info("Learning amplitudes...")
    start = time.time()

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    optimiser = config["optimiser"](ψ.parameters())
    loss_fn = config["loss"]

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=True)
    checkpoints = set(_make_checkpoints_for(epochs))
    for i in range(epochs):
        losses = []
        for samples, target in dataloader:
            optimiser.zero_grad()
            loss = loss_fn(ψ(samples), target)
            losses.append(loss.item())
            loss.backward()
            optimiser.step()
        if i in checkpoints:
            losses = torch.tensor(losses)
            logging.info("{}%: loss = {:.5e} ± {:.2e}; loss ∈ [{:.5e}, {:.5e}]".format(
                100 * (i + 1) // epochs, torch.mean(losses).item(), torch.std(losses).item(),
                torch.min(losses).item(), torch.max(losses).item()))

    finish = time.time()
    logging.info("Done in {:.2f} seconds!".format(finish - start))
    return ψ


def _train_phase(ψ: torch.nn.Module, dataset: torch.utils.data.Dataset, config):
    logging.info("Learning phases...")
    start = time.time()

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    optimiser = config["optimiser"](ψ.parameters())
    loss_fn = config["loss"]

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=True)
    checkpoints = set(_make_checkpoints_for(epochs))

    indices = torch.arange(len(dataset))
    samples_whole, target_whole = dataset[indices]
    del indices

    def accuracy():
        with torch.no_grad():
            _, predicted = torch.max(ψ(samples_whole), dim=1)
            return float(torch.sum(target_whole == predicted)) / target_whole.size(0)

    logging.info("Initial accuracy: {:.2f}%".format(100 * accuracy()))
    for i in range(epochs):
        losses = []
        for samples, target in dataloader:
            optimiser.zero_grad()
            loss = loss_fn(ψ(samples), target)
            losses.append(loss.item())
            loss.backward()
            optimiser.step()
        if i in checkpoints:
            losses = torch.tensor(losses)
            logging.info("{}%: loss = {:.5e} ± {:.2e}; loss ∈ [{:.5e}, {:.5e}]".format(
                100 * (i + 1) // epochs, torch.mean(losses).item(), torch.std(losses).item(),
                torch.min(losses).item(), torch.max(losses).item()))
    logging.info("Final accuracy: {:.2f}%".format(100 * accuracy()))

    finish = time.time()
    logging.info("Done in {:.2f} seconds!".format(finish - start))
    return ψ
   
# 
# def _train_amplitude(ψ: torch.nn.Module, target: torch.FloatTensor, samples, config):
#     logging.info("Learning amplitudes...")
#     start = time.time()
# 
#     checkpoints = set(_make_checkpoints_for(config.epochs))
#     optimiser = config.optimiser(ψ.parameters())
#     for i in range(config.epochs):
#         indices = torch.randperm(samples.size(0))
#         for batch in torch.split(indices, config.batch_size):
#             optimiser.zero_grad()
#             predicted = ψ(samples[batch]) #.view(-1)
#             if not config.use_log:
#                 with torch.no_grad():
#                     scale = normalisation_constant(predicted.detach()).item()
#                 assert not torch.any(torch.isnan(predicted))
#                 assert not torch.any(torch.isnan(torch.exp(scale + predicted)))
#                 loss = config.loss(torch.exp(scale + predicted), target[batch])
#             else:
#                 loss = config.loss(predicted, target[batch])
#             loss.backward()
#             optimiser.step()
#         if i in checkpoints:
#             with torch.no_grad():
#                 predicted = ψ(samples) #.view(-1)
#                 loss = config.loss(predicted, target)
#             logging.info("{}%: Loss = {}".format(100 * (i + 1) // config.epochs, loss))
# 
#     finish = time.time()
#     logging.info("Done in {:.2f} seconds!".format(finish - start))
#     return ψ
# 
# 
# def _train_phase(ψ: torch.nn.Module, target: torch.LongTensor, samples, config):
#     logging.info("Learning phases...")
#     start = time.time()
# 
#     checkpoints = set(_make_checkpoints_for(config.epochs))
#     optimiser = config.optimiser(ψ.parameters())
# 
#     def accuracy():
#         with torch.no_grad():
#             _, predicted = torch.max(ψ(samples), dim=1)
#             return float(torch.sum(target == predicted)) / target.size(0)
# 
#     logging.info("Initial accuracy: {:.2f}%".format(100 * accuracy()))
#     for i in range(config.epochs):
#         indices = torch.randperm(samples.size(0))
#         for batch in torch.split(indices, config.batch_size):
#             optimiser.zero_grad()
#             loss = config.loss(ψ(samples[batch]), target[batch])
#             loss.backward()
#             optimiser.step()
#         if i in checkpoints:
#             with torch.no_grad():
#                 loss = config.loss(ψ(samples), target)
#             logging.info("{}%: Loss = {}".format(100 * (i + 1) // config.epochs, loss))
#     logging.info("Final accuracy: {:.2f}%".format(100 * accuracy()))
# 
#     finish = time.time()
#     logging.info("Done in {:.2f} seconds!".format(finish - start))
#     return ψ
# 
#
# @torch.jit.script
# def _make_target_amplitudes(log_phi):
#     with torch.no_grad():
#         xs = log_phi[:, 0]
#         scale = normalisation_constant(xs)
#         return torch.exp(xs + scale)
# 
# 
# @torch.jit.script
# def _make_target_phases(log_phi):
#     with torch.no_grad():
#         xs = log_phi[:, 1]
#         return torch.fmod(
#             torch.round(torch.abs(torch.div(xs, math.pi))).long(), 2
#         )


def swo_step(ψ, config):
    ψ_amplitude, ψ_phase = ψ
    # φ = TargetState(
    #     CombiningState(deepcopy(ψ_amplitude), deepcopy(ψ_phase)), config.H, config.τ
    # )
    # poly = _C.Polynomial(_C.Heisenberg(config.H._graph, 1.0 / 10.0),
    #     [complex(0.2705557689322943, -2.5047759043624347),
    #      complex(0.2705557689322943, +2.5047759043624347),
    #      complex(1.7294442310677054, -0.8889743761218659),
    #      complex(1.7294442310677054, +0.8889743761218659),
    #     ])
    G = config["hamiltonian"]._graph
    J = 1.0 / len(G)
    roots = config["roots"]
    poly = _C.Polynomial(_C.Heisenberg(edges=G, coupling=J), roots)
    number_spins = config["hamiltonian"].number_spins
    magnetisation = config["magnetisation"]

    # φ = FastTargetState(
    #     deepcopy(ψ_amplitude), deepcopy(ψ_phase), poly
    # )
    # temp = sample_state(φ, H=config.H, magnetisation=0, explicit=True, requires_energy=True)
    # print(torch.mm(temp[0].weights.view(1, -1), temp[0].energies.view(-1, 2)))
    explicit = True # False

    logging.info("Generating the training data set...")
    if explicit:
        with torch.no_grad():
            samples = all_spins(number_spins, magnetisation)
            tempfile_name = ".swo.model.temp"
            FastCombiningState(ψ_amplitude, ψ_phase).save(tempfile_name)
            φ = _C.TargetState(tempfile_name, poly, 8192)
            φ_s_ = φ(samples)
            target_amplitudes = torch.abs(φ_s_)
            target_phases = torch.where(φ_s_ >= 0.0, torch.tensor([0]), torch.tensor([1]))
            del φ_s_
    else:
        raise NotImplementedError()

    logging.info("Training on {} spin configurations...".format(samples.size(0)))

    ψ_amplitude = _train_amplitude(
        ψ_amplitude,
        torch.utils.data.TensorDataset(samples, target_amplitudes),
        config["amplitude"],
        # _TrainConfig(
        #     optimiser=lambda p: torch.optim.Adam(p, lr=config.lr_amplitude),
        #     loss=lambda x, y: negative_log_overlap_real(x.view(-1), y),
        #     epochs=config.epochs_amplitude,
        #     use_log=True,
        #     batch_size=256,
        # ),
    )
    ψ_phase = _train_phase(
        ψ_phase,
        torch.utils.data.TensorDataset(samples, target_phases),
        config["phase"],
        # _TrainConfig(
        #     optimiser=lambda p: torch.optim.Adam(p, lr=config.lr_phase),
        #     loss=torch.nn.CrossEntropyLoss(),
        #     epochs=config.epochs_phase,
        #     use_log=None,
        #     batch_size=256,
        # ),
    )
    return ψ_amplitude, ψ_phase


