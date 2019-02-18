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
import itertools
# from functools import reduce
import logging
import math

import os

# import sys
import tempfile
import time
from typing import Dict, List, Tuple, Optional

import psutil

# import click
# import mpmath  # Just to be safe: for accurate computation of L2 norms
# from numba import jit, jitclass, uint8, int64, uint64, float32
# from numba.types import Bytes
# import numba.extending
# import numpy as np

# import scipy
# from scipy.sparse.linalg import lgmres, LinearOperator
import torch
import torch.utils.data
from torch.multiprocessing import Process, SimpleQueue

torch.multiprocessing.set_sharing_strategy("file_system")

# import torch.nn as nn
# import torch.nn.functional as F

from .core import MonteCarloState, negative_log_overlap_real
from .monte_carlo import all_spins, sample_some
from . import _C_nqs as _C
from .spawn import spawn

# _TrainConfig = namedtuple(
#     "_TrainConfig", ["optimiser", "loss", "epochs", "use_log", "batch_size"]
# )

# SWOConfig = namedtuple(
#     "SWOConfig",
#     [
#         "H",
#         "τ",
#         "steps",
#         "magnetisation",
#         "lr_amplitude",
#         "lr_phase",
#         "epochs_amplitude",
#         "epochs_phase",
#     ],
# )

ProcessorStats = namedtuple(
    "ProcessorStats", ["processor", "socket", "core"]
)

def get_cpuinfo():
    def parse_one(record):
        info = {}

        def extract_field(field: str, line: str):
            assert field not in info
            info[field] = int(line.split(":")[-1])

        for line in record:
            if line.startswith("processor"):
                extract_field("processor", line)
            elif line.startswith("physical id"):
                extract_field("socket", line)
            elif line.startswith("core id"):
                extract_field("core", line)
        return ProcessorStats(**info)

    with open("/proc/cpuinfo", "r") as cpuinfo_file:
        cpuinfo = [
            parse_one(g)
            for _, g in filter(
                lambda tpl: tpl[0] == True,
                itertools.groupby(cpuinfo_file, key=lambda l: l != "\n"),
            )
        ]
        return cpuinfo


def split_between_two():
    info = get_cpuinfo()
    number_sockets = max(map(lambda record: record.socket, info)) + 1
    if number_sockets == 1:
        assert (max(map(lambda record: record.core, info)) + 1) % 2 == 0
        cores0 = list(
            map(
                lambda record: record.processor,
                filter(lambda record: record.core % 2 == 0, info),
            )
        )
        cores1 = list(
            map(
                lambda record: record.processor,
                filter(lambda record: record.core % 2 != 0, info),
            )
        )
        return cores0, cores1
    elif number_sockets == 2:
        cores0 = list(
            map(
                lambda record: record.processor,
                filter(lambda record: record.socket == 0, info),
            )
        )
        cores1 = list(
            map(
                lambda record: record.processor,
                filter(lambda record: record.socket == 1, info),
            )
        )
        return cores0, cores1
    else:
        raise RuntimeError(
            "{} sockets?! I haven't considered such an architecture yet".format(
                number_sockets
            )
        )


def _make_checkpoints_for(n: int) -> List[int]:
    """
    If you have a task consisting of ``n`` steps, ``_make_checkpoints_for(n)``
    gives you indices of steps marking 0%, 10%, 20%, ... completion.
    """
    if n <= 10:
        return list(range(0, n))
    important_iterations = list(range(0, n, n // 10))
    if important_iterations[-1] != n - 1:
        important_iterations.append(n - 1)
    return important_iterations


class FastCombiningState(torch.jit.ScriptModule):
    """
    An NQS consisting of two neural networks: one predicting amplitudes and the
    second -- signs (i.e. we work under the assumption that the ground state is
    real).
    """

    def __init__(self, amplitude, phase):
        super().__init__()
        self._amplitude = amplitude
        self._phase = phase

    @property
    def number_spins(self):
        return self._amplitude.number_spins

    @torch.jit.script_method
    def forward(self, x):
        amplitude = self._amplitude(x)
        _, sign = torch.max(self._phase(x), dim=1, keepdim=True)
        sign = 1 - 2 * sign
        return amplitude * sign.float()


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


def _train_amplitude(ψ: torch.nn.Module, target, config):
    start = time.time()
    pid = os.getpid()
    logging.info("Learning amplitudes [pid={}]...".format(pid))

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    checkpoints = set(_make_checkpoints_for(epochs))
    optimiser = config["optimiser"](ψ.parameters())
    loss_fn = config["loss"]

    samples, _ = target
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*target), batch_size=batch_size, shuffle=True
    )

    start_useless = time.time()
    log_messages = ["Amplitudes' training report:"]
    with torch.no_grad():
        loss = loss_fn(ψ(target[0]), target[1]).item()
        log_messages.append("Initial loss: {:.5f}".format(loss))
    useless_total = time.time() - start_useless

    for i in range(epochs):
        losses = []
        for x, ŷ in dataloader:
            optimiser.zero_grad()
            loss = loss_fn(ψ(x), ŷ)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        if i in checkpoints:
            log_messages.append(
                "{}%: Losses: {}".format(100 * (i + 1) // epochs, losses)
            )

    start_useless = time.time()
    with torch.no_grad():
        loss = loss_fn(ψ(target[0]), target[1]).item()
        log_messages.append("Final loss: {:.5f}".format(loss))
    useless_total += time.time() - start_useless

    finish = time.time()
    log_messages.append(
        "Done in {:.2f} seconds ({}% spent on useful work)!".format(
            finish - start, int(100 * (1 - useless_total / (finish - start)))
        )
    )
    logging.info("\n".join(log_messages))
    return ψ


def _train_phase(ψ: torch.nn.Module, target, config):
    start = time.time()
    pid = os.getpid()
    logging.info("Learning phases [pid={}]...".format(pid))

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    checkpoints = set(_make_checkpoints_for(epochs))
    optimiser = config["optimiser"](ψ.parameters())
    loss_fn = config["loss"]

    samples, _ = target
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*target), batch_size=batch_size, shuffle=True
    )

    def accuracy():
        with torch.no_grad():
            _, predicted = torch.max(ψ(target[0]), dim=1)
            return float(torch.sum(target[1] == predicted)) / target[0].size(0)

    start_useless = time.time()
    log_messages = ["Phases' training report:"]
    with torch.no_grad():
        log_messages.append("Initial accuracy: {:.2f}%".format(100 * accuracy()))
    useless_total = time.time() - start_useless

    for i in range(epochs):
        losses = []
        for x, ŷ in dataloader:
            optimiser.zero_grad()
            loss = loss_fn(ψ(x), ŷ)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        if i in checkpoints:
            log_messages.append(
                "{}%: Losses: {}".format(100 * (i + 1) // epochs, losses)
            )

    start_useless = time.time()
    with torch.no_grad():
        log_messages.append("Final accuracy: {:.2f}%".format(100 * accuracy()))
    useless_total += time.time() - start_useless

    finish = time.time()
    log_messages.append(
        "Done in {:.2f} seconds ({}% spent on useful work)!".format(
            finish - start, int(100 * (1 - useless_total / (finish - start)))
        )
    )
    logging.info("\n".join(log_messages))
    return ψ


def init_processes(rank, size, fn, *args):
    """ Initialize the distributed environment. """
    # os.environ["MASTER_ADDR"] = "127.0.0.1"
    # os.environ["MASTER_PORT"] = "29532"
    # dist.init_process_group("gloo", rank=rank, world_size=size)
    fn(*args)
    # dist.destroy_process_group()


# if __name__ == "__main__":
#     size = 2
#     processes = []
#     affinities = list(split_between_two())
#     for rank in range(size):
#         p = Process(target=init_processes, args=(rank, size, run, affinities[rank]))
#         p.start()
#         processes.append(p)
#
#     for p in processes:
#         p.join()


# @torch.jit.script
# def _make_target_amplitudes(log_phi):
#     with torch.no_grad():
#         xs = log_phi[:, 0]
#         scale = normalisation_constant(xs)
#         return torch.exp(xs + scale)


# @torch.jit.script
# def _make_target_phases(log_phi):
#     with torch.no_grad():
#         xs = log_phi[:, 1]
#         return torch.fmod(
#             torch.round(torch.abs(torch.div(xs, math.pi))).long(), 2
#         )


def _generate_training_data_set(i, queue, ψ, poly, batch_size=8192, explicit=True):
    assert i == 0
    assert explicit
    with torch.no_grad():
        ψ = FastCombiningState(*ψ)
        samples = all_spins(ψ.number_spins, ψ.number_spins % 2)
        φ = _C.TargetState(_C.Machine(ψ), poly, batch_size=batch_size)
        target = φ.forward(samples)
        queue.put(samples)
        queue.put(target)


def _train(i, networks, target, config):
    assert i in [0, 1]
    psutil.Process().cpu_affinity(config["affinity"][i])
    if i == 0:
        # Training amplitudes
        ψ, _ = networks
        samples, φ = target
        amplitudes = torch.abs(φ)
        _train_amplitude(ψ, (samples, amplitudes), config["amplitude"])
    elif i == 1:
        # Training phases
        _, ψ = networks
        samples, φ = target
        n = samples.size(0)
        signs = torch.where(
            φ > 0,
            torch.zeros((n,), dtype=torch.int64),
            torch.ones((n,), dtype=torch.int64),
        )
        _train_phase(ψ, (samples, signs), config["phase"])


def swo_step(ψ, config):
    ψ_amplitude, ψ_phase = ψ
    ψ_amplitude.share_memory()
    ψ_phase.share_memory()

    # poly = _C.Polynomial(_C.Heisenberg(config.H._graph, 1.0 / 10.0),
    #     [complex(0.2705557689322943, -2.5047759043624347),
    #      complex(0.2705557689322943, +2.5047759043624347),
    #      complex(1.7294442310677054, -0.8889743761218659),
    #      complex(1.7294442310677054, +0.8889743761218659),
    #     ])
    poly = _C.Polynomial(
        _C.Heisenberg(
            config["hamiltonian"]._graph, 1.0 / len(config["hamiltonian"]._graph)
        ),
        [1, 1] * 2,
    )
    explicit = True

    logging.info("Generating the training data set...")
    if explicit:
        queue = SimpleQueue()
        spawn(
            _generate_training_data_set,
            args=(queue, (ψ_amplitude, ψ_phase), poly),
            nprocs=1,
            join=True,
        )
        samples = queue.get()
        target = queue.get()
    else:
        raise NotImplementedError()

    logging.info("Training on {} spin configurations...".format(samples.size(0)))
    spawn(
        _train,
        args=((ψ_amplitude, ψ_phase), (samples, target), config),
        nprocs=2,
        join=True,
    )
    return ψ_amplitude, ψ_phase
