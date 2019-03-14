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
import importlib

# from itertools import islice
# from functools import reduce
import logging
import math
import os
import sys
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
from torch.utils.data import TensorDataset, DataLoader

# import torch.nn as nn
# import torch.nn.functional as F

from .core import CompactSpin, negative_log_overlap_real, import_network
from .monte_carlo import all_spins
from .hamiltonian import read_hamiltonian
from . import _C_nqs as _C


class CombiningState(torch.jit.ScriptModule):
    """
    Given a regressor ``amplitude`` predicting wave function amplitude and a
    classifier ``phase`` predicting the signs (the first class means ``+`` and
    the second -- ``-``), constructs a regressor predicting the whole wave
    function.
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
        A = self._amplitude(x)
        _, phi = torch.max(self._phase(x), dim=1, keepdim=True)
        phi = 1 - 2 * phi
        return A * phi.float()


def _make_checkpoints_for(n: int):
    if n <= 10:
        return list(range(0, n))
    important_iterations = list(range(0, n, n // 10))
    if important_iterations[-1] != n - 1:
        important_iterations.append(n - 1)
    return important_iterations


def train_amplitude(ψ: torch.nn.Module, dataset: torch.utils.data.Dataset, config):
    logging.info("Learning amplitudes...")
    start = time.time()

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    optimiser = config["optimiser"](ψ.parameters())
    loss_fn = config["loss"]

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
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
            logging.info(
                "{}%: loss = {:.5e} ± {:.2e}; loss ∈ [{:.5e}, {:.5e}]".format(
                    100 * (i + 1) // epochs,
                    torch.mean(losses).item(),
                    torch.std(losses).item(),
                    torch.min(losses).item(),
                    torch.max(losses).item(),
                )
            )

    finish = time.time()
    logging.info("Done in {:.2f} seconds!".format(finish - start))
    return ψ


def train_phase(ψ: torch.nn.Module, dataset: torch.utils.data.Dataset, config):
    logging.info("Learning phases...")
    start = time.time()

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    optimiser = config["optimiser"](ψ.parameters())
    loss_fn = config["loss"]

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
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
            logging.info(
                "{}%: loss = {:.5e} ± {:.2e}; loss ∈ [{:.5e}, {:.5e}]".format(
                    100 * (i + 1) // epochs,
                    torch.mean(losses).item(),
                    torch.std(losses).item(),
                    torch.min(losses).item(),
                    torch.max(losses).item(),
                )
            )
    logging.info("Final accuracy: {:.2f}%".format(100 * accuracy()))

    finish = time.time()
    logging.info("Done in {:.2f} seconds!".format(finish - start))
    return ψ


def generate_train_data(filename, config):
    explicit = False # True
    number_spins = config["number_spins"]
    magnetisation = config["magnetisation"]
    poly = _C.Polynomial(config["hamiltonian"], config["roots"])
    if explicit:
        if (
            "cache" not in generate_train_data.__dict__
            or generate_train_data.cache[0] != number_spins
            or generate_train_data.cache[1] != magnetisation
        ):
            generate_train_data.cache = (
                number_spins,
                magnetisation,
                all_spins(number_spins, magnetisation),
            )

        samples = generate_train_data.cache[2]
        φ = _C.TargetState(filename, poly, (8192, number_spins))(samples)
        target_amplitudes = torch.abs(φ)
        target_signs = torch.where(φ >= 0.0, torch.tensor([0]), torch.tensor([1]))
    else:
        chain_options = _C.ChainOptions(number_spins=number_spins, magnetisation=magnetisation,
                batch_size=64, steps=config["steps"])
        samples, φ, _ = _C.sample_some(filename, poly, chain_options, (8, 2))
        target_amplitudes = torch.abs(φ)
        target_signs = torch.where(φ >= 0.0, torch.tensor([0]), torch.tensor([1]))
    return samples, target_amplitudes, target_signs


def swo_step(ψ, config):
    ψ_amplitude, ψ_phase = ψ
    tempfile_name = ".swo.model.temp"
    CombiningState(ψ_amplitude, ψ_phase).save(tempfile_name)

    logging.info("Generating the training data set...")
    samples, target_amplitudes, target_phases = generate_train_data(
        tempfile_name, config
    )

    logging.info("Training on {} spin configurations...".format(samples.size(0)))
    ψ_amplitude = train_amplitude(
        ψ_amplitude, TensorDataset(samples, target_amplitudes), config["amplitude"]
    )
    ψ_phase = train_phase(
        ψ_phase, TensorDataset(samples, target_phases), config["phase"]
    )
    return ψ_amplitude, ψ_phase


AmplitudeNet = import_network("small.py")

PhaseNet = import_network("phase.py")

_OPTIONS = {
    "number_spins": 10,
    "magnetisation": 0,
    "hamiltonian": read_hamiltonian("data/1x10.hamiltonian").to_cxx(),
    "roots": [(-1 - 1j, None), (1 + 1j, None)],
    "epochs": 20,
    "output": "result/swo",
    "steps": (8, 50, 80, 1),
    "amplitude": {
        "optimiser": lambda p: torch.optim.Adam(p, lr=0.00025),
        "epochs": 200,
        "batch_size": 32,
        "loss": lambda x, y: negative_log_overlap_real(x, y),
    },
    "phase": {
        "optimiser": lambda p: torch.optim.Adam(p, lr=0.003),
        "epochs": 100,
        "batch_size": 64,
        "loss": torch.nn.CrossEntropyLoss(),
    },
}


def main():
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )

    number_spins = _OPTIONS["number_spins"]
    ψ_amplitude = AmplitudeNet(number_spins)
    ψ_phase = PhaseNet(number_spins)
    for i in range(_OPTIONS["epochs"]):
        logging.info("-" * 10 + str(i + 1) + "-" * 10)
        ψ_amplitude, ψ_phase = swo_step((ψ_amplitude, ψ_phase), _OPTIONS)
        CombiningState(ψ_amplitude, ψ_phase).save(
            "{}.model.{}.pt".format(_OPTIONS["output"], i + 1)
        )


if __name__ == "__main__":
    main()
