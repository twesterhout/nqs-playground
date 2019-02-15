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

# import cmath
# import collections
# from copy import deepcopy
import cProfile
import importlib
# from itertools import islice
# from functools import reduce
import logging
# import math
import os
import sys
# import time
# from typing import Dict, List, Tuple, Optional

import click
# import mpmath  # Just to be safe: for accurate computation of L2 norms
# from numba import jit, jitclass, uint8, int64, uint64, float32
# from numba.types import Bytes
# import numba.extending
# import numpy as np
# import scipy
# from scipy.sparse.linalg import lgmres, LinearOperator
import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from ._C_nqs import CompactSpin

from .hamiltonian import read_hamiltonian
from .swo import SWOConfig, swo_step


def import_network(nn_file: str):
    module_name, extension = os.path.splitext(os.path.basename(nn_file))
    module_dir = os.path.dirname(nn_file)
    if extension != ".py":
        raise ValueError(
            "Could not import the network from '{}': not a python source file.".format(nn_file)
        )
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    sys.path.pop(0)
    return module.Net


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "amplitude-nn-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar="<amplitude torch.nn.Module>",
)
@click.argument(
    "phase-nn-file",
    type=click.Path(exists=True, resolve_path=True, path_type=str),
    metavar="<phase torch.nn.Module>",
)
@click.option(
    "--in-amplitude",
    type=click.File(mode="rb"),
    help="Pickled initial `state_dict` of the neural net predicting log amplitudes.",
)
@click.option(
    "--in-phase",
    type=click.File(mode="rb"),
    help="Pickled initial `state_dict` of the neural net predicting phases.",
)
@click.option("-o", "--out-file", type=str, help="Basename for output files.")
@click.option(
    "-H",
    "--hamiltonian",
    "hamiltonian_file",
    type=click.File(mode="r"),
    required=True,
    help="File containing the Heisenberg Hamiltonian specifications.",
)
@click.option(
    "--epochs-outer",
    type=click.IntRange(min=0),
    default=200,
    show_default=True,
    help="Number of outer learning steps (i.e. Lanczos steps) to perform.",
)
@click.option(
    "--epochs-amplitude",
    type=click.IntRange(min=0),
    default=2000,
    show_default=True,
    help="Number of epochs when learning the amplitude",
)
@click.option(
    "--epochs-phase",
    type=click.IntRange(min=0),
    default=2000,
    show_default=True,
    help="Number of epochs when learning the phase",
)
@click.option(
    "--lr-amplitude",
    type=click.FloatRange(min=1.0e-10),
    default=0.05,
    show_default=True,
    help="Learning rate for training the amplitude net.",
)
@click.option(
    "--lr-phase",
    type=click.FloatRange(min=1.0e-10),
    default=0.05,
    show_default=True,
    help="Learning rate for training the phase net.",
)
@click.option(
    "--tau",
    type=click.FloatRange(min=1.0e-10),
    default=0.2,
    show_default=True,
    help="τ",
)
@click.option(
    "--steps",
    type=click.IntRange(min=1),
    default=2000,
    show_default=True,
    help="Length of the Markov Chain.",
)
def swo(
    amplitude_nn_file,
    phase_nn_file,
    in_amplitude,
    in_phase,
    out_file,
    hamiltonian_file,
    epochs_outer,
    epochs_amplitude,
    epochs_phase,
    lr_amplitude,
    lr_phase,
    tau,
    steps,
):
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )
    H = read_hamiltonian(hamiltonian_file)
    number_spins = H.number_spins
    magnetisation = 0 if number_spins % 2 == 0 else 1

    thermalisation = steps // 10
    m_c_steps = (
        thermalisation * number_spins,
        (thermalisation + steps) * number_spins,
        1,
    )

    ψ_amplitude = import_network(amplitude_nn_file)(number_spins)
    if in_amplitude is not None:
        logging.info("Reading initial state...")
        ψ_amplitude.load_state_dict(torch.load(in_amplitude))
    else:
        logging.info("Using random weights...")

    ψ_phase = import_network(phase_nn_file)(number_spins)
    if in_phase is not None:
        logging.info("Reading initial state...")
        ψ_phase.load_state_dict(torch.load(in_phase))
    else:
        logging.info("Using random weights...")

    for i in range(epochs_outer):
        logging.info("\t#{}".format(i))
        swo_step(
            (ψ_amplitude, ψ_phase),
            SWOConfig(
                H=H,
                τ=tau,
                steps=(5,) + m_c_steps,
                magnetisation=magnetisation,
                lr_amplitude=lr_amplitude,
                lr_phase=lr_phase,
                epochs_amplitude=epochs_amplitude,
                epochs_phase=epochs_phase,
            ),
        )
        torch.save(ψ_amplitude.state_dict(), "{}.{}.amplitude.weights".format(out_file, i))
        torch.save(ψ_phase.state_dict(), "{}.{}.phase.weights".format(out_file, i))


if __name__ == "__main__":
    cli()
    # cProfile.run('cli()')
