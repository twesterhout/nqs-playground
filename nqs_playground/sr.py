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

import argparse
from collections import namedtuple
import json
import math
import os
import pickle
import sys
import tempfile
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import scipy
import torch
from torch.utils.tensorboard import SummaryWriter

from . import core, hamiltonian
from .core import _C


def num_parameters(module: torch.nn.Module) -> int:
    r"""Given a ``torch.nn.Module``, returns total number of parameters in it.
    """
    return sum(map(torch.numel, module.parameters()))


def jacobian(module: torch.nn.Module, inputs: torch.Tensor, out=None) -> torch.Tensor:
    r"""Given a ``torch.nn.Module`` and a ``torch.Tensor`` of inputs, computes
    the Jacobian ∂module(inputs)/∂W where W are module's parameters.

    It is assumed that if ``inputs`` has shape ``(batch_size, in_features)``,
    then ``module(inputs)`` has shape ``(batch_size, 1)``.
    """
    parameters = list(module.parameters())
    shape = (inputs.size(0), num_parameters(module))
    if out is None:
        out = torch.zeros(*shape)
    else:
        assert out.size() == shape
    for i, xs in enumerate(inputs):
        dws = torch.autograd.grad(
            module(xs.view(1, -1)),
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        torch.cat([dw.reshape(-1) for dw in dws], out=out[i])
    return out


def logarithmic_derivative(
    modules: Tuple[torch.nn.Module, torch.nn.Module], inputs: torch.Tensor
) -> np.ndarray:
    r"""Computes ``∂log(ψ(inputs))/∂W``.

    Wavefunction ψ is represented by two ``torch.nn.Module``s: amplitude and
    phase. Both modules, given a tensor of shape ``(batch_size, in_features)``
    shoule return a tensor of shape ``(batch_size, 1)``. Output of the
    amplitude module is interpreted as ``Re[log(ψ(inputs))]`` (i.e. logarithm
    of the amplitudes of the wavefunction) and output of the phase module is
    interpreted as ``Im[log(ψ(inputs))]`` (i.e. phases of the coefficients of
    the wavefunction).

    :return: a complex tensor of shape ``(#inputs, #parameters)``.
    """
    amplitude, phase = modules
    middle = num_parameters(amplitude)
    n = middle + num_parameters(phase)
    out = torch.zeros([inputs.size(0), n, 2], dtype=torch.float32)
    jacobian(amplitude, inputs, out[:, :middle, 0])
    jacobian(phase, inputs, out[:, middle:, 1])
    return out.numpy().view(np.complex64).squeeze(axis=2)


def energy_gradient(
    local_energies: np.ndarray,
    logarithmic_derivatives: np.ndarray,
    weights: np.ndarray = None,
) -> np.ndarray:
    r"""Calculates the gradient of energy with respect to variational
    parameters: ``∂⟨ψ|H|ψ⟩/∂W``.

    :param local_energies: local energy estimators ``⟨σ|H|ψ⟩/⟨σ|ψ⟩``.
    :param logarithmic_derivatives: **centered** logarithmic derivatives of the
        wavefunction with respect to variational parameters.
    :param weights: if specified, it is assumed that {σ} span the whole
        Hilbert space basis. Then ``weights`` are **normalized** probabilities
        ``|⟨σ|ψ⟩|²/‖ψ‖₂``. If ``weights`` is ``None``, then it is assumed that
        {σ} come from Monte Carlo sampling and are distributed according to
        ``|⟨σ|ψ⟩|²/‖ψ‖₂``.
    """
    if weights is not None:
        assert np.isclose(np.sum(weights), 1.0)
        local_energies = weights * local_energies
        local_energies = local_energies.conj()
        local_energies = local_energies.reshape(1, -1)
        gradient = local_energies @ logarithmic_derivatives
    else:
        local_energies = local_energies.conj()
        local_energies = local_energies.reshape(1, -1)
        gradient = local_energies @ logarithmic_derivatives
        gradient /= logarithmic_derivatives.shape[0]
    return np.ascontiguousarray(2.0 * gradient.real.squeeze())


def covariance_matrix(
    logarithmic_derivatives: np.ndarray, weights: np.ndarray = None
) -> np.ndarray:
    r"""Calculates the covariance matrix S.

    :param logarithmic_derivatives: **centered** logarithmic derivatives of the
        wavefunction with respect to variational parameters.
    :param weights: if specified, it is assumed that {σ} span the whole
        Hilbert space basis. Then ``weights`` are **normalized** probabilities
        ``|⟨σ|ψ⟩|²/‖ψ‖₂``. If ``weights`` is ``None``, then it is assumed that
        {σ} come from Monte Carlo sampling and are distributed according to
        ``|⟨σ|ψ⟩|²/‖ψ‖₂``.
    """
    if weights is not None:
        assert np.isclose(np.sum(weights), 1.0)
        matrix = (
            weights.reshape(-1, 1) * logarithmic_derivatives
        ).T.conj() @ logarithmic_derivatives
    else:
        matrix = logarithmic_derivatives.T.conj() @ logarithmic_derivatives
        matrix /= logarithmic_derivatives.shape[0]
    return np.ascontiguousarray(matrix.real)


class Runner:
    def __init__(self, config):
        self.config = config
        self.__load_hamiltonian()
        self.__load_model()
        self.__load_optimiser()
        self.__add_loggers()
        self.__add_mc_options()
        self.compute_overlap = self.__load_exact()
        self._iteration = 0

    def __load_hamiltonian(self):
        self.hamiltonian = hamiltonian.read_hamiltonian(self.config.hamiltonian)
        self.number_spins = self.hamiltonian.number_spins
        self.magnetisation = self.number_spins % 2
        self.hamiltonian = self.hamiltonian.to_cxx()

    def __load_model(self):
        if not isinstance(self.config.model, (tuple, list)):
            raise ValueError(
                "invalid config.model: {}; ".format(self.config.model)
                + "expected a pair of filenames"
            )

        def load(name):
            return torch.jit.script(core.import_network(name)(self.number_spins))

        amplitude_file, phase_file = self.config.model
        self.amplitude = load(amplitude_file)
        self.phase = load(phase_file)

    def __load_optimiser(self):
        self.optimiser = eval(self.config.optimiser)(
            list(self.amplitude.parameters()) + list(self.phase.parameters())
        )

    def __add_loggers(self):
        self.tb_writer = SummaryWriter(log_dir=self.config.output)
        self.tb_writer.add_graph(self.amplitude, torch.rand(32, self.number_spins))

    def __add_mc_options(self):
        self.mc_options = core.make_monte_carlo_options(self.config, self.number_spins)

    def __load_exact(self):
        if self.config.exact is None:
            return None
        x, y = core.with_file_like(self.config.exact, "rb", pickle.load)
        x = _C.unpack(x)
        y = torch.from_numpy(y).squeeze()
        y /= np.linalg.norm(y)

        def compute():
            with torch.no_grad():
                A = self.amplitude.forward(x)
                A -= torch.max(A) - 1.0
                φ = self.phase.forward(x)
                y_pred = np.exp(
                    torch.cat([A, φ], dim=1).numpy().view(np.complex64).squeeze(axis=1)
                )
                y_pred /= np.linalg.norm(y_pred)
                return abs(np.dot(y_pred.conj(), y))

        return compute


    def _sample_some(self, how: str):
        assert how in {"exact", "full"}
        if how == "exact":
            spins, values = core.sample_exact(self.amplitude, self.mc_options)
            return spins, values, None
        elif how == "full":
            spins = _C.all_spins(self.number_spins, self.magnetisation)
            values = core._forward_with_batches(self.amplitude, spins, batch_size=512)
            weights = core._log_amplitudes_to_probabilities(values)
            return spins, values, weights
        else:
            raise ValueError(
                "invalid sampling type: {}; expected one of 'exact', 'full'".format(how)
            )

    def monte_carlo(self):
        with torch.no_grad():
            spins, _, weights = self._sample_some("exact")
            # spins = _C.all_spins(self.number_spins, self.magnetisation)
            # inputs = _C.unpack(spins)
            # weights = self.amplitude.forward(inputs)
            # weights -= torch.max(weights)
            # weights = torch.exp(2 * weights)
            # weights /= weights.sum()
            if weights is not None:
                weights = weights.numpy().squeeze()

        local_energies = core.local_energy(
            core.combine_amplitude_and_phase(self.amplitude, self.phase),
            # CombiningState(self.amplitude, self.phase),
            self.hamiltonian,
            spins,
        )
        if weights is not None:
            energy = np.dot(weights, local_energies)
            variance = np.dot(weights, np.abs(local_energies - energy) ** 2)
        else:
            energy = np.mean(local_energies)
            variance = np.var(local_energies)
        self.tb_writer.add_scalar("SR/energy_real", energy.real, self._iteration)
        self.tb_writer.add_scalar("SR/energy_imag", energy.imag, self._iteration)
        self.tb_writer.add_scalar("SR/variance", variance, self._iteration)

        logarithmic_derivatives = logarithmic_derivative(
            (self.amplitude, self.phase), _C.unpack(spins)
        )
        # Centering
        if weights is not None:
            mean_derivative = (weights @ logarithmic_derivatives).reshape(1, -1)
        else:
            mean_derivative = np.mean(logarithmic_derivatives, axis=0).reshape(1, -1)
        logarithmic_derivatives -= mean_derivative

        force = energy_gradient(local_energies, logarithmic_derivatives, weights)
        self.tb_writer.add_scalar("SR/grad", np.linalg.norm(force), self._iteration)
        S = covariance_matrix(logarithmic_derivatives, weights)
        return force, S

    def solve(self, matrix, vector):
        n = matrix.shape[0]
        # diag = np.diag(matrix)
        # diag = np.where(diag <= 0.0, 1.0, diag)
        # diag = np.sqrt(diag)
        #
        # matrix = matrix / (diag.reshape(-1, 1) @ diag.reshape(1, -1))
        # vector = vector / diag

        matrix += 1e-2 * np.eye(matrix.shape[0])
        x = scipy.linalg.solve(matrix, vector)
        # x /= diag
        return x

    def set_gradient(self, grad):
        def run(m, i):
            for p in m.parameters():
                n = p.numel()
                if p.grad is not None:
                    p.grad.view(-1).copy_(grad[i : i + n])
                else:
                    p.grad = grad[i : i + n].view(p.size())
                i += n
            return i

        with torch.no_grad():
            i = run(self.amplitude, 0)
            _ = run(self.phase, i)

    def step(self):
        force, S = self.monte_carlo()
        delta = self.solve(S, force)
        self.tb_writer.add_scalar("SR/delta", np.linalg.norm(delta), self._iteration)
        self.set_gradient(torch.from_numpy(delta))
        self.optimiser.step()
        if self.compute_overlap is not None:
            self.tb_writer.add_scalar(
                "SR/overlap", self.compute_overlap(), self._iteration
            )
        self._iteration += 1


Config = namedtuple(
    "Config",
    [
        "model",
        "output",
        "hamiltonian",
        "epochs",
        "number_samples",
        "number_chains",
        "optimiser",
        ## OPTIONAL
        "exact",
        "magnetisation",
        "sweep_size",
        "number_discarded",
    ],
    defaults=[None, None, None, None],
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=argparse.FileType(mode="r"), help="path to JSON config file"
    )
    args = parser.parse_args()
    config = json.load(args.config_file)
    config = Config(**config)

    runner = Runner(config)
    for i in range(config.epochs):
        runner.step()

    # config = Config(
    #     model=("example/1x10/sr/amplitude_wip.py", "example/1x10/sr/phase_wip.py"),
    #     hamiltonian="/vol/tcm01/westerhout_tom/nqs-playground/data/1x10/hamiltonian.txt",
    #     epochs=300,
    #     number_samples=400,
    #     number_chains=1,
    #     output="sr/run/5",
    #     exact="/vol/tcm01/westerhout_tom/nqs-playground/data/1x10/ground_state.pickle",
    #     optimiser="lambda p: torch.optim.SGD(p, lr=1e-2)",
    # )


if __name__ == "__main__":
    main()
