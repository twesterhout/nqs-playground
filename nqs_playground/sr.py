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
from contextlib import contextmanager
import json
import math
import os
import pickle
from psutil import cpu_count
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ._C_nqs import v2 as _C
from . import core, hamiltonian
from .supervised import eliminate_phase
from .swo import sample_some, forward_with_batches

# SR requires one to run a whole lot of backward operations on independent
# inputs. We parallelise this using PyTorches interop thread-pool. The
# following value is determined emprically. Feel free to modify it.
torch.set_num_interop_threads(cpu_count() // 2)


def num_parameters(module: torch.nn.Module) -> int:
    r"""Given a ``torch.nn.Module``, returns total number of parameters in it.
    """
    return sum(map(torch.numel, module.parameters()))


# def jacobian(module: torch.nn.Module, inputs: torch.Tensor, out=None) -> torch.Tensor:
#     r"""Given a ``torch.nn.Module`` and a ``torch.Tensor`` of inputs, computes
#     the Jacobian ∂module(inputs)/∂W where W are module's parameters.
#
#     It is assumed that if ``inputs`` has shape ``(batch_size, in_features)``,
#     then ``module(inputs)`` has shape ``(batch_size, 1)``.
#     """
#     parameters = list(module.parameters())
#     shape = (inputs.size(0), num_parameters(module))
#     if out is None:
#         out = torch.zeros(*shape)
#     else:
#         assert out.size() == shape
#     for i, xs in enumerate(inputs):
#         dws = torch.autograd.grad(
#             module(xs.view(1, -1)),
#             parameters,
#             retain_graph=True,
#             create_graph=False,
#             allow_unused=True,
#         )
#         torch.cat([dw.reshape(-1) for dw in dws], out=out[i])
#     return out


def jacobian(module: torch.jit.ScriptModule, inputs: torch.Tensor) -> torch.Tensor:
    r"""Given a ``torch.nn.Module`` and a ``torch.Tensor`` of inputs, computes
    the Jacobian ∂module(inputs)/∂W where W are module's parameters.

    It is assumed that if ``inputs`` has shape ``(batch_size, in_features)``,
    then ``module(inputs)`` has shape ``(batch_size, 1)``.
    """

    @contextmanager
    def num_threads(n):
        old = torch.get_num_threads()
        torch.set_num_threads(n)
        yield
        torch.set_num_threads(old)

    parameters = list(module.parameters())
    forward = module._c._get_method("forward")

    @torch.jit.script
    def task(xs: torch.Tensor, parameters: List[torch.Tensor]):
        out = []
        for i in range(xs.size(0)):
            dws = torch.autograd.grad(
                [forward(xs[i])],
                parameters,
                keep_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            out.append(torch.cat([dw.reshape([-1]) for dw in dws]).view([1, -1]))
        return torch.cat(out, dim=0)

    @torch.jit.script
    def go(inputs: torch.Tensor, parameters: List[torch.Tensor]):
        futures = [
            torch.jit._fork(task, xs, parameters) for xs in torch.split(inputs, 64)
        ]
        return torch.cat([torch.jit._wait(f) for f in futures], dim=0)

    with num_threads(cpu_count() // torch.get_num_interop_threads()):
        return go(inputs, parameters)


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
    out[:, :middle, 0] = jacobian(amplitude, inputs)
    out[:, middle:, 1] = jacobian(phase, inputs)
    return out.detach().numpy().view(np.complex64).squeeze(axis=2)


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
        np.conj(local_energies, out=local_energies)
        local_energies = local_energies.reshape(1, -1)
        gradient = local_energies @ logarithmic_derivatives
    else:
        local_energies = local_energies.conj()
        local_energies = local_energies.reshape(1, -1)
        gradient = local_energies @ logarithmic_derivatives
        assert gradient.real.dtype == np.float32
    return torch.from_numpy(
        (2.0 / logarithmic_derivatives.shape[0]) * gradient.real.squeeze()
    ).contiguous()


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
    assert weights is None
    if weights is not None:
        assert np.isclose(np.sum(weights), 1.0)
        matrix = (
            weights.reshape(-1, 1) * logarithmic_derivatives
        ).T.conj() @ logarithmic_derivatives
    else:
        with torch.no_grad():
            real = torch.from_numpy(logarithmic_derivatives.real).contiguous()
            imag = torch.from_numpy(logarithmic_derivatives.imag).contiguous()
            matrix = torch.mm(real.t(), real)
            scale = 1.0 / logarithmic_derivatives.shape[0]
            torch.addmm(scale, matrix, scale, imag.t(), imag, out=matrix)
            # matrix = logarithmic_derivatives.T.conj() @ logarithmic_derivatives
            # matrix /= logarithmic_derivatives.shape[0]
    return matrix  # np.ascontiguousarray(matrix.real)


class Runner:
    def __init__(self, config):
        self.config = config
        self.__load_hamiltonian()
        self.__load_model()
        self.__load_optimiser()
        self.__add_loggers()
        self.__add_sampling_options()
        self.compute_overlap = self.__load_exact()
        self._iteration = 0

    def __load_hamiltonian(self):
        self.hamiltonian = hamiltonian.read_hamiltonian(self.config.hamiltonian)
        number_spins = self.hamiltonian.number_spins
        self.basis = _C.SpinBasis(
            [], number_spins=number_spins, hamming_weight=number_spins // 2
        )
        self.basis.build()
        self.hamiltonian = _C.Heisenberg(self.hamiltonian._specs, self.basis)

    def __load_model(self):
        if not isinstance(self.config.model, (tuple, list)):
            raise ValueError(
                "invalid config.model: {}; ".format(self.config.model)
                + "expected a pair of filenames"
            )

        def load(name):
            return torch.jit.script(core.import_network(name)(self.basis.number_spins))

        amplitude_file, phase_file = self.config.model
        self.amplitude = load(amplitude_file)
        self.phase = load(phase_file)

    def __load_optimiser(self):
        self.optimiser = eval(self.config.optimiser)(
            list(self.amplitude.parameters()) + list(self.phase.parameters())
        )

    def __add_loggers(self):
        self.tb_writer = SummaryWriter(log_dir=self.config.output)

    def __add_sampling_options(self):
        self.sampling_options = core.SamplingOptions(
            self.config.number_samples, self.config.number_chains
        )

    def __load_exact(self):
        if self.config.exact is None:
            return None
        if self.config.exact.endswith(".npy"):
            # Ground state was saved using NumPy binary format
            self.ground_state = eliminate_phase(np.load(self.config.exact).squeeze())
        else:
            r = core.with_file_like(self.config.exact, "rb", pickle.load)
            if isinstance(r, torch.Tensor):
                self.ground_state = r
            elif isinstance(r, np.ndarray):
                self.ground_state = eliminate_phase(r)
            else:
                _, self.ground_state = r
                self.ground_state = eliminate_phase(r)
        self.ground_state /= torch.norm(self.ground_state)

        def compute(A=None, φ=None):
            with torch.no_grad(), torch.jit.optimized_execution(True):
                if A is None:
                    A = forward_with_batches(
                        lambda x: self.amplitude(_C.unpack(x, self.basis.number_spins)),
                        self.basis.states,
                        8192,
                    ).squeeze()
                if φ is None:
                    φ = forward_with_batches(
                        lambda x: self.phase(_C.unpack(x, self.basis.number_spins)),
                        self.basis.states,
                        8192,
                    ).squeeze()
                A -= torch.max(A) - 1.0
                torch.exp_(A)
                A /= torch.norm(A)
                overlap = torch.sqrt(
                    torch.abs(torch.dot(A * torch.cos(φ), self.ground_state))
                    + torch.abs(torch.dot(A * torch.sin(φ), self.ground_state))
                ).item()
                return overlap

        return compute

    def monte_carlo(self):
        with torch.no_grad():
            spins, A, cache = sample_some(
                self.amplitude, self.basis, self.sampling_options, mode="exact"
            )
            weights = None
            # spins = _C.all_spins(self.number_spins, self.magnetisation)
            # inputs = _C.unpack(spins)
            # weights = self.amplitude.forward(inputs)
            # weights -= torch.max(weights)
            # weights = torch.exp(2 * weights)
            # weights /= weights.sum()
            # if weights is not None:
            #     weights = weights.numpy().squeeze()

        local_energies = core.local_energy(
            core.combine_amplitude_and_phase(self.amplitude, self.phase),
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
        if self.compute_overlap is not None:
            overlap = self.compute_overlap(A=cache["ys"])
            self.tb_writer.add_scalar("SR/overlap", overlap, self._iteration)

        # TODO: Fix this! Unpacking all the Monte Carlo data might not be a
        # good idea...
        logarithmic_derivatives = logarithmic_derivative(
            (self.amplitude, self.phase), _C.unpack(spins, self.basis.number_spins)
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
        diag = matrix.diagonal()
        diag += 1e-2
        x, _ = torch.solve(vector.view([-1, 1]), matrix)
        return x.squeeze()

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
        self.tb_writer.add_scalar("SR/delta", torch.norm(delta), self._iteration)
        self.set_gradient(delta)
        self.optimiser.step()
        torch.save(
            self.amplitude.state_dict(),
            os.path.join(self.config.output, "amplitude_weights.pt"),
        )
        torch.save(
            self.amplitude.state_dict(),
            os.path.join(self.config.output, "sign_weights.pt"),
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


if __name__ == "__main__":
    main()
