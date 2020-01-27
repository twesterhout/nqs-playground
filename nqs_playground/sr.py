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
import time
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import nqs_playground
from nqs_playground.core import forward_with_batches, combine_amplitude_and_phase
from nqs_playground import (
    _C,
    SamplingOptions,
    sample_some,
    unpack,
    local_values,
    local_values_slow,
)

# SR requires one to run a whole lot of backward operations on independent
# inputs. We parallelise this using PyTorches interop thread-pool. The
# following value is determined emprically. Feel free to modify it.
torch.set_num_interop_threads(cpu_count() // 2)


def _get_device(module: torch.nn.Module):
    r"""Determines the device on which ``module`` resides."""
    return next(module.parameters()).device


def jacobian(module: torch.jit.ScriptModule, inputs: Tensor) -> Tensor:
    r"""Given a ``torch.nn.Module`` and a ``torch.Tensor`` of inputs, computes
    the Jacobian ∂module(inputs)/∂W where W are module's parameters.

    It is assumed that if ``inputs`` has shape ``(batch_size, in_features)``,
    then ``module(inputs)`` has shape ``(batch_size, 1)``.
    """

    # This is a context manager to temporary change the number of threads used
    # by PyTorch for intraop parallelism. Think, number of OpenMP threads.
    @contextmanager
    def num_threads(n):
        old = torch.get_num_threads()
        torch.set_num_threads(n)
        yield
        torch.set_num_threads(old)

    parameters = list(module.parameters())
    n = sum(map(torch.numel, parameters))
    forward = module._c._get_method("forward")

    # This if statement if ugly and stupid, but in torch 1.3.1 the argument is
    # called keep_graph, but in later versions it's retain_graph. And we don't
    # want to interfere with JIT.
    if tuple(map(int, torch.__version__.split("."))) <= (1, 3, 1):

        # Serial implementation. It computes gradients with respect to
        # parameters for every input in xs. This is slow since we process one
        # spin configuration at a time. Unfortunately, I have no idea how to
        # optimise it. Perhaps, processing in a layer per layer fashion could
        # be faster...
        @torch.jit.script
        def task(xs: Tensor, n: int, parameters: List[Tensor]):
            batch_size = xs.size(0)
            out = torch.empty([batch_size, n], dtype=torch.float32, device=xs.device)
            for i in range(batch_size):
                dws = torch.autograd.grad(
                    [forward(xs[i])],
                    parameters,
                    keep_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                torch.cat([dw.view([-1]) for dw in dws], out=out[i])
            return out

    else:

        @torch.jit.script
        def task(xs: Tensor, n: int, parameters: List[Tensor]):
            batch_size = xs.size(0)
            out = torch.empty([batch_size, n], dtype=torch.float32, device=xs.device)
            for i in range(batch_size):
                dws = torch.autograd.grad(
                    [forward(xs[i])],
                    parameters,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                offset = 0
                for dw in dws:
                    if dw is not None:
                        out[i, offset:offset + dw.numel()] = dw.view([-1])
                        offset += dw.numel()
                # torch.cat([dw.view([-1]) for dw in dws], out=out[i])
            return out

    @torch.jit.script
    def go(inputs: torch.Tensor, parameters: List[Tensor], n: int, chunk_size: int):
        futures = [
            torch.jit._fork(task, xs, n, parameters)
            for xs in torch.split(inputs, chunk_size, dim=0)
        ]
        return torch.cat([torch.jit._wait(f) for f in futures], dim=0)

    with torch.jit.optimized_execution(True), num_threads(
        cpu_count() // torch.get_num_interop_threads()
    ):
        chunk_size = min(inputs.size(0) // torch.get_num_interop_threads(), 2048)
        return go(inputs, parameters, n, chunk_size)


class LogarithmicDerivatives:
    def __init__(self, modules, inputs):
        self._modules = modules
        self._inputs = inputs

    def __compute_jacobians(self):
        r"""Initialises :attr:`real` and :attr:`imag` with Jacobians of log
        amplitudes and phases respectively. Repeated calls to
        ``__compute_jacobian`` will do nothing.
        """
        if not hasattr(self, "real"):
            self.real = jacobian(self._modules[0], self._inputs)
        if not hasattr(self, "imag"):
            self.imag = jacobian(self._modules[1], self._inputs)

    def __center(self, weights=None):
        r"""Centers logarithmic derivatives, i.e. ``Oₖ <- Oₖ - ⟨Oₖ⟩``. Repeated
        calls to ``__center`` will do nothing.
        """
        if weights is not None:
            raise NotImplementedError()
        if not hasattr(self, "__are_centered"):
            self.__compute_jacobians()
            self.real -= self.real.mean(dim=0)
            self.imag -= self.imag.mean(dim=0)
            self.__are_centered = True

    def gradient(
        self, local_energies: np.ndarray, weights=None, compute_jacobian=True
    ) -> Tensor:
        r"""Computes the gradient of ``⟨H⟩`` with respect to neural network
        parameters given local energies ``{⟨s|H|ψ⟩/⟨s|ψ⟩ | s ~ |ψ|²}``.
        """
        if weights is not None:
            raise NotImplementedError()
        if not compute_jacobian:
            raise NotImplementedError()

        device = _get_device(self._modules[0])
        if isinstance(local_energies, np.ndarray):
            if local_energies.dtype != np.complex64:
                raise ValueError(
                    "local_energies has wrong dtype: {}; expected complex64"
                    "".format(local_energies.dtype)
                )
            # Reshape making sure we don't create a copy
            local_energies = local_energies.view()
            local_energies.shape = (-1, 1)
            # Treat complex array as Nx2 tensor of real numbers
            local_energies = torch.from_numpy(local_energies.view(np.float32))
        local_energies = local_energies.to(device, copy=False)

        self.__center()
        with torch.no_grad():
            # The following is an implementation of Eq. (6.22) (without the
            # minus sign) in "Quantum Monte Carlo Approaches for Correlated
            # Systems" by F.Becca & S.Sorella.
            #
            # Basically, what we need to compute is `2/N · Re[E*·(O - ⟨O⟩)]`,
            # where `E` is a `Nx1` vector complex-numbered local energies, `O`
            # is a `NxM` matrix of logarithmic derivatives, `⟨O⟩` is a `1xM`
            # vector of mean logarithmic derivatives, and `N` is the number of
            # Monte Carlo samples.
            #
            # Now, `Re[a*·b] = Re[a]·Re[b] - Im[a*]·Im[b] = Re[a]·Re[b] +
            # Im[a]·Im[b]` that's why there are no conjugates or minus signs in
            # the code.
            real_energies = local_energies[:, 0].view([1, -1])
            imag_energies = local_energies[:, 1].view([1, -1])
            scale = 2.0 / self.real.size(0)
            gradient = torch.cat(
                [
                    torch.mm(real_energies, self.real),  # (1xN) x (NxM) -> (1xM)
                    torch.mm(imag_energies, self.imag),  # (1xN) x (NxK) -> (1xK)
                ],
                dim=1,
            )
            gradient *= scale
            return gradient.squeeze()

    def solve(self, gradient: Tensor, scale_inv_reg=1e-3, diag_reg=None, weights=None) -> Tensor:
        r"""Given the gradient of ``⟨H⟩``, calculates ``S⁻¹⟨H⟩ = ⟨(O - ⟨O⟩)†(O
        - ⟨O⟩)⟩⁻¹⟨H⟩``.
        """
        if weights is not None:
            raise NotImplementedError()

        def solve_part(matrix, vector):
            scale = 1.0 / self.real.size(0)
            matrix *= scale

            if scale_inv_reg is not None:
                diag = torch.sqrt(torch.abs(torch.diag(matrix)))  # diag = sqrt(diag(|S_cov|))
                vector_pc = vector / diag
                matrix_pc = torch.einsum('i,ij,j->ij', 1.0 / diag, matrix, 1.0 / diag)  # S[m, n] -> S[m, n] / sqrt(diag[m] * diag[n])
                matrix_pc += scale_inv_reg * torch.eye(matrix_pc.size(0))
                x = torch.cholesky_solve(vector_pc.view([-1, 1]), torch.cholesky(matrix_pc))
                x = x / diag
                return x.squeeze()

            if diag_reg is not None:
                matrix.diagonal()[:] += diag_reg
            # x, _ = torch.solve(vector.view([-1, 1]), matrix)
            x = torch.cholesky_solve(vector.view([-1, 1]), torch.cholesky(matrix))
            return x.squeeze()

        with torch.no_grad():
            middle = self.real.size(1)
            return torch.cat(
                [
                    solve_part(torch.mm(self.real.t(), self.real), gradient[:middle]),
                    solve_part(torch.mm(self.imag.t(), self.imag), gradient[middle:]),
                ]
            )


class Runner:
    def __init__(self, config):
        self.config = config
        self.__load_device()
        self.__load_hamiltonian()
        self.__load_model()
        self.__load_optimiser()
        self.__add_loggers()
        self.__add_sampling_options()
        self.__load_exact()
        self._iteration = 0

    def __load_device(self):
        r"""Determines which device to use for the simulation and initialises
        self.device with it. We support specifying the device as either
        'torch.device' (for cases when you e.g. have multiple GPUs and want to
        use some particular one) or as a string with device type (i.e. either
        "gpu" or "cpu").
        """
        device = self.config.device
        if isinstance(device, (str, bytes)):
            device = torch.device(device)
        elif not isinstance(device, torch.device):
            raise TypeError(
                "config.device has wrong type: {}; must be either a "
                "'torch.device' or a 'str'".format(type(device))
            )
        self.device = device

    def __load_hamiltonian(self):
        r"""Constructs the Heisenberg Hamiltonian for the simulation and
        initialises self.hamiltonian with it. If ``self.config`` already
        contains a constructed _C.Heisenberg object, we simply use it.
        Otherwise, we assume that ``self.config`` is a path to the file
        specifying edges and couplings. We construct a graph based on it and
        create a basis with no symmetries, but fixed magnetisation.
        """
        hamiltonian = self.config.hamiltonian
        if not isinstance(hamiltonian, _C.Heisenberg):
            if not isinstance(hamiltonian, str):
                raise TypeError(
                    "config.hamiltonian has wrong type: {}; must be either a "
                    "'_C.Heisenberg' or a 'str' path to Heisenberg Hamiltonian "
                    "specification.".format(type(hamiltonian))
                )
            # The following notices that we didn't specify a basis and will try
            # to construct one automatically
            hamiltonian = nqs_playground.read_hamiltonian(hamiltonian, basis=None)
        self.hamiltonian = hamiltonian
        self.basis = self.hamiltonian.basis

    def __load_model(self):
        r"""Constructs amplitude and phase networks and initialises
        ``self.amplitude`` and ``self.phase``.
        """
        model = self.config.model
        if not isinstance(self.config.model, (tuple, list)):
            raise TypeError(
                "config.model has wrong type: {}; expected either a pair of of "
                "torch.nn.Modules or a pair of filenames".format(type(model))
            )
        amplitude, phase = model

        def load(name):
            # name is already a ScriptModule, nothing to be done.
            if isinstance(name, torch.jit.ScriptModule):
                return name
            # name is a Module, so we just compile it.
            elif isinstance(name, torch.nn.Module):
                return torch.jit.script(name)
            # name is a string
            # If name is a Python script, we import the Net class from it,
            # construct the model, and JIT-compile it. Otherwise, we assume
            # that the user wants to continue the simulation and has provided a
            # path to serialised TorchScript module. We simply load it.
            _, extension = os.path.splitext(os.path.basename(name))
            if extension == ".py":
                return torch.jit.script(
                    nqs_playground.core.import_network(name)(self.basis.number_spins)
                )
            return torch.jit.load(name)

        self.amplitude = load(amplitude).to(self.device)
        self.phase = load(phase).to(self.device)

    def __load_optimiser(self):
        optimiser = self.config.optimiser
        if isinstance(optimiser, str):
            # NOTE: Yes, this is unsafe, but terribly convenient!
            optimiser = eval(optimiser)
        if not isinstance(optimiser, torch.optim.Optimizer):
            # assume that optimiser is a lambda
            params = list(self.amplitude.parameters()) + list(self.phase.parameters())
            optimiser = optimiser(params)
        self.optimiser = optimiser

    def __add_loggers(self):
        self.tb_writer = SummaryWriter(log_dir=self.config.output)

    def __add_sampling_options(self):
        self.sampling_options = SamplingOptions(
            self.config.number_samples, self.config.number_chains, device=self.device
        )

    @property
    def combined_state(self):
        if not hasattr(self, "__combined_state"):
            self.__combined_state = combine_amplitude_and_phase(
                self.amplitude, self.phase, number_spins=self.basis.number_spins
            )
        return self.__combined_state

    def __load_exact(self):
        if self.config.exact is None:
            # We don't know the exact ground state
            self.compute_overlap = None
            return

        ground_state = self.config.exact
        if isinstance(ground_state, str):
            # Ground state was saved using NumPy binary format
            ground_state = np.load(ground_state)
        ground_state = ground_state.squeeze()

        def compute():
            with torch.no_grad(), torch.jit.optimized_execution(True):
                state = forward_with_batches(
                    self.combined_state,
                    torch.from_numpy(self.basis.states.view(np.int64)),
                    8192,
                )
                state[:, 0] -= torch.max(state[:, 0])
                state = state.numpy().view(np.complex64).squeeze()
                state = np.exp(state, out=state)
                overlap = abs(np.dot(state, ground_state)) / np.linalg.norm(state)
                return overlap

        self.compute_overlap = compute

    def monte_carlo(self):
        with torch.no_grad():
            spins, _ = sample_some(
                lambda x: self.amplitude(unpack(x, self.basis.number_spins)),
                self.basis,
                self.sampling_options,
                mode="monte_carlo",
            )
            weights = None

        local_energies = local_values(
            spins, self.hamiltonian, self.combined_state, batch_size=8192
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
            overlap = self.compute_overlap()
            self.tb_writer.add_scalar("SR/overlap", overlap, self._iteration)

        # TODO: Fix this! Unpacking all the Monte Carlo data might not be a
        # good idea...
        logarithmic_derivatives = LogarithmicDerivatives(
            (self.amplitude, self.phase), unpack(spins, self.basis.number_spins)
        )
        force = logarithmic_derivatives.gradient(local_energies, weights)
        self.tb_writer.add_scalar("SR/grad", torch.norm(force), self._iteration)
        delta = logarithmic_derivatives.solve(force, weights=weights)
        return force, delta

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
        tick = time.time()
        force, delta = self.monte_carlo()
        # delta = self.solve(S, force)
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
        print("Information :: Done in {} seconds...".format(time.time() - tick))
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
        "device",
        "exact",
        "magnetisation",
        "sweep_size",
        "number_discarded",
    ],
    defaults=["cpu", None, None, None, None],
)


def run(config):
    runner = Runner(config)
    for i in range(config.epochs):
        runner.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=argparse.FileType(mode="r"), help="path to JSON config file"
    )
    args = parser.parse_args()
    config = json.load(args.config_file)
    config = Config(**config)
    run(config)


if __name__ == "__main__":
    main()
