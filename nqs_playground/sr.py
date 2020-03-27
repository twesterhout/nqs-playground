# Copyright Tom Westerhout (c) 2019-2020
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
import time
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

if torch.has_cuda:
    import threading
    from torch._utils import ExceptionWrapper

import nqs_playground
from nqs_playground.core import forward_with_batches, combine_amplitude_and_phase
from nqs_playground import (
    _C,
    SamplingOptions,
    sample_some,
    unpack,
    jacobian_simple,
    jacobian_cpu,
    jacobian_cuda,
    local_values,
    local_values_slow,
)


def _get_device(module: torch.nn.Module):
    r"""Determines the device on which ``module`` resides."""
    return next(module.parameters()).device


def _remove_singularity(matrix: Tensor, eps: float = 1e-4) -> Tensor:
    r"""Removes singularities from ``matrix``. Application of this function to
    covariance matrix essentially means that we exclude some variational
    parameters from optimization.

    .. warning:: Modifies matrix in-place!
    """
    # obtains all indices i for which matrix[i, i] < eps
    indices = torch.nonzero(matrix.diagonal().abs() < eps).squeeze(dim=1)
    matrix.index_fill_(0, indices, 0.0)  # sets matrix[i, :] = 0.0 for all i
    matrix.index_fill_(1, indices, 0.0)  # sets matrix[:, i] = 0.0 for all i
    matrix.diagonal().index_fill_(0, indices, 1.0)  # sets matrix[i, i] = 1.0
    return matrix
    # Old version:
    # for i in range(matrix.size(0)):
    #     if matrix[i, i] < eps:
    #         matrix[i, :] = 0.0
    #         matrix[:, i] = 0.0
    #         matrix[i, i] = 1.0
    # return matrix


class LogarithmicDerivatives:
    def __init__(
        self,
        modules: Tuple[torch.jit.ScriptModule, torch.jit.ScriptModule],
        inputs: Tensor,
    ):
        r"""

        :param modules: is a tuple (or list) of two
            ``torch.jit.ScriptModule``s. The predict respectively real and
            imaginary parts of the logarithm of the wavefunction.
        :param inputs: is a Tensor with spin configurations obtained from Monte
            Carlo sampling.
        """
        self._modules = modules
        self._inputs = inputs
        self._real, self._imag = None, None
        self.__is_centered = False

    def _compute_jacobians(self):
        r"""Initialises :attr:`_real` and :attr:`_imag` with Jacobians of log
        amplitudes and phases respectively. Repeated calls to
        ``_compute_jacobian`` will do nothing.

        .. note:: When running on GPU, ``jacobian`` function will automatically
                  use all available GPUs on the system.
        """
        if self._real is not None and self._imag is not None:
            return
        if self._inputs.device.type == "cpu":
            # TODO: Tune number of threads
            self._real = jacobian_cpu(self._modules[0], self._inputs)
            self._imag = jacobian_cpu(self._modules[1], self._inputs)
        else:
            # If we have more than one CUDA device, we put Jacobians of
            # amplitude and phase networks on different devices such that
            # operations on them can be parallelised.
            if torch.cuda.device_count() > 1:
                output_devices = ["cuda:0", "cuda:1"]
            else:
                output_devices = ["cuda", "cuda"]
            self._real = jacobian_cuda(
                self._modules[0], self._inputs, output_device=output_devices[0]
            )
            self._imag = jacobian_cuda(
                self._modules[1], self._inputs, output_device=output_devices[1]
            )

    def _center(self, weights=None):
        r"""Centers logarithmic derivatives, i.e. ``Oₖ <- Oₖ - ⟨Oₖ⟩``. Repeated
        calls to ``__center`` will do nothing.
        """

        def center(matrix, weights):
            if weights is not None:
                raise NotImplementedError()
            matrix -= matrix.mean(dim=0)

        if not self.__is_centered:
            self._compute_jacobians()
            with torch.no_grad():
                center(self._real, weights)
                center(self._imag, weights)
            self.__is_centered = True

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

        self._center()
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
            real_energies = local_energies[:, 0].view([1, -1]).to(self._real.device)
            imag_energies = local_energies[:, 1].view([1, -1]).to(self._imag.device)
            scale = 2.0 / self._real.size(0)
            snd_part = torch.mm(imag_energies, self._imag)  # (1xN) x (NxK) -> (1xK)
            snd_part *= scale
            snd_part = snd_part.to(self._real.device, non_blocking=True)
            fst_part = torch.mm(real_energies, self._real)  # (1xN) x (NxM) -> (1xM)
            fst_part *= scale
            return torch.cat([fst_part, snd_part], dim=1).squeeze(dim=0)

    def __solve_part(
        self,
        matrix: Tensor,
        vector: Tensor,
        scale_inv_reg: Optional[float],
        diag_reg: Optional[float],
    ) -> Tensor:
        r"""Calculates ``matrix⁻¹·vector``, i.e. solves the linear system."""
        if scale_inv_reg is not None and diag_reg is not None:
            raise ValueError("scale_inv_reg and diag_reg are mutually exclusive")

        matrix = _remove_singularity(matrix)
        # The following is an implementation of Eq. (6.51) and (6.52)
        # from "Quantum Monte Carlo Approaches for Correlated Systems"
        # by F.Becca & S.Sorella.
        if scale_inv_reg is not None:
            # diag <- sqrt(diag(|S|))
            diag = torch.sqrt(torch.abs(torch.diagonal(matrix)))
            vector_pc = vector / diag
            # S_pc[m, n] <- S[m, n] / sqrt(diag[m] * diag[n])
            inv_diag = 1.0 / diag
            matrix_pc = torch.einsum("i,ij,j->ij", inv_diag, matrix, inv_diag)
            # regularizes the preconditioned matrix
            matrix_pc.diagonal()[:] += scale_inv_reg
            # solves the linear system
            try:
                u = torch.cholesky(matrix_pc)
            except RuntimeError as e:
                print("Warning :: {} Retrying with bigger diagonal shift...".format(e))
                matrix_pc.diagonal()[:] += scale_inv_reg
                u = torch.cholesky(matrix_pc)
            x = torch.cholesky_solve(vector_pc.view([-1, 1]), u).squeeze()
            x *= inv_diag
            return x
        # The simpler approach
        if diag_reg is not None:
            matrix.diagonal()[:] += diag_reg
        try:
            u = torch.cholesky(matrix)
        except RuntimeError as e:
            print("Warning :: {} Retrying with bigger diagonal shift...".format(e))
            matrix.diagonal()[:] += diag_reg
            u = torch.cholesky(matrix)
        x = torch.cholesky_solve(vector.view([-1, 1]), u).squeeze()
        return x

    def solve(
        self, gradient: Tensor, scale_inv_reg=None, diag_reg=1e-2, weights=None
    ) -> Tensor:
        r"""Given the gradient of ``⟨H⟩``, calculates
        ``S⁻¹⟨H⟩ = ⟨(O - ⟨O⟩)†(O - ⟨O⟩)⟩⁻¹⟨H⟩``.
        """
        if weights is not None:
            raise NotImplementedError()

        @torch.no_grad()
        def task(
            log_derivatives: Tensor, operator_gradient: Tensor, **kwargs
        ) -> Tensor:
            device = log_derivatives.device
            operator_gradient = operator_gradient.to(device, non_blocking=True)
            covariance_matrix = torch.mm(log_derivatives.t(), log_derivatives)
            covariance_matrix *= 1.0 / log_derivatives.size(0)
            return self.__solve_part(covariance_matrix, operator_gradient, **kwargs)

        middle = self._real.size(1)
        kwargs = {"scale_inv_reg": scale_inv_reg, "diag_reg": diag_reg}
        if self._real.device == self._imag.device:
            # Serial implementation
            return torch.cat(
                [
                    task(self._real, gradient[:middle], **kwargs),
                    task(self._imag, gradient[middle:], **kwargs),
                ]
            )
        else:
            # Parallel implementation
            results = [None, None]

            def _worker(i, *args, **kwargs):
                try:
                    results[i] = task(*args, **kwargs)
                except Exception:
                    results[i] = ExceptionWrapper(
                        where="in thread {} on device {}".format(i, args[0].device)
                    )

            threads = [
                threading.Thread(target=_worker, args=(i, d, f), kwargs=kwargs)
                for i, (d, f) in enumerate(
                    [(self._real, gradient[:middle]), (self._imag, gradient[middle:])]
                )
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            for result in results:
                if isinstance(result, ExceptionWrapper):
                    result.reraise()
            return torch.cat([results[0], results[1].to(results[0].device)])


def load_device(config) -> torch.device:
    r"""Determines which device to use for the simulation. We support
    specifying the device as either 'torch.device' (for cases when you e.g.
    have multiple GPUs and want to use some particular one) or as a string with
    device type (i.e. either "gpu" or "cpu").
    """
    device = config.device
    if isinstance(device, (str, bytes)):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        raise TypeError(
            "config.device has wrong type: {}; must be either a "
            "'torch.device' or a 'str'".format(type(device))
        )
    return device


def load_hamiltonian(config) -> _C.Heisenberg:
    r"""Constructs the Heisenberg Hamiltonian for the simulation. If ``config``
    already contains a constructed _C.Heisenberg object, we simply use it.
    Otherwise, we assume that ``config`` is a path to the file specifying edges
    and couplings. We construct a graph based on it and create a basis with no
    symmetries, but fixed magnetisation.
    """
    hamiltonian = config.hamiltonian
    if isinstance(hamiltonian, _C.Heisenberg):
        return hamiltonian
    if not isinstance(hamiltonian, str):
        raise TypeError(
            "config.hamiltonian has wrong type: {}; must be either a "
            "'_C.Heisenberg' or a 'str' path to Heisenberg Hamiltonian "
            "specification.".format(type(hamiltonian))
        )
    # The following notices that we didn't specify a basis and will try
    # to construct one automatically
    hamiltonian = nqs_playground.read_hamiltonian(hamiltonian, basis=None)
    return hamiltonian


def load_model(model, number_spins=None) -> torch.jit.ScriptModule:
    # model is already a ScriptModule, nothing to be done.
    if isinstance(model, torch.jit.ScriptModule):
        return model
    # model is a Module, so we just compile it.
    elif isinstance(model, torch.nn.Module):
        return torch.jit.script(model)
    # model is a string
    # If model is a Python script, we import the Net class from it,
    # construct the model, and JIT-compile it. Otherwise, we assume
    # that the user wants to continue the simulation and has provided a
    # path to serialised TorchScript module. We simply load it.
    _, extension = os.path.splitext(os.path.basename(model))
    if extension == ".py":
        if number_spins is None:
            raise ValueError(
                "cannot construct the network imported from {}, because "
                "the number of spins is not given".format(model)
            )
        return torch.jit.script(nqs_playground.core.import_network(name)(number_spins))
    return torch.jit.load(model)


def load_optimiser(optimiser, parameters) -> torch.optim.Optimizer:
    if isinstance(optimiser, str):
        # NOTE: Yes, this is unsafe, but terribly convenient!
        optimiser = eval(optimiser)
    if not isinstance(optimiser, torch.optim.Optimizer):
        # assume that optimiser is a lambda
        optimiser = optimiser(parameters)
    return optimiser


def load_exact(ground_state):
    if ground_state is None:
        return None
    if isinstance(ground_state, str):
        # Ground state was saved using NumPy binary format
        ground_state = np.load(ground_state)
        if ground_state.ndim > 1:
            raise ValueError("ground state must be a vector")
    return ground_state.squeeze().astype(np.complex64)


class Runner:
    def __init__(self, config):
        self.config = config
        self.device = load_device(self.config)
        self.hamiltonian = load_hamiltonian(self.config)
        self.basis = self.hamiltonian.basis
        self.amplitude, self.phase = self._load_models()
        self.optimiser = self._load_optimiser()
        self.tb_writer = self._load_loggers()
        self.sampling_options = self._load_sampling_options()
        self.ground_state = load_exact(self.config.exact)
        self._iteration = 0

    def _load_models(self):
        r"""Constructs amplitude and phase networks."""
        model = self.config.model
        if not isinstance(self.config.model, (tuple, list)):
            raise TypeError(
                "config.model has wrong type: {}; expected either a pair of of "
                "torch.nn.Modules or a pair of filenames".format(type(model))
            )
        amplitude, phase = model
        load = lambda m: load_model(m, self.basis.number_spins).to(self.device)
        amplitude, phase = load(amplitude), load_model(phase)
        # We check that all parameters are contiguous. Otherwise jacobian
        # function might misbehave.
        for p in amplitude.parameters():
            assert p.is_contiguous()
        for p in phase.parameters():
            assert p.is_contiguous()
        return amplitude, phase

    def _load_optimiser(self):
        return load_optimiser(
            self.config.optimiser,
            list(self.amplitude.parameters()) + list(self.phase.parameters()),
        )

    def _load_loggers(self):
        return SummaryWriter(log_dir=self.config.output)

    def _load_sampling_options(self):
        return SamplingOptions(
            self.config.number_samples, self.config.number_chains, device=self.device
        )

    @property
    def combined_state(self):
        if not hasattr(self, "__combined_state"):
            self.__combined_state = combine_amplitude_and_phase(
                self.amplitude, self.phase
            )
        return self.__combined_state

    def compute_overlap(self):
        if self.ground_state is None:
            return None
        with torch.no_grad():
            state = forward_with_batches(
                self.combined_state,
                torch.from_numpy(self.basis.states.view(np.int64)).to(self.device),
                8192,
            )
            state[:, 0] -= torch.max(state[:, 0])
            state = state.cpu().numpy().view(np.complex64).squeeze()
            state = np.exp(state, out=state)
            overlap = abs(np.dot(state, self.ground_state)) / np.linalg.norm(state)
            return overlap

    def _energy_and_variance(self, local_values, weights) -> Tuple[complex, float]:
        r"""Given local energies, computes an energy estimate and variance
        estimate.
        """
        if weights is not None:
            energy = np.dot(weights, local_values)
            variance = np.dot(weights, np.abs(local_values - energy) ** 2)
        else:
            energy = np.mean(local_values)
            variance = np.var(local_values)
        return energy, variance

    def monte_carlo(self):
        # First of all, do Monte Carlo sampling
        with torch.no_grad():
            spins, _, weights = sample_some(
                lambda x: self.amplitude(x),
                self.basis,
                self.sampling_options,
                mode=self.config.sampling_mode,
            )

        local_energies = local_values(spins, self.hamiltonian, self.combined_state)
        energy, variance = self._energy_and_variance(local_energies, weights)
        overlap = self.compute_overlap()
        logarithmic_derivatives = LogarithmicDerivatives(
            (self.amplitude, self.phase), spins
        )
        force = logarithmic_derivatives.gradient(local_energies, weights)
        delta = logarithmic_derivatives.solve(force, weights=weights)

        self.tb_writer.add_scalar("SR/energy_real", energy.real, self._iteration)
        self.tb_writer.add_scalar("SR/energy_imag", energy.imag, self._iteration)
        self.tb_writer.add_scalar("SR/variance", variance, self._iteration)
        if overlap is not None:
            self.tb_writer.add_scalar("SR/overlap", overlap, self._iteration)
        self.tb_writer.add_scalar("SR/grad", torch.norm(force), self._iteration)
        self.tb_writer.add_scalar("SR/delta", torch.norm(delta), self._iteration)
        return force, delta

    def _set_gradient(self, grad):
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
        self._set_gradient(delta)
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
        "sampling_mode",
        "magnetisation",
        "sweep_size",
        "number_discarded",
    ],
    defaults=["cpu", None, "exact", None, None, None],
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
