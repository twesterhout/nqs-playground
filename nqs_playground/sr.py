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
from nqs_playground.core import (
    forward_with_batches,
    combine_amplitude_and_phase,
    combine_amplitude_and_sign_classifier,
)

from nqs_playground import (
    _C,
    SamplingOptions,
    sample_some,
    unpack,
    jacobian_cpu,
    jacobian_cuda,
    local_values,
    local_values_diagonal,
    local_values_slow,
)

def _get_device(module: torch.nn.Module):
    r"""Determines the device on which ``module`` resides."""
    return next(module.parameters()).device


class LogarithmicDerivatives:
    def __init__(
        self,
        modules: Tuple[torch.jit.ScriptModule, torch.jit.ScriptModule],
        inputs: Tensor
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

    def __compute_jacobians(self):
        r"""Initialises :attr:`_real` and :attr:`_imag` with Jacobians of log
        amplitudes and phases respectively. Repeated calls to
        ``__compute_jacobian`` will do nothing.

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
                output_devices = ['cuda:0', 'cuda:0']
            else:
                output_devices = ['cuda', 'cuda']
            self._real = jacobian_cuda(
                self._modules[0], self._inputs, output_device=output_devices[0])
            self._imag = jacobian_cuda(
                self._modules[1], self._inputs, output_device=output_devices[1])
            

    def __center(self, weights=None):
        r"""Centers logarithmic derivatives, i.e. ``Oₖ <- Oₖ - ⟨Oₖ⟩``. Repeated
        calls to ``__center`` will do nothing.
        """
        def center(matrix, weights):
            if weights is not None:
                raise NotImplementedError()
            matrix -= matrix.mean(dim=0)

        if not self.__is_centered:
            self.__compute_jacobians()
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
            real_energies = local_energies[:, 0].view([1, -1]).to(self._real.device)
            imag_energies = local_energies[:, 1].view([1, -1]).to(self._imag.device)
            scale = 2.0 / self._real.size(0)
            snd_part = torch.mm(imag_energies, self._imag)  # (1xN) x (NxK) -> (1xK)
            snd_part *= scale
            snd_part = snd_part.to(self._real.device, non_blocking=True)
            fst_part = torch.mm(real_energies, self._real)  # (1xN) x (NxM) -> (1xM)
            fst_part *= scale
            return torch.cat([fst_part, snd_part], dim=1).squeeze(dim=0)


    def __remove_singularity(self, matrix: Tensor, eps: float = 1e-4) -> Tensor:
        r"""Removes singularities from ``matrix``. Application of this function to
        covariance matrix essentially means that we exclude some variational
        parameters from optimization.
        """
        # obtains all indices i for which matrix[i, i] < eps
        indices = torch.nonzero(matrix.diagonal().abs() < eps).squeeze(dim=1)
        matrix.index_fill_(0, indices, 0.0) # sets matrix[i, :] = 0.0 for all i
        matrix.index_fill_(1, indices, 0.0) # sets matrix[:, i] = 0.0 for all i
        matrix.diagonal().index_fill_(0, indices, 1.0) # sets matrix[i, i] = 1.0
        return matrix
        # Old version:
        # for i in range(matrix.size(0)):
        #     if matrix[i, i] < eps:
        #         matrix[i, :] = 0.0
        #         matrix[:, i] = 0.0
        #         matrix[i, i] = 1.0
        # return matrix

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

        matrix = self.__remove_singularity(matrix)
        # The following is an implementation of Eq. (6.51) and (6.52)
        # from "Quantum Monte Carlo Approaches for Correlated Systems"
        # by F.Becca & S.Sorella.
        if scale_inv_reg is not None:
            # diag <- sqrt(diag(|S|))
            diag = torch.sqrt(torch.abs(torch.diagonal(matrix)))
            vector_pc = vector / diag
            # S_pc[m, n] <- S[m, n] / sqrt(diag[m] * diag[n])
            inv_diag = 1.0 / diag
            matrix_pc = torch.einsum('i,ij,j->ij', inv_diag, matrix, inv_diag)
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

    def solve(self, gradient: Tensor, scale_inv_reg=None, diag_reg=1e-1, weights=None) -> Tensor:
        r"""Given the gradient of ``⟨H⟩``, calculates
        ``S⁻¹⟨H⟩ = ⟨(O - ⟨O⟩)†(O - ⟨O⟩)⟩⁻¹⟨H⟩``.
        """
        if weights is not None:
            raise NotImplementedError()

        @torch.no_grad()
        def task(log_derivatives: Tensor, operator_gradient: Tensor, **kwargs) -> Tensor:
            device = log_derivatives.device
            operator_gradient = operator_gradient.to(device, non_blocking=True)
            covariance_matrix = torch.mm(log_derivatives.t(), log_derivatives)
            covariance_matrix *= 1.0 / log_derivatives.size(0)
            return self.__solve_part(covariance_matrix, operator_gradient, **kwargs)

        middle = self._real.size(1)
        kwargs = {"scale_inv_reg": scale_inv_reg, "diag_reg": diag_reg}
        if self._real.device == self._imag.device:
            # Serial implementation
            return torch.cat([
                task(self._real, gradient[:middle], **kwargs),
                task(self._imag, gradient[middle:], **kwargs),
            ])
        else:
            # Parallel implementation
            results = [None, None]

            def _worker(i, *args, **kwargs):
                try:
                    results[i] = task(*args, **kwargs)
                except Exception:
                    results[i] = ExceptionWrapper(
                        where="in thread {} on device {}".format(i, args[0].device))
            
            threads = [threading.Thread(target=_worker, args=(i, d, f), kwargs=kwargs)
                       for i, (d, f) in enumerate([(self._real, gradient[:middle]),
                                                   (self._imag, gradient[middle:])])]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            for result in results:
                if isinstance(result, ExceptionWrapper):
                    result.reraise()
            return torch.cat([results[0], results[1].to(results[0].device)])


class LogarithmicDerivativesClassifier(LogarithmicDerivatives):
    def __compute_jacobians(self):
        r"""Initialises :attr:`_real` and :attr:`_imag` with Jacobians of log
        amplitudes and phases respectively. Repeated calls to
        ``__compute_jacobian`` will do nothing.

        .. note:: When running on GPU, ``jacobian`` function will automatically
                  use all available GPUs on the system.
        """
        if self._real is not None and self._imag is not None:
            return
        if self._inputs.device.type == "cpu":
            self._derlogphi = jacobian_cpu(self._modules[0], self._inputs)  # \partial\log\varphi_i
            self._n_phi_weights = self._derlogphi.size(1)

            self._derp = jacobian_cpu(self._modules[1], self._inputs)  # \partial p_i
            self._jacobian = torch.zeros([self._derlogphi.size(0), self._n_phi_weights + self._derp.size(1), 2])

            self.p = torch.squeeze(self._modules[1](self._inputs))
            self._jacobian[:, :self._n_phi_weights, 0] = self._derlogphi
            self._jacobian[:, self._n_phi_weights:, 1] = torch.einsum('ij,i->ij', self._derp, 1. / self.p)

            self.weight_a = 4. * self.p * (1. - self.p)
            self.weight_v = (2. * self.p - 1.) ** 2
        else:
            raise NotImplementedError()

    def __remove_singularity(self, matrix: Tensor, eps: float = 1e-4) -> Tensor:
        r"""Removes singularities from ``matrix``. Application of this function to
        covariance matrix essentially means that we exclude some variational
        parameters from optimization.
        """
        # obtains all indices i for which matrix[i, i] < eps
        indices = torch.nonzero(matrix.diagonal().abs() < eps).squeeze(dim=1)
        matrix.index_fill_(0, indices, 0.0) # sets matrix[i, :] = 0.0 for all i
        matrix.index_fill_(1, indices, 0.0) # sets matrix[:, i] = 0.0 for all i
        matrix.diagonal().index_fill_(0, indices, 1.0) # sets matrix[i, i] = 1.0
        return matrix

    def gradient(
        self, local_energies_v: np.ndarray, 
        local_energies_a: np.ndarray,
        weights=np.ndarray, 
        compute_jacobian=True
    ) -> Tensor:
        r"""Computes the gradient of ``⟨H⟩`` with respect to neural network
        parameters given local energies ``{⟨s|H|ψ⟩/⟨s|ψ⟩ | s ~ |ψ|²}``.
        """
        if not compute_jacobian:
            raise NotImplementedError()
        
        self.__compute_jacobians()
        with torch.no_grad():
            ### now jacobian conlains in [..., 0] == d_k \log \varphi(j), [..., 1] = d_k \log p(j)
            self.O_a_der = torch.einsum('i,ij->ij', (1 - 2 * self.p) / (1 - self.p), self._jacobian[..., 1]) + \
                           2 * self._jacobian[..., 0]
            self.O_v_der = torch.einsum('i,ij->ij', 4 * self.p / (2 * self.p - 1), self._jacobian[..., 1]) + \
                           2 * self._jacobian[..., 0]

            # self.O_a_der * self.weight_a + self.O_v_der * self.weight_v = 2 * self.jacobian[..., 0]
            self.O_mean = torch.einsum('i,ij->j', weights, 2 * self._jacobian[..., 0]) # it is faster to compute O_mean explicitly

            # \sum_j H_ij (2 p_i - 1) (2 p_j - 1) phi_j / phi_i + H_ii * 4 p_i (1 - p_i)
            H_mean = torch.sum((local_energies_v * self.weight_v + local_energies_a * self.weight_a) * weights)
            print(H_mean.item(), self.O_v_der.size(0), torch.min(self.p), torch.max(self.p))
            #O_a = torch.einsum('i,ij->ij', 4 * self.p * (1 - 2 * self.p), self._jacobian[..., 1]) + \
            #      torch.einsum('i,ij->ij', 8 * self.p * (1 - self.p), self._jacobian[..., 0])
            #O_v = torch.einsum('i,ij->ij', 4 * self.p * (2 * self.p - 1), self._jacobian[..., 1]) + \
            #      torch.einsum('i,ij->ij', 2 * (2 * self.p - 1) ** 2, self._jacobian[..., 0])

            OH_mean = torch.einsum('ij,i,i,i->j', self.O_a_der, local_energies_a, self.weight_a, weights) + \
                      torch.einsum('ij,i,i,i->j', self.O_v_der, local_energies_v, self.weight_v, weights)

            OH_correlator = OH_mean - self.O_mean * H_mean

            return OH_correlator

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

        matrix = self.__remove_singularity(matrix)
        # The following is an implementation of Eq. (6.51) and (6.52)
        # from "Quantum Monte Carlo Approaches for Correlated Systems"
        # by F.Becca & S.Sorella.
        if scale_inv_reg is not None:
            # diag <- sqrt(diag(|S|))
            diag = torch.sqrt(torch.abs(torch.diagonal(matrix)))
            vector_pc = vector / diag
            # S_pc[m, n] <- S[m, n] / sqrt(diag[m] * diag[n])
            inv_diag = 1.0 / diag
            matrix_pc = torch.einsum('i,ij,j->ij', inv_diag, matrix, inv_diag)
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

    def solve(self, gradient: Tensor, scale_inv_reg=None, diag_reg=1e-2, weights=None) -> Tensor:
        r"""Given the gradient of ``⟨H⟩``, calculates
        ``S⁻¹⟨H⟩ = ⟨(O - ⟨O⟩)†(O - ⟨O⟩)⟩⁻¹⟨H⟩``.
        """
        if scale_inv_reg is not None and diag_reg is not None:
            raise ValueError("scale_inv_reg and diag_reg are mutually exclusive")
        with torch.no_grad():
            OO_corr = (torch.einsum('i,i,ij,ik->jk', weights, self.weight_a, self.O_a_der, self.O_a_der) + \
                      torch.einsum('i,i,ij,ik->jk', weights, self.weight_v, self.O_v_der, self.O_v_der)) - \
                      torch.einsum('i,j->ij', self.O_mean, self.O_mean)
            OO_corr = self.__remove_singularity(OO_corr)

            if scale_inv_reg is not None:
                # diag <- sqrt(diag(|S|))
                diag = torch.sqrt(torch.abs(torch.diagonal(OO_corr)))
                vector_pc = gradient / diag
                # S_pc[m, n] <- S[m, n] / sqrt(diag[m] * diag[n])
                inv_diag = 1.0 / diag
                matrix_pc = torch.einsum('i,ij,j->ij', inv_diag, OO_corr, inv_diag)
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
                OO_corr.diagonal()[:] += diag_reg
            try:
                u = torch.cholesky(OO_corr)
            except RuntimeError as e:
                print("Warning :: {} Retrying with bigger diagonal shift...".format(e))
                print(OO_corr.diagonal()[:])
                OO_corr.diagonal()[:] += diag_reg
                u = torch.cholesky(OO_corr)
            x = torch.cholesky_solve(gradient.view([-1, 1]), u).squeeze()
            return x

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
        for p in self.amplitude.parameters():
            assert p.is_contiguous()
        for p in self.phase.parameters():
            assert p.is_contiguous()

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
                    torch.from_numpy(self.basis.states.view(np.int64)).to(self.device),
                    8192,
                )
                state[:, 0] -= torch.max(state[:, 0])
                state = state.cpu().numpy().view(np.complex64).squeeze()
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
                mode="exact",
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


class RunnerClassifier(Runner):
    def __init__(self, config):
        super().__init__(config)
        self.composite_state = combine_amplitude_and_sign_classifier(self.amplitude, self.phase)

    def monte_carlo(self):
        with torch.no_grad():  # for Classifier this part remains intact
            if not self.config.full_basis:
                spins, _ = sample_some(
                    lambda x: self.amplitude(unpack(x, self.basis.number_spins)),
                    self.basis,
                    self.sampling_options,
                    mode="exact",
                )

                self.weights = 1. * torch.ones(len(spins)) / len(spins)
            else:
                spins = self.basis.states
                spins = torch.from_numpy(spins.view(np.int64))
                self.weights = torch.squeeze(torch.exp(self.amplitude(unpack(spins, self.basis.number_spins))) ** 2)
            
            self.p = torch.squeeze(self.phase(unpack(spins, self.basis.number_spins))).numpy()
            self.weights = self.weights / torch.sum(self.weights).item()

        '''
            to compute the local energies, one considers with the density matrix two contributions:
                1) <4 p_i (1 - p_i) H_ii>_i; 
                2) <(2 p_i - 1) / phi_i [\sum_j H_ij \phi_j (2 p_j - 1)]>_i
            the (1) contribution is the diagonal term is done in Tom's Heisenberg hamiltonian
            the (2) contribution: Tom can predict \sum H_ij \phi_j, so I need to forge the networks that will 
                                  predict |\phi_j (2 p_j - 1)| and its sign (non-differentiable) +/-
        '''

        #  obtain just H_ii for all i in spins (part of the term (1))
        # this shoulw be later multiplied by w_a = 4 p_i (1 - p_i)
        local_energies_a = local_values_diagonal(spins, self.hamiltonian).numpy()[..., 0].real  # always real


        # the composive-state cooked-up network predicts [\log \varphi_i + \log |2 p_i - 1|, sign(2 p_i - 1)]
        # local_values outputs \sum_j H_ij [phi_j (2 p_j - 1)] / [phi_i (2 p_i - 1)]
        # this should be later multiplied by w_v = (2 p_i - 1) ** 2
        local_energies_v = local_values(
            spins, self.hamiltonian, self.combined_state, batch_size=8192
        )[..., 0].real  # always real

        # \sum_j H_ij (2 p_i - 1) (2 p_j - 1) phi_j / phi_i + H_ii * 4 p_i (1 - p_i)
        weights_a = 4 * self.p * (1 - self.p)
        weights_v = (2. * self.p - 1) ** 2
        energies_weighted = local_energies_v * weights_v + local_energies_a * weights_a
        print(local_energies_a.shape, local_energies_v.shape, weights_a.shape, weights_v.shape, self.weights.size())
        energy = np.dot(self.weights.numpy(), energies_weighted)
        variance = np.dot(self.weights.numpy(), np.abs(energies_weighted - energy) ** 2)

        self.tb_writer.add_scalar("SR/energy", energy.real, self._iteration)
        self.tb_writer.add_scalar("SR/variance", variance, self._iteration)

        #if self.compute_overlap is not None:
        #    raise NotImplementedError()

        logarithmic_derivatives = LogarithmicDerivativesClassifier(
            (self.amplitude, self.phase), unpack(spins, self.basis.number_spins)
        )
        force = logarithmic_derivatives.gradient(torch.from_numpy(local_energies_v), torch.from_numpy(local_energies_a), self.weights)
        self.tb_writer.add_scalar("SR/grad", torch.norm(force), self._iteration)
        delta = logarithmic_derivatives.solve(force, weights=self.weights)
        return force, delta



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
        "classifier",
        "full_basis",
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
    if config.classifier:
        runner = RunnerClassifier(config)
    else:
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
