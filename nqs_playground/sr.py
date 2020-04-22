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

from collections import namedtuple
import os
import time
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

if torch.has_cuda:
    import threading
    from torch._utils import ExceptionWrapper

from nqs_playground import *

Config = namedtuple(
    "Config",
    [
        # 
        "model",
        "output",
        "hamiltonian",
        "epochs",
        "number_samples",
        "number_chains",
        "optimiser",
        ## OPTIONAL
        "classifier",
        "regularisation",
        "device",
        "exact",
        "sampling_mode",
        "sweep_size",
        "number_discarded",
    ],
    defaults=[False, {"scale_inv_reg": 1e-3}, "cpu", None, "monte_carlo", None, None],
)


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

    def _center(self, weights):
        r"""Centers logarithmic derivatives, i.e. ``Oₖ <- Oₖ - ⟨Oₖ⟩``. Repeated
        calls to ``__center`` will do nothing.
        """
        if not self.__is_centered:
            self._compute_jacobians()
            with torch.no_grad():
                self._real -= weights @ self._real
                self._imag -= weights @ self._imag
            self.__is_centered = True

    def gradient(self, local_energies, weights, compute_jacobian=True) -> Tensor:
        r"""Computes the gradient of ``⟨H⟩`` with respect to neural network
        parameters given local energies ``{⟨s|H|ψ⟩/⟨s|ψ⟩ | s ~ |ψ|²}``.
        """
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

        self._center(weights)
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
            real_energies *= weights
            real_energies *= 2
            imag_energies = local_energies[:, 1].view([1, -1]).to(self._imag.device)
            imag_energies *= weights
            imag_energies *= 2
            # scale = 2.0 / self._real.size(0)
            snd_part = torch.mm(imag_energies, self._imag)  # (1xN) x (NxK) -> (1xK)
            # snd_part *= scale
            snd_part = snd_part.to(self._real.device, non_blocking=True)
            fst_part = torch.mm(real_energies, self._real)  # (1xN) x (NxM) -> (1xM)
            # fst_part *= scale
            return torch.cat([fst_part, snd_part], dim=1).squeeze(dim=0)

    def _solve_part(
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
            u = torch.cholesky(matrix_pc)
            x = torch.cholesky_solve(vector_pc.view([-1, 1]), u).squeeze()
            x *= inv_diag
            return x
        # The simpler approach
        if diag_reg is not None:
            matrix.diagonal()[:] += diag_reg
        u = torch.cholesky(matrix)
        x = torch.cholesky_solve(vector.view([-1, 1]), u).squeeze()
        return x

    def solve(
        self, gradient: Tensor, weights, scale_inv_reg=None, diag_reg=None
    ) -> Tensor:
        r"""Given the gradient of ``⟨H⟩``, calculates
        ``S⁻¹⟨H⟩ = ⟨(O - ⟨O⟩)†(O - ⟨O⟩)⟩⁻¹⟨H⟩``.
        """
        @torch.no_grad()
        def task(log_derivatives, operator_gradient, **kwargs) -> Tensor:
            device = log_derivatives.device
            operator_gradient = operator_gradient.to(device, non_blocking=True)
            covariance_matrix = torch.einsum("ij,j,jk",
                log_derivatives.t(), weights, log_derivatives)
            return self._solve_part(covariance_matrix, operator_gradient, **kwargs)

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


def load_exact(ground_state):
    if ground_state is None:
        return None
    if isinstance(ground_state, str):
        # Ground state was saved using NumPy binary format
        ground_state = np.load(ground_state)
        if ground_state.ndim > 1:
            raise ValueError("ground state must be a vector")
    return ground_state.squeeze().astype(np.complex64)


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
            self._derlogphi = jacobian_cpu(
                self._modules[0], self._inputs
            )  # \partial\log\varphi_i
            self._n_phi_weights = self._derlogphi.size(1)

            self._derp = jacobian_cpu(self._modules[1], self._inputs)  # \partial p_i
            self._jacobian = torch.zeros(
                [self._derlogphi.size(0), self._n_phi_weights + self._derp.size(1), 2]
            )

            self.p = torch.squeeze(self._modules[1](self._inputs))
            self._jacobian[:, : self._n_phi_weights, 0] = self._derlogphi
            self._jacobian[:, self._n_phi_weights :, 1] = torch.einsum(
                "ij,i->ij", self._derp, 1.0 / self.p
            )

            self.weight_a = 4.0 * self.p * (1.0 - self.p)
            self.weight_v = (2.0 * self.p - 1.0) ** 2
        else:
            raise NotImplementedError()

    def __remove_singularity(self, matrix: Tensor, eps: float = 1e-4) -> Tensor:
        r"""Removes singularities from ``matrix``. Application of this function to
        covariance matrix essentially means that we exclude some variational
        parameters from optimization.
        """
        # obtains all indices i for which matrix[i, i] < eps
        indices = torch.nonzero(matrix.diagonal().abs() < eps).squeeze(dim=1)
        matrix.index_fill_(0, indices, 0.0)  # sets matrix[i, :] = 0.0 for all i
        matrix.index_fill_(1, indices, 0.0)  # sets matrix[:, i] = 0.0 for all i
        matrix.diagonal().index_fill_(0, indices, 1.0)  # sets matrix[i, i] = 1.0
        return matrix

    def gradient(
        self,
        local_energies_v: np.ndarray,
        local_energies_a: np.ndarray,
        weights=np.ndarray,
        compute_jacobian=True,
    ) -> Tensor:
        r"""Computes the gradient of ``⟨H⟩`` with respect to neural network
        parameters given local energies ``{⟨s|H|ψ⟩/⟨s|ψ⟩ | s ~ |ψ|²}``.
        """
        if not compute_jacobian:
            raise NotImplementedError()

        self.__compute_jacobians()
        with torch.no_grad():
            ### now jacobian conlains in [..., 0] == d_k \log \varphi(j), [..., 1] = d_k \log p(j)
            self.O_a_der = (
                torch.einsum(
                    "i,ij->ij", (1 - 2 * self.p) / (1 - self.p), self._jacobian[..., 1]
                )
                + 2 * self._jacobian[..., 0]
            ) / 2.0
            self.O_v_der = (
                torch.einsum(
                    "i,ij->ij", 4 * self.p / (2 * self.p - 1), self._jacobian[..., 1]
                )
                + 2 * self._jacobian[..., 0]
            ) / 2.0

            # self.O_a_der * self.weight_a + self.O_v_der * self.weight_v = 2 * self.jacobian[..., 0]
            self.O_mean = (
                torch.einsum("i,ij->j", weights, 2 * self._jacobian[..., 0]) / 2.0
            )  # it is faster to compute O_mean explicitly

            # \sum_j H_ij (2 p_i - 1) (2 p_j - 1) phi_j / phi_i + H_ii * 4 p_i (1 - p_i)
            H_mean = torch.sum(
                (local_energies_v * self.weight_v + local_energies_a * self.weight_a)
                * weights
            )

            OH_mean = torch.einsum(
                "ij,i,i,i->j", self.O_a_der, local_energies_a, self.weight_a, weights
            ) + torch.einsum(
                "ij,i,i,i->j", self.O_v_der, local_energies_v, self.weight_v, weights
            )

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
        if scale_inv_reg is not None and diag_reg is not None:
            raise ValueError("scale_inv_reg and diag_reg are mutually exclusive")
        with torch.no_grad():
            OO_corr = (
                torch.einsum(
                    "i,i,ij,ik->jk", weights, self.weight_a, self.O_a_der, self.O_a_der
                )
                + torch.einsum(
                    "i,i,ij,ik->jk", weights, self.weight_v, self.O_v_der, self.O_v_der
                )
            ) - torch.einsum("i,j->ij", self.O_mean, self.O_mean)
            OO_corr = self.__remove_singularity(OO_corr)

            if scale_inv_reg is not None:
                # diag <- sqrt(diag(|S|))
                diag = torch.sqrt(torch.abs(torch.diagonal(OO_corr)))
                vector_pc = gradient / diag
                # S_pc[m, n] <- S[m, n] / sqrt(diag[m] * diag[n])
                inv_diag = 1.0 / diag
                matrix_pc = torch.einsum("i,ij,j->ij", inv_diag, OO_corr, inv_diag)
                # regularizes the preconditioned matrix
                matrix_pc.diagonal()[:] += scale_inv_reg
                # solves the linear system
                try:
                    u = torch.cholesky(matrix_pc)
                except RuntimeError as e:
                    print(
                        "Warning :: {} Retrying with bigger diagonal shift...".format(e)
                    )
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
                OO_corr.diagonal()[:] += diag_reg
                u = torch.cholesky(OO_corr)
            x = torch.cholesky_solve(gradient.view([-1, 1]), u).squeeze()
            return x


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
        if self.config.sampling_mode not in {"monte_carlo", "exact", "full"}:
            raise ValueError(
                "invalid sampling_mode: {}; expected 'monte_carlo', 'exact' or 'full'"
                "".format(self.config.sampling_mode)
            )
        return SamplingOptions(
            self.config.number_samples, self.config.number_chains, device=self.device
        )

    def _using_full_sum(self):
        return self.config.sampling_mode == "full"

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
            spins, log_prob, debug_info = sample_some(
                self.amplitude,
                self.basis,
                self.sampling_options,
                mode=self.config.sampling_mode,
            )
            # Ignoring the fact that we have multiple chains
            spins = spins.view(-1, 8)
            log_prob = log_prob.view(-1)
            if self._using_full_sum():
                weights = log_prob
                weights -= torch.max(weights)
                weights = torch.exp_(weights)
            else:
                weights = torch.ones_like(log_prob)
            weights /= torch.sum(weights)
            log_prob = None

        local_energies = local_values(spins, self.hamiltonian, self.combined_state)
        energy, variance = self._energy_and_variance(local_energies, weights)
        overlap = self.compute_overlap()
        logarithmic_derivatives = LogarithmicDerivatives(
            (self.amplitude, self.phase), spins
        )
        force = logarithmic_derivatives.gradient(local_energies, weights)
        delta = logarithmic_derivatives.solve(force, weights=weights,
                **self.config.regularisation)

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
            for p in filter(lambda x: x.requires_grad, m.parameters()):
                assert p.is_leaf
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

    def save_checkpoint(self):
        torch.save(
            self.amplitude.state_dict(),
            os.path.join(self.config.output, "amplitude_weights_{}.pt".format(self._iteration)),
        )
        torch.save(
            self.phase.state_dict(),
            os.path.join(self.config.output, "sign_weights_{}.pt".format(self._iteration)),
        )

    def step(self):
        tick = time.time()
        force, delta = self.monte_carlo()
        self._set_gradient(delta)
        self.optimiser.step()
        self._iteration += 1
        self.save_checkpoint()
        print("Information :: Done in {} seconds...".format(time.time() - tick))


class RunnerClassifier(Runner):
    def __init__(self, config):
        super().__init__(config)
        self.composite_state = combine_amplitude_and_sign_classifier(
            self.amplitude, self.phase, number_spins=self.basis.number_spins
        )

    def monte_carlo(self):
        with torch.no_grad():  # for Classifier this part remains intact
            if not self.config.full_basis:
                spins, _ = sample_some(
                    lambda x: self.amplitude(unpack(x, self.basis.number_spins)),
                    self.basis,
                    self.sampling_options,
                    mode="exact",
                )

                self.weights = 1.0 * torch.ones(len(spins)) / len(spins)
            else:
                spins = self.basis.states
                spins = torch.from_numpy(spins.view(np.int64))
                self.weights = torch.squeeze(
                    torch.exp(self.amplitude(unpack(spins, self.basis.number_spins)))
                    ** 2
                )
            self.p = torch.squeeze(
                self.phase(unpack(spins, self.basis.number_spins))
            ).numpy()
            self.weights = self.weights / torch.sum(self.weights).item()
            self.logpsi = torch.squeeze(
                self.amplitude(unpack(spins, self.basis.number_spins))
            ).numpy()

        """
            to compute the local energies, one considers with the density matrix two contributions:
                1) <4 p_i (1 - p_i) H_ii>_i; 
                2) <(2 p_i - 1) / phi_i [\sum_j H_ij \phi_j (2 p_j - 1)]>_i
            the (1) contribution is the diagonal term is done in Tom's Heisenberg hamiltonian
            the (2) contribution: Tom can predict \sum H_ij \phi_j, so I need to forge the networks that will 
                                  predict |\phi_j (2 p_j - 1)| and its sign (non-differentiable) +/-
        """

        #  obtain just H_ii for all i in spins (part of the term (1))
        # this shoulw be later multiplied by w_a = 4 p_i (1 - p_i)
        local_energies_a = (
            local_values_diagonal(spins, self.hamiltonian).numpy()[..., 0].real
        )  # always real

        # the composive-state cooked-up network predicts [\log \varphi_i + \log |2 p_i - 1|, sign(2 p_i - 1)]
        # local_values outputs \sum_j H_ij [phi_j (2 p_j - 1)] / [phi_i (2 p_i - 1)]
        # this should be later multiplied by w_v = (2 p_i - 1) ** 2
        local_energies_v = local_values(
            spins, self.hamiltonian, self.composite_state, batch_size=8192
        )[
            ..., 0
        ].real  # always real

        # \sum_j H_ij (2 p_i - 1) (2 p_j - 1) phi_j / phi_i + H_ii * 4 p_i (1 - p_i)
        weights_a = 4 * self.p * (1 - self.p)
        weights_v = (2.0 * self.p - 1) ** 2
        energies_weighted = local_energies_v * weights_v + local_energies_a * weights_a
        energy = np.dot(self.weights.numpy(), energies_weighted)
        variance = np.dot(self.weights.numpy(), np.abs(energies_weighted - energy) ** 2)

        self.tb_writer.add_scalar("SR/energy", energy.real, self._iteration)
        self.tb_writer.add_scalar("SR/variance", variance, self._iteration)

        # if self.compute_overlap is not None:
        #    raise NotImplementedError()

        logarithmic_derivatives = LogarithmicDerivativesClassifier(
            (self.amplitude, self.phase), unpack(spins, self.basis.number_spins)
        )
        force = logarithmic_derivatives.gradient(
            torch.from_numpy(local_energies_v),
            torch.from_numpy(local_energies_a),
            self.weights,
        )
        self.tb_writer.add_scalar("SR/grad", torch.norm(force), self._iteration)
        delta = logarithmic_derivatives.solve(force, weights=self.weights)
        return force, delta


def run(config):
    if config.classifier:
        runner = RunnerClassifier(config)
    else:
        runner = Runner(config)

    for i in range(config.epochs):
        runner.step()
