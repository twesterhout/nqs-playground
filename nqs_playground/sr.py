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

from collections import namedtuple
import math
import os
import pickle
import sys
import tempfile
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import scipy
import torch

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
            module(xs),
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        torch.cat([dw.view(-1) for dw in dws], out=out[i])
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


def CombiningState(amplitude, phase):
    class CombiningState(torch.nn.Module):
        def __init__(self, amplitude, phase):
            super().__init__()
            self.amplitude = amplitude
            self.phase = phase

        def forward(self, x):
            return torch.cat([self.amplitude.forward(x), self.phase.forward(x)], dim=1)

    return torch.jit.script(CombiningState(amplitude, phase))


def local_energy(
    ψ: torch.jit.ScriptModule,
    hamiltonian: _C.Heisenberg,
    σ: np.ndarray,
    log_σψ: Optional[np.ndarray] = None,
    batch_size: int = 128,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError(
            "invalid batch_size: {}; expected a positive integer".format(batch_size)
        )

    # Since torch.jit.ScriptModules can't be directly passed to C++
    # code as torch::jit::script::Modules, we first save ψ to a
    # temporary file and then load it back in C++ code.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        filename = f.name
    try:
        with torch.no_grad():
            ψ.save(filename)
            if log_σψ is None:
                log_σψ = ψ.forward(_C.unpack(σ)).numpy().view(np.complex64)
            log_σHψ = (
                _C.PolynomialState(
                    _C.Polynomial(hamiltonian, [0.0]),
                    filename,
                    (batch_size, len(_C.unsafe_get(σ, 0))),
                )(σ)
                .numpy()
                .view(np.complex64)
            )
            return np.exp(log_σHψ - log_σψ).squeeze(axis=1)
    finally:
        os.remove(filename)


def energy_gradient(
    local_energies: np.ndarray,
    logarithmic_derivatives: np.ndarray,
    weights: np.ndarray = None,
) -> np.ndarray:
    if weights is not None:
        assert np.isclose(np.sum(weights), 1.0)
        gradient = (weights * local_energies).T @ logarithmic_derivatives
    else:
        gradient = local_energies.T @ logarithmic_derivatives
        gradient /= local_energies.shape[0]
    return np.ascontiguousarray(2.0 * gradient.real)


def covariance_matrix(
    logarithmic_derivatives: np.ndarray, weights: np.ndarray = None
) -> np.ndarray:
    if weights is not None:
        assert np.isclose(np.sum(weights), 1.0)
        matrix = (
            weights.reshape(-1, 1) * logarithmic_derivatives
        ).T.conj() @ logarithmic_derivatives
    else:
        matrix = logarithmic_derivatives.T @ logarithmic_derivatives
        matrix /= logarithmic_derivatives.shape[0]
    return np.ascontiguousarray(matrix.real)


class Runner:
    def __init__(self, config):
        self.config = config
        self.__load_hamiltonian()
        self.__load_model()
        self.__load_optimiser()
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

    def monte_carlo(self):
        with torch.no_grad():
            spins = _C.all_spins(self.number_spins, self.magnetisation)
            inputs = _C.unpack(spins)
            weights = self.amplitude.forward(inputs)
            weights -= torch.max(weights)
            torch.exp_(2 * weights)
            weights /= weights.sum()
            weights = weights.numpy().squeeze()

        local_energies = local_energy(
            CombiningState(self.amplitude, self.phase), self.hamiltonian, spins
        )
        E = weights * local_energies
        print("{}, E = {} ± {}".format(E.shape, np.sum(E), len(weights) * np.std(E)))
        logarithmic_derivatives = logarithmic_derivative(
            (self.amplitude, self.phase), inputs
        )
        logarithmic_derivatives -= (weights.T @ logarithmic_derivatives).reshape(1, -1)
        force = -energy_gradient(local_energies, logarithmic_derivatives, weights)
        S = covariance_matrix(logarithmic_derivatives, weights)
        return force, S

    def load_checkpoint(self, i: int):
        def load(target, model):
            pattern = os.path.join(self.config.output, str(i), target, "best_model_*")
            [filename] = glob.glob(pattern)
            model.load_state_dict(torch.load(filename))

        load("amplitude", self.amplitude)
        load("sign", self.sign)

    def solve(self, matrix, vector):
        n = matrix.shape[0]
        diag = np.diag(matrix)
        diag = np.where(diag <= 0.0, 1.0, diag)
        diag = np.sqrt(diag)
        
        matrix = matrix / (diag.reshape(-1, 1) @ diag.reshape(1, -1)) 
        vector = vector / diag

        matrix += 1e-3 * np.eye(matrix.shape[0])
        # print(matrix.shape, vector.shape)
        x = scipy.linalg.solve(matrix, vector)
        x /= diag
        return x

    def set_gradient(self, grad):
        i = 0
        inputs = _C.unpack(_C.all_spins(self.number_spins, self.magnetisation))
        # self.amplitude.forward(inputs).sum().backward()
        with torch.no_grad():
            for p in self.amplitude.parameters():
                n = p.numel()
                if p.grad is not None:
                    p.grad.data.copy_(grad[i : i + n].view(p.size()))
                else:
                    p.grad = grad[i : i + n].view(p.size())
                i += n

        self.phase.forward(inputs).sum().backward()
        for dp in map(
            lambda p_: p_.grad.data.view(-1),
            filter(lambda p_: p_.requires_grad, self.phase.parameters()),
        ):
            n = dp.numel()
            dp.copy_(grad[i : i + n])
            i += n

    def step(self):
        force, S = self.monte_carlo()
        dx = self.solve(S, force)
        self.set_gradient(torch.from_numpy(dx))
        self.optimiser.step()
        if self.compute_overlap is not None:
            print(self.compute_overlap())
        self._iteration += 1


class Optimiser(object):
    def __init__(
        self,
        machine,
        hamiltonian,
        magnetisation,
        epochs,
        monte_carlo_steps,
        learning_rate,
        use_sr,
        regulariser,
        model_file,
        time_limit,
    ):
        self._machine = machine
        self._hamiltonian = hamiltonian
        self._magnetisation = magnetisation
        self._epochs = epochs
        self._monte_carlo_steps = monte_carlo_steps
        self._learning_rate = learning_rate
        self._use_sr = use_sr
        self._model_file = model_file
        self._time_limit = time_limit
        if use_sr:
            self._regulariser = regulariser
            self._delta = None
            self._optimizer = torch.optim.SGD(
                self._machine.parameters(), lr=self._learning_rate
            )
        else:
            self._optimizer = torch.optim.Adam(
                self._machine.parameters(), lr=self._learning_rate
            )

    def learning_cycle(self, iteration):
        logging.info("==================== {} ====================".format(iteration))
        # Monte Carlo
        spin = random_spin(self._machine.number_spins, self._magnetisation)
        (Os, mean_O, E, var_E, F) = monte_carlo(
            self._machine, self._hamiltonian, spin, self._monte_carlo_steps
        )
        logging.info("E = {}, Var[E] = {}".format(E, var_E))
        # Calculate the "true" gradients
        if self._use_sr:
            # We also cache δ to use it as a guess the next time we're computing
            # S⁻¹F.
            self._delta = DenseCovariance(
                Os, mean_O, self._regulariser(iteration)
            ).solve(F, x0=self._delta)
            delta_norm = np.linalg.norm(self._delta)
            # if delta_norm > 10:
            #     self._delta /= (delta_norm / 10)
            self._machine.set_gradients(self._delta)
            logging.info("∥F∥₂ = {}, ∥δ∥₂ = {}".format(np.linalg.norm(F), delta_norm))
        else:
            self._machine.set_gradients(F.real)
            logging.info(
                "∥F∥₂ = {}, ∥Re[F]∥₂ = {}".format(
                    np.linalg.norm(F), np.linalg.norm(F.real)
                )
            )
        # Update the variational parameters
        self._optimizer.step()
        self._machine.clear_cache()

    def __call__(self):
        if self._model_file is not None:

            def save():
                # NOTE: This is important, because we want to overwrite the
                # previous weights
                self._model_file.seek(0)
                self._model_file.truncate()
                torch.save(self._machine.state_dict(), self._model_file)

        else:
            save = lambda: None
        if self._time_limit is not None:
            start = time.time()
            for i in range(self._epochs):
                if time.time() - start > self._time_limit:
                    save()
                    start = time.time()
                self.learning_cycle(i)
        else:
            for i in range(self._epochs):
                self.learning_cycle(i)
        save()
        return self._machine


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
    config = Config(
        model=("example/1x10/amplitude_wip.py", "example/1x10/phase_wip.py"),
        hamiltonian="/vol/tcm01/westerhout_tom/nqs-playground/data/1x10/hamiltonian.txt",
        epochs=50,
        number_samples=1000,
        number_chains=2,
        output="swo/run/3",
        exact="/vol/tcm01/westerhout_tom/nqs-playground/data/1x10/ground_state.pickle",
        optimiser="lambda p: torch.optim.RMSprop(p, lr=1e-3)",
    )

    if True:
        # Running the simulation
        runner = Runner(config)
        for i in range(10000):
            runner.step()


if __name__ == "__main__":
    main()
