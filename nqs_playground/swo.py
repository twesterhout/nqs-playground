#!/usr/bin/env python3

from collections import namedtuple
import glob
from math import sqrt
from math import pi as PI
import os
import pickle
import pwd
import sys
import tempfile
import time
from typing import Dict, List, Tuple, Optional

import cProfile
import numpy as np

import ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import ignite.contrib.handlers.tqdm_logger

import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from . import _C
from .hamiltonian import read_hamiltonian
from . import core as _core
from .v2.supervised import subtract


class SamplingOptions:
    r"""Options for Monte Carlo sampling spin configurations."""

    def __init__(
        self,
        number_samples: int,
        number_chains: int = 1,
        number_discarded: Optional[int] = None,
    ):
        r"""Initialises the options.

        :param number_samples: specifies the number of samples per Markov
            chain. Must be a positive integer.
        :param number_chains: specifies the number of independent Markov
            chains. Must be a positive integer.
        :param number_discarded: specifies the number of samples to discard
            in the beginning of each Markov chain (i.e. how long should the
            thermalisation procedure be). If specified, must be a positive
            integer. Otherwise, 10% of ``number_samples`` is used.
        """
        self.number_samples = int(number_samples)
        if self.number_samples <= 0:
            raise ValueError(
                "invalid number_samples: {}; expected a positive integer"
                "".format(number_samples)
            )
        self.number_chains = int(number_chains)
        if self.number_chains <= 0:
            raise ValueError(
                "invalid number_chains: {}; expected a positive integer"
                "".format(number_chains)
            )
        if number_discarded is not None:
            self.number_discarded = int(number_discarded)
            if self.number_discarded <= 0:
                raise ValueError(
                    "invalid number_discarded: {}; expected either a positive "
                    "integer or None".format(number_chains)
                )
        else:
            self.number_discarded = self.number_samples // 10


def forward_with_batches(f, xs, batch_size: int):
    r"""Applies ``f`` to all ``xs`` propagating no more than ``batch_size`` at
    a time. ``xs`` is split into batches along the first dimension (i.e.
    dim=0).
    """
    n = xs.shape[0]
    if n == 0:
        raise ValueError("invalid xs: {}; input should not be empty".format(xs))
    if batch_size <= 0:
        raise ValueError(
            "invalid batch_size: {}; expected a positive integer".format(batch_size)
        )
    i = 0
    out = []
    while i + batch_size <= n:
        out.append(f(xs[i : i + batch_size]))
        i += batch_size
    if i != n:  # Remaining part
        out.append(f(xs[i:]))
    return torch.cat(out, dim=0)


def sample_some(
    log_ψ: torch.jit.ScriptModule, basis, options: SamplingOptions, mode: str = "exact"
) -> Tuple[np.ndarray, torch.FloatTensor, float]:
    r"""Samples states from the Hilbert space basis ``basis`` according to the
    probability distribution proportional to ‖ψ‖².

    There are three modes of operation:

    1. ``mode == "exact"`` (Exact sampling). This means the we compute
       ``‖ψ(s)‖²`` for all states ``s`` in ``basis`` and then directly sample
       from this discrete probability distribution.

       Number of samples is ``options.number_chains * options.number_samples``,
       and ``options.number_discarded`` is ignored, since there is no need for
       thermalisation.

    2. ``mode == "full"`` (Use the full Hilbert space basis). This means that
       we simply return ``basis.states``.

       In this mode, ``options`` is completely ignored.

    3. ``mode == "monte_carlo"`` (Use Monte Carlo for sampling). This means
       that Metropolis-Hasting algorithm is used to sample basis states.

       **Not yet implemented**
    
    """
    if mode == "exact":
        basis.build()  # Initialises internal cache if not done already
        # Compute log amplitudes
        with torch.no_grad(), torch.jit.optimized_execution(True):
            xs = basis.states
            ys = forward_with_batches(
                lambda xs: log_ψ(_C.v2.unpack(xs, basis.number_spins)),
                xs,
                batch_size=8192,
            ).squeeze()
            if ys.dim() != 1:
                raise ValueError(
                    "log_ψ should return real part of the logarithm of the "
                    "wavefunction, but output tensor has dimension {}".dim(ys.dim())
                )
            # Convert them to probabilities
            probabilities = _core._log_amplitudes_to_probabilities(ys)
            if len(probabilities) < (1 << 24):
                # Sample from discrete distribution
                indices = torch.multinomial(
                    probabilities,
                    num_samples=options.number_chains * options.number_samples,
                    replacement=True,
                )
            else:
                probabilities = probabilities.to(torch.float64)
                probabilities /= torch.sum(probabilities)
                indices = np.random.choice(
                    len(probabilities),
                    size=options.number_chains * options.number_samples,
                    replace=True,
                    p=probabilities,
                )
            return xs[indices], ys[indices], None
    elif mode == "full":
        basis.build()  # Initialises internal cache if not done already
        # Compute log amplitudes
        with torch.no_grad(), torch.jit.optimized_execution(True):
            xs = basis.states
            ys = forward_with_batches(
                lambda xs: log_ψ(_C.v2.unpack(xs, basis.number_spins)),
                xs,
                batch_size=8192,
            ).squeeze()
            if ys.dim() != 1:
                raise ValueError(
                    "log_ψ should return real part of the logarithm of the "
                    "wavefunction, but output tensor has dimension {}".dim(ys.dim())
                )
            return xs, ys, None
        pass
    elif mode == "monte_carlo":
        raise NotImplementedError()
    else:
        raise ValueError(
            'invalid mode: {!r}; expected either "exact", "full", or '
            '"monte_carlo"'.format(mode)
        )


def apply_polynomial(
    polynomial: _C.Polynomial,
    spins: np.ndarray,
    ψ: torch.jit.ScriptModule,
    batch_size: int = 128,
) -> torch.Tensor:
    r"""Computes log(⟨σ|P|ψ⟩) for every |σ⟩ in ``spins``.
    
    :param polynomial: Polynomial P to apply to |ψ⟩.
    :param spins: An array of :py:class:`CompactSpin` for which to compute the
        coefficients.
    :param ψ: Current state. Given a ``torch.FloatTensor`` of shape
        ``(batch_size, num_spins)`` ψ must return a ``torch.FloatTensor`` of
        shape ``(batch_size, 2)``. Columns of the output tensor are interpreted
        as real and imaginary parts of log(⟨σ|ψ⟩).
    """
    if not isinstance(ψ, torch.jit.ScriptModule):
        raise TypeError(
            "ψ has wrong type: {}; expected torch.jit.ScriptModule".format(type(ψ))
        )
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
        ψ.save(filename)
        return _C.PolynomialState(
            polynomial, filename, (32, len(_C.unsafe_get(spins, 0)))
        )(spins)
    finally:
        os.remove(filename)


@torch.jit.script
def safe_real_exp(values: torch.Tensor) -> torch.Tensor:
    assert values.dim() == 2 and values.size(1) == 2
    amplitude = values[:, 0]
    amplitude -= torch.max(amplitude)
    torch.exp_(amplitude)
    phase = values[:, 1]
    phase /= 3.141592653589793
    torch.round_(phase)
    torch.abs_(phase)
    phase = torch.fmod(phase, 2.0)
    return amplitude * (1.0 - 2.0 * phase)


def _create_sign_trainer(
    model: torch.nn.Module, optimiser: torch.optim.Optimizer
) -> ignite.engine.Engine:
    r"""Creates an engine for training the network on the sign structure.

    Cross entropy is used as loss function.

    :param model: PyTorch model to train. Given a ``torch.FloatTensor`` of
        shape ``(batch_size, num_spins)``, ``model`` must return a
        ``torch.FloatTensor`` of shape ``(batch_size, 2)``. Columns of the
        output tensor are interpreted as unnormalised probabilities of spin
        configurations having signs ``+1`` or ``-1``.
    """
    return create_supervised_trainer(
        model, optimiser, loss_fn=torch.nn.CrossEntropyLoss()
    )


def _create_sign_evaluator(model: torch.nn.Module) -> ignite.engine.Engine:
    r"""Creates an engine for evaluating how well the network learned the sign
    structure.
    """
    return create_supervised_evaluator(
        model,
        metrics={
            "cross_entropy": ignite.metrics.Loss(torch.nn.CrossEntropyLoss()),
            "accuracy": ignite.metrics.Accuracy(
                lambda output: (torch.argmax(output[0], dim=1), output[1])
            ),
        },
    )


def _create_amplitude_trainer(model, optimiser):
    r"""Creates an engine for training the network on wavefunction amplitudes."""

    @torch.jit.script
    def loss_fn(predicted, expected):
        predicted = predicted.squeeze()
        expected = expected.squeeze()
        quotient = predicted / expected
        return 1 - torch.sum(quotient) ** 2 / torch.norm(quotient) ** 2

    # @torch.jit.script
    # def loss_fn(predicted: torch.Tensor, expected: torch.Tensor):
    #     predicted = predicted.squeeze()
    #     return (
    #         1
    #         - torch.dot(predicted, expected) ** 2
    #         / torch.norm(predicted) ** 2
    #         / torch.norm(expected) ** 2
    #     )

    return create_supervised_trainer(model, optimiser, loss_fn=loss_fn)


def _create_amplitude_evaluator(model):
    return create_supervised_evaluator(model, metrics={"overlap": OverlapMetric()})


def create_trainer(
    target: str, model: torch.nn.Module, optimiser: torch.optim.Optimizer
) -> ignite.engine.Engine:
    r"""Creates an engine for optimising either the amplitude or the sign of the
    wavefunction.

    :param target: either "amplitude" or "sign".
    :param model: PyTorch model to train.
    :param optimiser: PyTorch optimiser to use for training.
    """
    return {"sign": _create_sign_trainer, "amplitude": _create_amplitude_trainer}[
        target
    ](model, optimiser)


def create_evaluators(target: str, model: torch.nn.Module) -> ignite.engine.Engine:
    make = {"sign": _create_sign_evaluator, "amplitude": _create_amplitude_evaluator}[
        target
    ]
    return namedtuple("Evaluators", ["training", "validation"])(
        make(model), make(model)
    )


class OverlapMetric(ignite.metrics.metric.Metric):
    r"""An Ignite metric for computing overlap of with the target state.
    """

    def __init__(self, output_transform=lambda x: x):
        self._dot = None
        self._norm_predicted = None
        self._norm_expected = None
        super().__init__(output_transform=output_transform)

    def reset(self):
        self._dot = 0
        self._norm_predicted = 0
        self._norm_expected = 0

    def update(self, output):
        predicted, expected = output
        predicted = predicted.squeeze()
        expected = expected.squeeze()
        _quotient = predicted / expected
        self._dot += torch.sum(_quotient).item()
        self._norm_predicted += torch.norm(_quotient).item() ** 2
        self._norm_expected = 1

    def compute(self):
        if self._norm_expected == 0:
            raise ignite.exceptions.NotComputableError(
                "OverlapMetric must have at least one example before it can be computed."
            )
        return abs(self._dot) / sqrt(self._norm_predicted * self._norm_expected)


class _Trainer(object):
    def __init__(self, target, model, dataset, basis, output, config):
        self.target = target
        self.output = output
        self.config = config
        self.prefix = "best"
        self.basis = basis
        self.__add_model(model, dataset)
        self.optimiser = config.optimiser(self.model)
        self.__add_loaders(dataset)
        self.__add_engines()
        self.__add_loggers()
        self.__add_handlers()

    def __add_model(self, model, dataset):
        if self.target == "sign":
            self.model = model
        else:

            class _Model(torch.nn.Module):
                __constants__ = ["scale"]

                def __init__(self, module, scale):
                    super().__init__()
                    self.module = module
                    self.scale = scale

                def forward(self, x):
                    return torch.exp(self.module(x) - self.scale)

            xs, _ = dataset
            ys = forward_with_batches(
                lambda x: model(_C.v2.unpack(x, self.basis.number_spins)),
                xs,
                batch_size=8192,
            )
            scale = float(torch.max(ys)) - 5.0
            self.original_model = model
            self.model = torch.jit.script(_Model(model, scale))

    def __add_loaders(self, dataset: Tuple[torch.Tensor, torch.Tensor]):
        with torch.no_grad():
            x, y = dataset
            if self.target == "sign":
                y = (y < 0).to(torch.long)
            else:
                y = torch.abs(y).squeeze()
            dataset = (x, y)

        Loaders = namedtuple("Loaders", ["training", "validation"], defaults=[None])
        # Special case when we don't want to use validation dataset
        if self.config.train_fraction == 1.0:
            self.loaders = Loaders(
                _core.make_spin_dataloader(
                    *dataset,
                    batch_size=self.config.train_batch_size,
                    drop_last=True,
                    unpack=lambda x, i: _C.v2.unpack(x, i, self.basis.number_spins),
                )
            )
        else:
            train, val = _core.random_split(
                dataset, self.config.train_fraction, weights=None, replacement=False
            )
            val = subtract(val, train)
            self.loaders = Loaders(
                _core.make_spin_dataloader(
                    *train,
                    batch_size=self.config.train_batch_size,
                    drop_last=True,
                    unpack=lambda x, i: _C.v2.unpack(x, i, self.basis.number_spins),
                ),
                _core.make_spin_dataloader(
                    *val,
                    batch_size=self.config.val_batch_size,
                    unpack=lambda x: _C.v2.unpack(x, self.basis.number_spins),
                    shuffle=False,
                ),
            )

    def __add_engines(self):
        self.trainer = create_trainer(self.target, self.model, self.optimiser)
        self.evaluators = create_evaluators(self.target, self.model)

    def __add_loggers(self):
        self.tb_writer = SummaryWriter(log_dir=self.output, max_queue=1000)
        self.tqdm_writer = ignite.contrib.handlers.tqdm_logger.ProgressBar()

    def __add_handlers(self):
        self.tqdm_writer.attach(
            self.trainer,
            event_name=Events.EPOCH_COMPLETED,
            closing_event_name=Events.COMPLETED,
        )

        # TODO(twesterhout): Check how badly this hurts performance
        # @self.trainer.on(Events.ITERATION_COMPLETED)
        # def log_loss(engine):
        #     self.tb_writer.add_scalar(
        #         "training/loss", engine.state.output, engine.state.iteration
        #     )

        self.__add_evaluators()
        self.__add_checkpoint()
        self.__add_early_stopping()

    def __add_evaluators(self):
        def add(tag, evaluator, dataloader):
            def log_results(engine):
                n = engine.state.epoch
                evaluator.run(dataloader)
                for name, value in evaluator.state.metrics.items():
                    self.tb_writer.add_scalar("{}/{}".format(tag, name), value, n)

            self.trainer.add_event_handler(Events.STARTED, log_results)
            self.trainer.add_event_handler(Events.EPOCH_COMPLETED, log_results)

        add("training", self.evaluators.training, self.loaders.training)
        if self.loaders.validation is not None:
            add("validation", self.evaluators.validation, self.loaders.validation)

    def __score_fn(self):
        if self.target == "sign":
            return lambda e: -e.state.metrics["cross_entropy"]
        else:
            return lambda e: e.state.metrics["overlap"]

    def __add_checkpoint(self):
        evaluator = (
            self.evaluators.validation
            if self.loaders.validation is not None
            else self.evaluators.training
        )
        evaluator.add_event_handler(
            Events.COMPLETED,
            ignite.handlers.ModelCheckpoint(
                dirname=self.output,
                filename_prefix=self.prefix,
                score_function=self.__score_fn(),
                require_empty=True,
                n_saved=1,
                save_as_state_dict=True,
            ),
            {"model": self.model},
        )

    def load_best(self):
        matched = [f for f in os.listdir(self.output) if f.startswith(self.prefix)]
        if len(matched) == 0:
            raise ValueError(
                "Could not load the checkpoint. There are no files "
                "prefixed with {!r} in the directory {!r}."
                "".format(self.prefix, self.output)
            )
        if len(matched) > 1:
            raise ValueError(
                "Could not load the checkpoint. There are more than "
                "one files prefixed with {!r} in the directory {!r}."
                "".format(self.prefix, self.output)
            )
        matched = next(iter(matched))
        self.model.load_state_dict(torch.load(os.path.join(self.output, matched)))

    def __add_early_stopping(self):
        self.evaluators.validation.add_event_handler(
            Events.COMPLETED,
            ignite.handlers.EarlyStopping(
                patience=self.config.patience,
                score_function=self.__score_fn(),
                trainer=self.trainer,
            ),
        )

    def run(self):
        self.trainer.run(self.loaders.training, max_epochs=self.config.max_epochs)
        self.load_best()
        if self.target == "amplitude":
            state_dict = self.model.state_dict()
            for name, tensor in self.original_model.state_dict().items():
                assert torch.all(tensor == state_dict["module." + name])


def train(target, model, dataset, basis, output, config):
    _Trainer(target, model, dataset, basis, output, config).run()


class Runner(object):
    def __init__(self, config):
        self.config = config
        self.amplitude, self.sign = self.config.model
        self.polynomial = _C.v2.Polynomial(self.config.hamiltonian, self.config.roots)
        self.sampling_options = SamplingOptions(
            self.config.number_samples, self.config.number_chains
        )
        self.exact = self.config.exact
        self.tb_writer = SummaryWriter(log_dir=self.config.output)
        self._iteration = 0

    def compute_statistics(self):
        if self.exact is None:
            return None

        with torch.no_grad(), torch.jit.optimized_execution(True):
            basis = self.polynomial.hamiltonian.basis
            log_ψ = _core.combine_amplitude_and_sign(
                self.amplitude, self.sign, apply_log=False, out_dim=2
            )
            y = forward_with_batches(
                lambda x: log_ψ(_C.v2.unpack(x, basis.number_spins)),
                basis.states,
                batch_size=8192,
            )
            y[:, 0] -= torch.max(y[:, 0])
            y = np.exp(y.numpy().view(np.complex64).squeeze())
            y /= np.linalg.norm(y)

            assert np.isclose(np.linalg.norm(self.exact), 1.0)
            overlap = np.abs(np.dot(y.conj(), self.exact))

            # Hy <- H*y
            n = basis.number_states
            Hy = np.empty(n, dtype=np.complex128)
            self.polynomial.hamiltonian(y, Hy)
            # E <- y.conj() * H * y = y.o
            energy = np.dot(y.conj(), Hy)
            Hy -= energy * y
            variance = np.linalg.norm(Hy) ** 2
            return overlap, energy, variance

    def monte_carlo(self):
        tqdm.write("Monte Carlo sampling from |ψ|²...", end="")
        start = time.time()

        spins, values, _ = sample_some(
            self.amplitude,
            self.polynomial.hamiltonian.basis,
            self.sampling_options,
            mode="exact",
        )
        acceptance = 1.0

        with torch.no_grad(), torch.jit.optimized_execution(True):
            log_ψ = _core.combine_amplitude_and_sign(
                self.amplitude, self.sign, apply_log=False, out_dim=2
            )
            values = forward_with_batches(
                _C.v2.PolynomialState(
                    self.polynomial, log_ψ._c._get_method("forward"), 8192
                ),
                spins,
                batch_size=32,
            )
            values = safe_real_exp(values)

        size = len(set(spins))
        stop = time.time()
        tqdm.write(
            " Done in {:.2f} seconds. ".format(stop - start)
            + "Acceptance {:.2f}%. ".format(100 * acceptance)
            + "Sampled {} spin configurations".format(size)
        )
        return spins, values

    def load_checkpoint(self, i: int):
        def load(target, model):
            pattern = os.path.join(self.config.output, str(i), target, "best_model_*")
            [filename] = glob.glob(pattern)
            model.load_state_dict(torch.load(filename))

        load("amplitude", self.amplitude)
        load("sign", self.sign)

    def step(self):
        dataset = self.monte_carlo()
        basis = self.polynomial.hamiltonian.basis
        train(
            "amplitude",
            self.amplitude,
            dataset,
            basis,
            "{}/{}/amplitude".format(self.config.output, self._iteration),
            self.config.amplitude,
        )
        train(
            "sign",
            self.sign,
            dataset,
            basis,
            "{}/{}/sign".format(self.config.output, self._iteration),
            self.config.sign,
        )
        self._iteration += 1
        metrics = self.compute_statistics()
        if metrics is not None:
            overlap, energy, variance = metrics
            self.tb_writer.add_scalar("overlap", overlap, self._iteration)
            self.tb_writer.add_scalar("energy_real", energy.real, self._iteration)
            self.tb_writer.add_scalar("energy_imag", energy.imag, self._iteration)
            self.tb_writer.add_scalar("variance", variance, self._iteration)


TrainingOptions = namedtuple(
    "TrainingOptions",
    [
        # Batch size to use for training
        "train_batch_size",
        # Max number of epochs. Usually, early stopping will kick in before
        # this number is reached.
        "max_epochs",
        # Fraction of the dataset to use for training. The remaining part will
        # be used as validation dataset for regularisation techniques such as
        # early stopping.
        "train_fraction",
        # Patience for early stopping. If validation loss does not decrease for
        # patience epochs, training loop is terminated.
        "patience",
        # A function which given a torch.jit.ScriptModule returns a
        # torch.optim.Optimizer to be used for training.
        "optimiser",
        # Batch size for validation dataset. Is is a good idea to pick a pretty
        # big size here to eliminate Python overhead as much as possible.
        "val_batch_size",
    ],
    defaults=[
        16,  # train_batch_size
        200,  # max_epochs
        0.90,  # train_fraction
        10,  # patience
        lambda m: torch.optim.RMSprop(
            m.parameters(), lr=1e-3, weight_decay=1e-4
        ),  # optimiser
        1024,  # val_batch_size
    ],
)

Config = namedtuple(
    "Config",
    [
        # A pair of two torch.jit.ScriptModules: (amplitude, sign).
        #
        # amplitude receives as input tensors of shape (batch_size,
        # number_spins) and returns tensors of shape (batch_size, 1)
        # representing **logarithms** of wavefunction amplitudes.
        #
        # sign receives as input tensors of shape (batch_size, number_spins)
        # and returns tensors of shape (batch_size, 2) representing
        # unnormalised probabilities for wavefunction signs being respectively
        # +1 or -1.
        "model",
        # Folder where to save the results.
        "output",
        # Hamiltonian of the system (of type nqs_playground._C.v2.Heisenberg).
        "hamiltonian",
        # Roots of the polynomial.
        "roots",
        # Number of outer iterations of the algorithm.
        "epochs",
        # Number of Monte Carlo samples per Markov chain.
        "number_samples",
        # Number of Markov chains.
        "number_chains",
        # TrainingOptions for amplitude network.
        "amplitude",
        # TrainingOptions for sign network.
        "sign",
        ## OPTIONAL
        ## ====================
        # Location of the exact ground state. If specified, it will be used to
        # compute overlap at each epoch.
        "exact",
        # Sweep size in Metropolis-Hasting algorithm
        # **IGNORED** since Monte Carlo is not yet functional
        "sweep_size",
        # Thermalisation length in Metropolis-Hasting algorithm
        # **IGNORED** since Monte Carlo is not yet functional
        "number_discarded",
    ],
    defaults=[None, None, None],
)


def run(config):
    # Running the simulation
    runner = Runner(config)
    for i in range(config.epochs):
        runner.step()


def main():
    config = Config(
        model=("example/1x16/swo/amplitude.py", "example/1x16/swo/sign.py"),
        hamiltonian="data/1x16/hamiltonian.txt",
        epochs=50,
        roots=[16.0, 16.0, 16.0, 16.0, 16.0, 16.0],
        number_samples=1000,
        number_chains=1,
        output="example/1x16/swo/runs/4",
        exact="data/1x16/ground_state.pickle",
        amplitude=TrainingOptions(
            max_epochs=200,
            patience=20,
            train_fraction=0.9,
            train_batch_size=16,
            optimiser="lambda m: torch.optim.RMSprop(m.parameters(), lr=5e-4, weight_decay=2e-4)",
        ),
        sign=TrainingOptions(
            max_epochs=200,
            patience=20,
            train_fraction=0.9,
            train_batch_size=16,
            optimiser="lambda m: torch.optim.RMSprop(m.parameters(), lr=1e-3, weight_decay=1e-4)",
        ),
    )

    if True:
        # Running the simulation
        runner = Runner(config)
        for i in range(config.epochs):
            runner.step()
    else:
        # Analysis of the results
        runner = Runner(config)
        for i in range(config.epochs):
            runner.load_checkpoint(i)
            tqdm.write("{}\t{}".format(i, runner.compute_overlap()))


if __name__ == "__main__":
    main()
