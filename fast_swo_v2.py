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

from nqs_playground import _C

# from nqs_playground.core import ExplicitState, load_explicit
from nqs_playground.hamiltonian import read_hamiltonian
import nqs_playground.core as _core


def sample_some(
    ψ: torch.jit.ScriptModule, options: _C._Options, explicit: bool = False
) -> Tuple[np.ndarray, torch.FloatTensor, float]:
    r"""Runs Monte Carlo sampling for state ψ.

    If ``explicit`` is ``False``, we sample from ``|ψ|²``. Otherwise, we simply
    use the whole Hilbert space basis (constraining the magnetisation)
    """
    if not explicit:
        # Since torch.jit.ScriptModules can't be directly passed to C++
        # code as torch::jit::script::Modules, we first save ψ to a
        # temporary file and then load it back in C++ code.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f.name
        try:
            ψ.save(filename)
            spins, values, acceptance = _C._sample_some(filename, options)
            values = torch.from_numpy(values)
            return spins, values, acceptance
        finally:
            os.remove(filename)
    else:
        spins = _C.all_spins(options.number_spins, options.magnetisation)
        with torch.no_grad():
            values = ψ.forward(_C.unpack(spins))
        return spins, values, 1.0


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
    def loss_fn(predicted: torch.Tensor, expected: torch.Tensor):
        predicted = predicted.squeeze()
        return (
            1
            - torch.dot(predicted, expected) ** 2
            / torch.norm(predicted) ** 2
            / torch.norm(expected) ** 2
        )

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
        self._dot += torch.dot(predicted, expected).item()
        self._norm_predicted += torch.norm(predicted).item() ** 2
        self._norm_expected += torch.norm(expected).item() ** 2

    def compute(self):
        if self._norm_expected == 0:
            raise ignite.exceptions.NotComputableError(
                "OverlapMetric must have at least one example before it can be computed."
            )
        return abs(self._dot) / sqrt(self._norm_predicted * self._norm_expected)


class _Trainer(object):
    def __init__(self, target, model, dataset, output, config):
        self.target = target
        self.model = model
        self.output = output
        self.config = config
        self.optimiser = eval(config.optimiser)(self.model)
        self.prefix = "best"
        self.__add_loaders(dataset)
        self.__add_engines()
        self.__add_loggers()
        self.__add_handlers()

    def __add_loaders(self, dataset: Tuple[torch.Tensor, torch.Tensor]):
        with torch.no_grad():
            x, y = dataset
            if self.target == "sign":
                y = torch.where(y >= 0, torch.tensor([0]), torch.tensor([1])).squeeze()
            else:
                y = torch.abs(y).squeeze()
            dataset = (x, y)

        Loaders = namedtuple("Loaders", ["training", "validation"], defaults=[None])
        # Special case when we don't want to use validation dataset
        if self.config.train_fraction == 1.0:
            self.loaders = Loaders(
                _core.make_spin_dataloader(
                    *dataset, batch_size=self.config.train_batch_size
                )
            )
        else:
            train, val = _core.random_split(
                dataset, self.config.train_fraction, weights=None, replacement=False
            )
            self.loaders = Loaders(
                _core.make_spin_dataloader(
                    *train, batch_size=self.config.train_batch_size
                ),
                _core.make_spin_dataloader(*val, batch_size=self.config.val_batch_size),
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
        @self.trainer.on(Events.ITERATION_COMPLETED)
        def log_loss(engine):
            self.tb_writer.add_scalar(
                "training/loss", engine.state.output, engine.state.iteration
            )

        self.__add_evaluators()
        self.__add_checkpoint()
        self.__add_early_stopping()

    def __add_evaluators(self):
        def add(tag, evaluator, dataloader):
            @self.trainer.on(Events.EPOCH_COMPLETED)
            def log_results(engine):
                evaluator.run(dataloader)
                for name, value in evaluator.state.metrics.items():
                    self.tb_writer.add_scalar(
                        "{}/{}".format(tag, name), value, engine.state.epoch
                    )

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


def train(target, model, dataset, output, config):
    _Trainer(target, model, dataset, output, config).run()


class Runner(object):
    def __init__(self, config):
        self.config = config
        self.__load_hamiltonian()
        self.__load_model()
        self.mc_options = self.__build_options()
        self.polynomial = self.__build_polynomial()
        self.compute_overlap = self.__load_exact()
        self._iteration = 0

    def __load_hamiltonian(self):
        self.hamiltonian = read_hamiltonian(self.config.hamiltonian)
        self.number_spins = self.hamiltonian.number_spins
        self.hamiltonian = self.hamiltonian.to_cxx()

    def __load_model(self):
        if not isinstance(self.config.model, (tuple, list)):
            raise ValueError(
                "invalid config.model: {}; ".format(self.config.model)
                + "expected a pair of filenames"
            )

        def load(name):
            return torch.jit.script(_core.import_network(name)(self.number_spins))

        amplitude_file, sign_file = self.config.model
        self.amplitude = load(amplitude_file)
        self.sign = load(sign_file)

    def __load_model_v2(self, target):
        def load(name):
            return torch.jit.script(_core.import_network(name)(self.number_spins))

        amplitude_file, sign_file = self.config.model
        if target == "sign":
            return load(sign_file)
        else:
            return load(amplitude_file)

    def __build_options(self):
        sweep_size = (
            self.config.sweep_size
            if self.config.sweep_size is not None
            else self.number_spins
        )
        number_discarded = (
            self.config.number_discarded
            if self.config.number_discarded is not None
            else self.config.number_samples // 10
        )
        magnetisation = (
            self.config.magnetisation
            if self.config.magnetisation is not None
            else self.number_spins % 2
        )
        return _C._Options(
            number_spins=self.number_spins,
            magnetisation=magnetisation,
            number_chains=self.config.number_chains,
            number_samples=self.config.number_samples,
            sweep_size=sweep_size,
            number_discarded=number_discarded,
        )

    def __build_polynomial(self):
        return _C.Polynomial(self.hamiltonian, self.config.roots)

    def __load_exact(self):
        if self.config.exact is None:
            return None
        _, y = _core.with_file_like(self.config.exact, "rb", pickle.load)
        y = torch.from_numpy(y).squeeze()
        x = _C.all_spins(self.mc_options.number_spins, self.mc_options.magnetisation)
        dataset = _core.make_spin_dataloader(x, y, batch_size=2048)
        evaluator = create_supervised_evaluator(
            _core.combine_amplitude_and_sign(self.amplitude, self.sign),
            metrics={"overlap": OverlapMetric()},
        )

        def evaluate():
            evaluator.run(dataset)
            return evaluator.state.metrics["overlap"]

        return evaluate

    def monte_carlo(self):
        tqdm.write("Monte Carlo sampling from |ψ|²...", end="")
        start = time.time()
        (spins, values, acceptance) = sample_some(
            self.amplitude, self.mc_options, explicit=True
        )

        with torch.no_grad():
            state = _core.combine_amplitude_and_phase(
                self.amplitude, self.sign, apply_log=True, use_classifier=True
            )
            values = apply_polynomial(self.polynomial, spins, state)
            values = safe_real_exp(values)

        size = len(set((_C.unsafe_get(spins, i) for i in range(len(spins)))))
        stop = time.time()
        tqdm.write(
            " Done in {:.2f} seconds. ".format(stop - start)
            + "Acceptance {:.2f}%".format(100 * acceptance)
            + " Sampled {} spin configurations".format(size)
        )
        return (spins, values)

    def load_checkpoint(self, i: int):
        def load(target, model):
            pattern = os.path.join(self.config.output, str(i), target, "best_model_*")
            [filename] = glob.glob(pattern)
            model.load_state_dict(torch.load(filename))

        load("amplitude", self.amplitude)
        load("sign", self.sign)

    def step(self):
        dataset = self.monte_carlo()
        # ψ = self.__load_model_v2("amplitude")
        train(
            "amplitude",
            # ψ,
            self.amplitude,
            dataset,
            "{}/{}/amplitude".format(self.config.output, self._iteration),
            self.config.amplitude,
        )
        # self.amplitude.load_state_dict(ψ.state_dict())
        train(
            "sign",
            self.sign,
            dataset,
            "{}/{}/sign".format(self.config.output, self._iteration),
            self.config.sign,
        )
        if self.compute_overlap is not None:
            overlap = self.compute_overlap()
            tqdm.write("[{}] Overlap: {}".format(self._iteration, overlap))
        self._iteration += 1


Config = namedtuple(
    "Config",
    [
        "model",
        "output",
        "hamiltonian",
        "roots",
        "epochs",
        "number_samples",
        "number_chains",
        "amplitude",
        "sign",
        ## OPTIONAL
        "exact",
        "magnetisation",
        "sweep_size",
        "number_discarded",
    ],
    defaults=[None, None, None, None],
)

TrainingOptions = namedtuple(
    "TrainingOptions",
    [
        "train_batch_size",
        "max_epochs",
        "train_fraction",
        "patience",
        "optimiser",
        "val_batch_size",
    ],
    defaults=[
        16,
        200,
        0.90,
        10,
        "lambda m: torch.optim.RMSprop(m.parameters(), lr=1e-3, weight_decay=1e-4)",
        1024,
    ],
)


def main():
    config = Config(
        model=("example/1x10/amplitude_wip.py", "example/1x10/sign_wip.py"),
        hamiltonian="/vol/tcm01/westerhout_tom/nqs-playground/data/1x10/hamiltonian.txt",
        epochs=50,
        roots=[10.0, 10.0],
        number_samples=1000,
        number_chains=2,
        output="swo/run/3",
        exact="/vol/tcm01/westerhout_tom/nqs-playground/data/1x10/ground_state.pickle",
        amplitude=TrainingOptions(
            max_epochs=200,
            patience=20,
            train_fraction=1.0,
            train_batch_size=16,
            optimiser="lambda m: torch.optim.RMSprop(m.parameters(), lr=5e-4, weight_decay=2e-4)",
        ),
        sign=TrainingOptions(
            max_epochs=200,
            patience=20,
            train_fraction=1.0,
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
