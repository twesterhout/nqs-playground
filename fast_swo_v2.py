#!/usr/bin/env python3

import cmath
from copy import deepcopy
import enum
import itertools
import logging
import math
import glob
import os
import pwd  # Used by Scratch.getusername() function
import sys
import tempfile
import time
from typing import Dict, List, Tuple, Optional
import pickle

import numpy as np

import cProfile

# from mpi4py import MPI
import ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import ignite.contrib.handlers.tqdm_logger

import collections

from nqs_playground import _C
from nqs_playground.core import ExplicitState, load_explicit
from nqs_playground.hamiltonian import read_hamiltonian

import training as _core

from tqdm import tqdm

# from nqs_playground import mpi
import torch
from torch.utils.tensorboard import SummaryWriter


class _Train(object):
    """
    Temporary sets the module (i.e. a ``torch.nn.Module``) into training mode.

    We rely on the following: if ``m`` is a module, then

      * ``m.training`` returns whether ``m`` is currently in the training mode;
      * ``m.train()`` puts ``m`` into training mode;
      * ``m.eval()`` puts ``m`` into inference mode.

    This class is meant to be used in the ``with`` construct:

    .. code:: python

       with _Train(m):
           ...
    """

    def __init__(self, module):
        """
        :param module:
            a ``torch.nn.Module`` or an intance of another class with the same
            interface.
        """
        self._module = module
        # Only need to update the mode if not already training
        self._update = module.training == False

    def __enter__(self):
        if self._update:
            self._module.train()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._update:
            self._module.eval()


class _Eval(object):
    """
    Temporary sets the network (i.e. a ``torch.nn.Module``) into inference mode.

    We rely on the following: if ``m`` is a network, then

      * ``m.training`` returns whether ``m`` is currently in the training mode;
      * ``m.train()`` puts ``m`` into training mode;
      * ``m.eval()`` puts ``m`` into inference mode.

    This class is meant to be used in the ``with`` construct:

    .. code:: python

       with _Train(m):
           ...
    """

    def __init__(self, module):
        """
        :param module:
            a ``torch.nn.Module`` or an intance of another class with the same
            interface.
        """
        self._module = module
        # Only need to update the mode if currently training
        self._update = module.training == True

    def __enter__(self):
        if self._update:
            self._module.eval()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._update:
            self._module.train()


def _NumThreads(object):
    """
    Temporary changes the number of threads used by PyTorch.

    This is useful when we, for example, want to run multiple different neural
    networks in parallel rather than use multiple threads within a single
    network.
    """

    def __init__(self, num_threads):
        self._new = num_threads
        self._old = torch.get_num_threads()

    def __enter__(self):
        if self._new != self._old:
            torch.set_num_threads(self._new)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._new != self._old:
            torch.set_num_threads(self._old)


def sample_some(ψ: torch.jit.ScriptModule, options: _C._Options, explicit=False):
    if not explicit:
        with torch.no_grad():
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


def apply_polynomial(polynomial: _C.Polynomial, spins: np.ndarray, ψ):
    r = torch.empty(spins.shape[0], dtype=torch.float32)
    for i in range(spins.shape[0]):
        polynomial(1.0, _C.unsafe_get(spins, i))
        r[i] = torch.dot(ψ.forward(polynomial.vectors()), polynomial.coefficients())
    return r


class CombiningState(torch.nn.Module):
    def __init__(self, amplitude, sign):
        super().__init__()
        self.amplitude = amplitude
        self.sign = sign

    def forward(self, x):
        a = self.amplitude.forward(x).squeeze()
        b = (1 - 2 * torch.argmax(self.sign.forward(x), dim=1)).to(dtype=torch.float32)
        return a * b


# def sign_prepare_batch(batch, device, non_blocking):
#     x, y = batch
#     if device is not None:
#         x = x.to(device=device, non_blocking=non_blocking)
#         y = y.to(device=device, non_blocking=non_blocking)
#     y = torch.where(y >= 0, torch.tensor([0]), torch.tensor([1])).squeeze()
#     return x, y
#
#
# def amplitude_prepare_batch(batch, device, non_blocking):
#     x, y = batch
#     if device is not None:
#         x = x.to(device=device, non_blocking=non_blocking)
#         y = y.to(device=device, non_blocking=non_blocking)
#     y = torch.abs(y).squeeze()
#     return x, y


def create_sign_trainer(model, optimiser):
    r"""Creates an engine for training the network on the sign structure.

    Cross entropy is used as loss function.
    """
    return create_supervised_trainer(
        model, optimiser, loss_fn=torch.nn.CrossEntropyLoss()
    )


def create_sign_evaluator(model):
    return create_supervised_evaluator(
        model,
        metrics={
            "cross_entropy": ignite.metrics.Loss(torch.nn.CrossEntropyLoss()),
            "accuracy": ignite.metrics.Accuracy(
                lambda output: (torch.argmax(output[0], dim=1), output[1])
            ),
        },
    )


def create_amplitude_trainer(model, optimiser):
    r"""Creates an engine for training the network on wavefunction amplitudes."""

    def loss_fn(predicted, expected):
        predicted = predicted.squeeze()
        return (
            1
            - torch.dot(predicted, expected) ** 2
            / torch.norm(predicted) ** 2
            / torch.norm(expected) ** 2
        )

    return create_supervised_trainer(model, optimiser, loss_fn=loss_fn)


def create_trainer(target, model, optimiser):
    return {"sign": create_sign_trainer, "amplitude": create_amplitude_trainer}[target](
        model, optimiser
    )


def create_evaluators(target, model):
    make = {"sign": create_sign_evaluator, "amplitude": create_amplitude_evaluator}[
        target
    ]
    return collections.namedtuple("Evaluators", ["training", "validation"])(
        make(model), make(model)
    )


class OverlapMetric(ignite.metrics.metric.Metric):
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
        assert torch.all(expected >= 0)
        self._dot += torch.dot(predicted, expected).item()
        self._norm_predicted += torch.norm(predicted).item() ** 2
        self._norm_expected += torch.norm(expected).item() ** 2

    def compute(self):
        if self._norm_expected == 0:
            raise ignite.exceptions.NotComputableError(
                "OverlapMetric must have at least one example before it can be computed."
            )
        return abs(self._dot) / math.sqrt(self._norm_predicted * self._norm_expected)


def create_amplitude_evaluator(model):
    return create_supervised_evaluator(model, metrics={"overlap": OverlapMetric()})


class Trainer(object):
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

    def __add_engines(self):
        self.trainer = create_trainer(self.target, self.model, self.optimiser)
        self.evaluators = create_evaluators(self.target, self.model)

    def __add_loggers(self):
        self.tb_writer = SummaryWriter(log_dir=self.output, max_queue=1000)
        self.tqdm_writer = ignite.contrib.handlers.tqdm_logger.ProgressBar()

    def __add_loaders(self, dataset):
        with torch.no_grad():
            x, y = dataset
            if self.target == "sign":
                y = torch.where(y >= 0, torch.tensor([0]), torch.tensor([1])).squeeze()
            else:
                y = torch.abs(y).squeeze()
            dataset = (x, y)

            if self.config.train_fraction == 1.0:
                self.loaders = collections.namedtuple(
                    "Loaders", ["training", "validation"]
                )(
                    _core.make_spin_dataloader(
                        *dataset, batch_size=self.config.train_batch_size
                    ),
                    None,
                )
            else:
                train, val = _core.random_split(
                    dataset, self.config.train_fraction, weights=None, replacement=False
                )
                self.loaders = collections.namedtuple(
                    "Loaders", ["training", "validation"]
                )(
                    _core.make_spin_dataloader(
                        *train, batch_size=self.config.train_batch_size
                    ),
                    _core.make_spin_dataloader(
                        *val, batch_size=self.config.val_batch_size
                    ),
                )

    def __add_handlers(self):
        self.tqdm_writer.attach(
            self.trainer,
            event_name=Events.EPOCH_COMPLETED,
            closing_event_name=Events.COMPLETED,
        )

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


class Runner(object):
    def __init__(self, config):
        self.config = config
        self.__load_hamiltonian()
        self.__load_model()
        self.mc_options = self.__build_options()
        self.polynomial = self.__build_polynomial()
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

    def monte_carlo(self):
        tqdm.write("Monte Carlo sampling from |ψ|²...")
        start = time.time()
        (spins, values, acceptance) = sample_some(
            self.amplitude, self.mc_options, explicit=True
        )
        state = torch.jit.script(CombiningState(self.amplitude, self.sign))
        values = apply_polynomial(self.polynomial, spins, state)
        # values *= (
        #     (1 - 2 * torch.argmax(self.sign.forward(_C.unpack(spins)), dim=1))
        #     .to(torch.float32)
        #     .squeeze()
        # )
        stop = time.time()
        tqdm.write(
            "Done in {:.2f} seconds. ".format(stop - start)
            + "Acceptance {:.2f}%".format(100 * acceptance)
        )
        return (spins, values)

    def step(self):
        dataset = self.monte_carlo()
        Trainer(
            "amplitude",
            self.amplitude,
            dataset,
            "{}/{}/amplitude".format(self.config.output, self._iteration),
            self.config.amplitude,
        ).run()
        Trainer(
            "sign",
            self.sign,
            dataset,
            "{}/{}/sign".format(self.config.output, self._iteration),
            self.config.sign,
        ).run()
        self._iteration += 1


Config = collections.namedtuple(
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
        "magnetisation",
        "sweep_size",
        "number_discarded",
    ],
    defaults=[None, None, None],
)

TrainingOptions = collections.namedtuple(
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
        0.95,
        10,
        "lambda m: torch.optim.RMSprop(m.parameters(), lr=1e-3, weight_decay=1e-4)",
        1024,
    ],
)


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ones(x.size(0))


def foo():
    m = torch.jit.script(Dummy())
    m.save("dummy.pt")

    r = _C._sample_some(
        "dummy.pt",
        _C._Options(
            number_spins=10,
            magnetisation=0,
            number_chains=4,
            number_samples=100000,
            sweep_size=1,
            number_discarded=0,
        ),
    )

    all_spins = _C.all_spins(10, 0)
    ids = [(_C.unsafe_get(all_spins, i), i) for i in range(all_spins.shape[0])]
    # for (k, v) in ids[:20]:
    #     print(k, v)
    ids = dict(ids)

    # for i in range(min(r[0].shape[0], 10)):
    #     print(_C.unsafe_get(r[0], i))
    spins = [ids[_C.unsafe_get(r[0], i)] for i in range(r[0].shape[0])]
    hist = np.histogram(spins, bins=252, range=(0, 252))
    print(hist)
    print(hist[0].mean(), hist[0].std())


# def load_exact(filename):
#     phi = _core.with_file_like(filename, "rb", lambda f: load_explicit(f)[0])
#     number_spins = len(next(iter(phi)))
#     phi = ExplicitState(phi)
#     magnetisation = number_spins % 2
#     spins = _C.unpack(_C.all_spins(number_spins, magnetisation))
#     values = phi(spins)
# 
#     with open("1x10.exact", "wb") as output:
#         pickle.dump((_C.all_spins(number_spins, magnetisation), values), output)
#     return spins, values

def load_exact(filename):
    (spins, values) = _core.with_file_like(filename, "rb", pickle.load)
    return torch.from_numpy(spins), values


def analyse(dirname, exact):

    spins, exact_coefficients = load_exact(exact)
    number_spins = spins.size(1)

    def load(name):
        return torch.jit.script(_core.import_network(name)(number_spins))

    def overlap(predicted, expected):
        predicted = predicted.squeeze().detach().numpy()
        # expected = expected.view(-1).detach().numpy().view(np.complex64)
        return (
            abs(np.dot(predicted, expected))
            / np.linalg.norm(predicted)
            / np.linalg.norm(expected)
        )

    psi = CombiningState(load("test_amplitude.py"), load("test_phase.py"))
    dirnames = sorted(
        filter(lambda p: os.path.isdir(os.path.join(dirname, p)), os.listdir(dirname)),
        key=lambda x: int(x),
    )
    for d in dirnames:
        [amplitude] = glob.glob(os.path.join(dirname, d, "amplitude/best_model_*"))
        [sign] = glob.glob(os.path.join(dirname, d, "sign/best_model_*"))
        psi.amplitude.load_state_dict(torch.load(amplitude))
        psi.sign.load_state_dict(torch.load(sign))
        coefficients = psi.forward(spins)
        print(d, overlap(coefficients, exact_coefficients))


def main():
    config = Config(
        model=("test_amplitude.py", "test_phase.py"),
        hamiltonian="/vol/tcm01/westerhout_tom/nqs-playground/data/1x10.hamiltonian",
        epochs=50,
        roots=[(10.0, None), (10.0, None)],
        number_samples=1000,
        number_chains=2,
        output="swo/run/2",
        amplitude=TrainingOptions(
            max_epochs=200,
            patience=100,
            train_fraction=1.0,
            optimiser="lambda m: torch.optim.RMSprop(m.parameters(), lr=5e-3, weight_decay=1e-5)",
        ),
        sign=TrainingOptions(
            max_epochs=200,
            patience=100,
            train_fraction=1.0,
            optimiser="lambda m: torch.optim.RMSprop(m.parameters(), lr=5e-3, weight_decay=1e-5)",
        ),
    )

    runner = Runner(config)

    for i in range(config.epochs):
        runner.step()

    # analyse("swo/run/2", "workdir/cxx/1x10.out")
    analyse("swo/run/2", "../nqs_frustrated_phase/data/chain/10/exact/dataset_1000.pickle")
    return


if __name__ == "__main__":
    main()
