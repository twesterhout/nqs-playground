import argparse
import collections
import json
import math
import os
import pickle
import shutil
import sys
from typing import Tuple

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import ignite
import ignite.contrib.handlers.tqdm_logger
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from tqdm import tqdm

from . import _C
from . import core
from .core import make_spin_dataloader, import_network


def eliminate_phase(zs):
    r"""``zs`` are wavefunction coefficients. Even though we know that the
    ground state is real, after diagonalisation ``zs`` might have some complex
    phase. This function removes this common phase factor.
    """
    # We do everything on the CPU here, because it's fast anyway and zs could
    # be very big.
    if zs.dtype == np.float32 or zs.dtype == np.float64:
        return torch.from_numpy(zs).to(torch.float32)
    assert len(zs) > 0
    phase = np.log(zs[0]).imag
    zs *= np.exp(-1j * phase)
    assert np.all(np.isclose(zs.imag, 0))
    return torch.from_numpy(zs.real).to(torch.float32).contiguous()


r"""In general, for supervised learning we have 4 different data loaders:
  * ``training``: our training dataset. Samples from this dataset are used to
    calculate gradients and update model's weights.
  * ``validation``: validation dataset. Samples from this dataset are used to
    evaluate convergence and employ early stopping regularisation method.
  * ``rest``: test dataset. Usually everything not included into training and
    validation datasets.
  * ``all``: really everything. Can be used to estimate the overall performance
    of the model on the full Hilbert space.
"""
Loaders = collections.namedtuple("Loaders", ["training", "validation", "rest", "all"])


def subtract(xs, ys):
    """Given two datasets, computes the set difference between them. Elements
    are compared by their first column.
    """
    if not all((isinstance(x, np.ndarray) or isinstance(x, torch.Tensor) for x in xs)):
        raise ValueError("xs should be a tuple of NumPy arrays or Torch tensors")
    if not all((isinstance(y, np.ndarray) or isinstance(y, torch.Tensor) for y in ys)):
        raise ValueError("ys should be a tuple of NumPy arrays or Torch tensors")
    n = xs[0].shape[0]
    if not all((x.shape[0] == n for x in xs)):
        raise ValueError(
            "all elements of xs should have the same size along the first dimension"
        )
    exclude = set(ys[0])
    keys = xs[0]
    count = 0
    for j in range(n):
        if keys[j] not in exclude:
            if count != j:
                for t in xs:
                    t[count] = t[j]
            count += 1
    return tuple(t[:count] for t in xs)


def make_data_loaders(config):
    r"""Constructs the dataloaders.

    We sample ``config.train_fraction + config.val_fraction`` elements from the
    dataset either uniformly or with probabilities proportional to squared
    amplitudes. ``config.replacement`` controls whether we sample with of
    without replacement. After that, we split the sampled elements into
    training and validation datasets.

    In total, four dataloaders are returned:

        * Training
        * Validation
        * Rest (everything except for training and validation parts)
        * All (simply the whole dataset)

    If ``train_fraction + val_fraction == 1``, then rest and all loaders are
    None.
    """
    assert config.sampling in {"uniform", "quadratic"}
    x = np.load(config.x) if isinstance(config.x, str) else config.x
    y = np.load(config.y) if isinstance(config.y, str) else config.y
    y = eliminate_phase(y)
    dataset = (x, y)
    weights = None if config.sampling == "uniform" else y ** 2
    with torch.no_grad():
        train, rest = core.random_split(
            dataset,
            config.train_fraction + config.val_fraction,
            weights=weights,
            replacement=config.replacement,
        )
        train, val = core.random_split(
            train,
            config.train_fraction / (config.train_fraction + config.val_fraction),
            weights=None,
            replacement=False,
        )
        val = subtract(val, train)

        unpack = lambda batch, idx: _C.v2.unpack(batch, idx, config.number_spins)
        train_loader = make_spin_dataloader(
            *train, batch_size=config.train_batch_size, drop_last=True, unpack=unpack
        )
        unpack = lambda batch: _C.v2.unpack(batch, config.number_spins)
        val_loader = make_spin_dataloader(
            *val,
            batch_size=config.val_batch_size,
            drop_last=False,
            unpack=unpack,
            shuffle=False
        )
        if len(rest[0]) > 0:
            rest_loader, all_loader = [
                make_spin_dataloader(
                    *t,
                    batch_size=config.val_batch_size,
                    drop_last=False,
                    unpack=unpack,
                    shuffle=False
                )
                for t in (rest, dataset)
            ]
        else:
            rest_loader, all_loader = None, None
    return Loaders(train_loader, val_loader, rest_loader, all_loader)


def load_model(config) -> torch.nn.Module:
    if isinstance(config.model, str):
        m = import_network(config.model)
    elif isinstance(config.model, torch.nn.Module):
        m = config.model
    else:
        raise TypeError(
            "invalid config.model: {}; expected either a filename "
            "(i.e. a 'str') or PyTorch module (i.e. a 'torch.nn.Module')"
            "".format(config.model)
        )
    if not isinstance(m, torch.nn.Module):
        # If not already a torch.nn.Module, it is probably a function,
        # which returns a torch.nn.Module when given the number of spins in
        # the system
        m = m(config.number_spins)
    if config.use_jit and not isinstance(m, torch.jit.ScriptModule):
        m = torch.jit.script(m)
    return m


def make_optimiser(config, model) -> torch.optim.Optimizer:
    if isinstance(config.optimiser, str):
        return eval(config.optimiser)(model)
    return config.optimiser(model)


r"""Configuration options.
"""
Config = collections.namedtuple(
    "Config",
    [
        # Path to NumPy binary file with packed input spin configurations.
        "x",
        # Path to NumPy binary file with coefficients of the wavefunction. It
        # can be either an array of real or complex valued coefficients.
        "y",
        # Number of spins in the system. This is needed to correctly unpack
        # `x`. Must be a positive integer smaller than 64
        "number_spins",
        # Path of the Python source file which defines the Net class
        "model",
        # Directory where to save the results
        "output",
        # Fraction of `x` to use for training
        "train_fraction",
        # Fraction of `y` to use for validation
        "val_fraction",
        # Batch size of training
        "train_batch_size",
        "max_epochs",
        "patience",
        "optimiser",  #
        # Either 'sign' or 'amplitude'
        "target",
        # How to sample. Must be either 'uniform' or 'quadratic'. Uniform means
        # that spin configurations are chosen randomly from `x`. Quadratic
        # implies that spin configurations are samples according to the
        # probability distribution proportional to `|y|²`.
        "sampling",
        # Whether to sampling with replacement
        "replacement",
        # Interval between logging training loss
        "log_interval",
        # Batch size for validation and test datasets
        "val_batch_size",
        # Use jit
        "use_jit",
    ],
    defaults=[
        50,  # patience
        "lambda m: torch.optim.RMSprop(m.parameters(), lr=1e-3, weight_decay=5e-4)",
        "sign",  # target
        "uniform",  # sampling
        True,  # replacement
        10,  # log_interval
        8192,  # val_batch_size
        True,  # use_jit
    ],
)


def create_sign_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    non_blocking: bool,
) -> ignite.engine.Engine:
    r"""Creates an engine for training the network on the sign structure.

    Cross entropy is used as loss function.
    """

    def prepare_batch(batch, device: str, non_blocking: bool):
        x, y = batch
        if device:
            x = x.to(device=device, non_blocking=non_blocking)
            y = y.to(device=device, non_blocking=non_blocking)
        # Convert coefficients to classes
        return x, (y < 0).to(torch.long)

    return create_supervised_trainer(
        model,
        optimizer,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=device,
        non_blocking=non_blocking,
        prepare_batch=prepare_batch,
    )


def create_amplitude_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    non_blocking: bool,
) -> ignite.engine.Engine:
    r"""Creates an engine for training the network on wavefunction amplitudes.

    Cross entropy is used as loss function.
    """

    def prepare_batch(batch, device, non_blocking):
        x, y = batch
        if device:
            x = x.to(device=device, non_blocking=non_blocking)
            y = y.to(device=device, non_blocking=non_blocking)
        # Convert coefficients to amplitudes
        return x, torch.abs(y)

    # @torch.jit.script
    # def loss_fn(predicted, expected):
    #     predicted = predicted.squeeze()
    #     expected = expected.squeeze()
    #     return (
    #         1
    #         - torch.dot(predicted, expected) ** 2
    #         / torch.norm(predicted) ** 2
    #         / torch.norm(expected) ** 2
    #     )
    @torch.jit.script
    def loss_fn(predicted, expected):
        predicted = predicted.squeeze()
        expected = expected.squeeze()
        quotient = predicted / expected
        return 1 - torch.sum(quotient) ** 2 / torch.norm(quotient) ** 2

    return create_supervised_trainer(
        model,
        optimizer,
        loss_fn=loss_fn,
        device=device,
        non_blocking=non_blocking,
        prepare_batch=prepare_batch,
    )


def make_trainer(target: str, *args, **kwargs) -> ignite.engine.Engine:
    return {"amplitude": create_amplitude_trainer, "sign": create_sign_trainer}[target](
        *args, **kwargs
    )


class Loss(ignite.metrics.metric.Metric):
    r"""Custom metric to simplify computation of weighted losses and
    accuracies.
    """

    def __init__(self, loss_fn, output_transform=lambda x: x, weighted=False):
        self._loss_fn = loss_fn
        self._weighted = weighted
        self._sum = None
        self._norm = None
        super().__init__(output_transform=output_transform)

    def reset(self):
        self._sum = 0
        self._norm = 0

    def update(self, output):
        if len(output) != 3:
            raise ValueError(
                "{}.update expects `output` ".format(self.__class__.__name__)
                + "to be a tuple `(predicted, expected, weight)`"
            )
        predicted, expected, weight = output
        if self._weighted:
            self._sum += torch.dot(self._loss_fn(predicted, expected), weight).item()
            self._norm += torch.sum(weight).item()
        else:
            self._sum += torch.sum(self._loss_fn(predicted, expected), dim=0).item()
            self._norm += expected.shape[0]

    def compute(self):
        if self._norm == 0:
            raise ignite.exceptions.NotComputableError(
                "Loss must have at least one example before it can be computed."
            )
        return self._sum / self._norm


class OverlapMetric(ignite.metrics.metric.Metric):
    def __init__(self, output_transform=lambda x: x, weighted=True):
        self._weighted = weighted
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
        if self._weighted:
            self._dot += torch.dot(predicted, expected).item()
            self._norm_predicted += torch.norm(predicted).item() ** 2
            self._norm_expected += torch.norm(expected).item() ** 2
        else:
            quotient = predicted / expected
            self._dot += torch.sum(qoutient).item()
            self._norm_predicted += torch.norm(quotient).item() ** 2
            self._norm_expected += expected.size(0)

    def compute(self):
        if self._norm_predicted == 0 or self._norm_expected == 0:
            print(self._called, self._norm_predicted, self._norm_expected)
            raise ignite.exceptions.NotComputableError(
                "OverlapMetric must have at least one example before it can be computed."
            )
        return abs(self._dot) / math.sqrt(self._norm_predicted * self._norm_expected)


def create_sign_evaluator(model, device, non_blocking):
    def accuracy_output_transform(output):
        y_pred, y = output
        return torch.argmax(y_pred, dim=1), y < 0, y ** 2

    def loss_output_transform(output):
        y_pred, y = output
        return y_pred, (y < 0).to(torch.long), y ** 2

    mk_loss_metric = lambda weighted: Loss(
        torch.nn.CrossEntropyLoss(reduction="none"),
        output_transform=loss_output_transform,
        weighted=weighted,
    )
    mk_accuracy_metric = lambda weighted: Loss(
        lambda y_pred, y: (y_pred == y).to(torch.float32),
        output_transform=accuracy_output_transform,
        weighted=weighted,
    )
    return create_supervised_evaluator(
        model,
        metrics={
            "cross_entropy": mk_loss_metric(False),
            "weighted_cross_entropy": mk_loss_metric(True),
            "accuracy": mk_accuracy_metric(False),
            "weighted_accuracy": mk_accuracy_metric(True),
        },
    )


def create_amplitude_evaluator(model, device, non_blocking, weighted=True):
    def prepare_batch(batch, device, non_blocking):
        x, y = batch
        if device:
            x = x.to(device=device, non_blocking=non_blocking)
            y = y.to(device=device, non_blocking=non_blocking)
        y = torch.abs(y).view([-1, 1])
        return x, y

    return create_supervised_evaluator(
        model, metrics={"overlap": OverlapMetric(weighted)}, prepare_batch=prepare_batch
    )


Evaluators = collections.namedtuple("Evaluators", ["training", "validation", "testing"])


def make_evaluators(target, *args, **kwargs):
    if target == "sign":
        make = lambda: create_sign_evaluator(*args, **kwargs)
        return Evaluators(make(), make(), make())
    else:
        assert target == "amplitude"
        make = lambda w: create_amplitude_evaluator(*args, **kwargs, weighted=w)
        return Evaluators(make(False), make(False), make(True))


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = self.config.device
        self.loaders = make_data_loaders(config)
        self.model = load_model(config).to(device=self.device)
        self.optimiser = make_optimiser(config, self.model)
        self.__add_engines()
        self.__add_loggers()
        self.__add_handlers()

    def __add_engines(self):
        kwargs = {
            "target": self.config.target,
            "model": self.model,
            "device": self.device,
            "non_blocking": False,
        }
        self.trainer = make_trainer(**kwargs, optimizer=self.optimiser)
        self.evaluators = make_evaluators(**kwargs)

    def __add_loggers(self):
        self.tb_writer = SummaryWriter(log_dir=self.config.output)
        self.tqdm_writer = ignite.contrib.handlers.tqdm_logger.ProgressBar()

    def __add_handlers(self):
        self.tqdm_writer.attach(
            self.trainer,
            event_name=Events.EPOCH_COMPLETED,
            closing_event_name=Events.COMPLETED,
        )

        # @self.trainer.on(Events.ITERATION_COMPLETED)
        # def log_loss(engine):
        #     self.tb_writer.add_scalar(
        #         "training/loss", engine.state.output, engine.state.iteration
        #     )

        self.__add_evaluators()
        self.__add_checkpoint()
        self.__add_early_stopping()

    def __add_evaluators(self):
        def add(tag, evaluator, dataloader, interval=1):
            def log_results(engine):
                n = engine.state.epoch
                if n % interval == 0:
                    evaluator.run(dataloader)
                    for name, value in evaluator.state.metrics.items():
                        self.tb_writer.add_scalar("{}/{}".format(tag, name), value, n)

            self.trainer.add_event_handler(Events.STARTED, log_results)
            self.trainer.add_event_handler(Events.EPOCH_COMPLETED, log_results)

        add("training", self.evaluators.training, self.loaders.training)
        add("validation", self.evaluators.validation, self.loaders.validation)
        if self.loaders.rest is not None:
            add("rest", self.evaluators.testing, self.loaders.rest, 3)
        if self.loaders.all is not None:
            add("all", self.evaluators.testing, self.loaders.all, 3)

    def __score_fn(self):
        if self.config.target == "sign":
            return lambda e: -e.state.metrics["cross_entropy"]
        return lambda e: e.state.metrics["overlap"]

    def __add_checkpoint(self):
        self.evaluators.validation.add_event_handler(
            Events.COMPLETED,
            ignite.handlers.ModelCheckpoint(
                dirname=self.config.output,
                filename_prefix="best",
                score_function=self.__score_fn(),
                save_as_state_dict=True,
            ),
            {"model": self.model},
        )

    def __add_early_stopping(self):
        self.evaluators.validation.add_event_handler(
            Events.COMPLETED,
            ignite.handlers.EarlyStopping(
                patience=self.config.patience,
                score_function=self.__score_fn(),
                trainer=self.trainer,
            ),
        )

    def load_best(self):
        matched = [
            f for f in os.listdir(self.config.output) if f.startswith("best_model_")
        ]
        if len(matched) == 0:
            raise ValueError(
                "Could not load the checkpoint. There are no files "
                "prefixed with {!r} in the directory {!r}."
                "".format(self.prefix, self.config.output)
            )
        if len(matched) > 1:
            raise ValueError(
                "Could not load the checkpoint. There are more than "
                "one files prefixed with {!r} in the directory {!r}."
                "".format(self.prefix, self.config.output)
            )
        matched = next(iter(matched))
        self.model.load_state_dict(
            torch.load(os.path.join(self.config.output, matched))
        )

    def __call__(self):
        with torch.jit.optimized_execution(True):
            self.trainer.run(self.loaders.training, max_epochs=self.config.max_epochs)
        self.tb_writer.close()
        self.load_best()


def run(config):
    return Trainer(config)()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=argparse.FileType(mode="r"), help="path to JSON config file"
    )
    args = parser.parse_args()
    config = json.load(args.config_file)
    config = Config(**config)
    if not os.path.exists(config.output):
        os.makedirs(config.output)
    shutil.copy2(args.config_file.name, config.output)
    shutil.copy2(config.model, config.output)
    run(config)

    # target = "amplitude"
    # if target == "sign":
    #     config = Config(
    #         dataset="dataset_1000.pickle",
    #         model="test_model.py",
    #         output="ignite/sign/1",
    #         target="sign",
    #         train_fraction=0.05,
    #         val_fraction=0.05,
    #         train_batch_size=16,
    #         patience=100,
    #         max_epochs=500,
    #         sampling="quadratic",
    #         replacement=True,
    #     )
    # else:
    #     config = Config(
    #         dataset="dataset_1000.pickle",
    #         model="amplitude.py",
    #         output="ignite/amplitude/23",
    #         target="amplitude",
    #         optimiser="lambda m: torch.optim.RMSprop(m.parameters(), lr=1e-3, weight_decay=1e-4)",
    #         train_fraction=0.05,
    #         val_fraction=0.05,
    #         train_batch_size=32,
    #         patience=100,
    #         max_epochs=500,
    #         sampling="quadratic",
    #         replacement=True,
    #     )
    # run(config)


if __name__ == "__main__":
    # cProfile.run("main()")
    main()
