from argparse import ArgumentParser
import collections
import os
import sys

import math
import numpy as np

# from torch import nn
# from torch.optim import SGD
# from torch.utils.data import DataLoader
import torch

from torch.utils.tensorboard import SummaryWriter

# import torch.nn.functional as F
# from torchvision.transforms import Compose, ToTensor, Normalize
# from torchvision.datasets import MNIST

import ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

import ignite.contrib.handlers.tqdm_logger

# from ignite.metrics import Accuracy, Loss

from tqdm import tqdm

import training
from training import make_spin_dataloader, import_network

import pickle

import cProfile

def get_data_loaders(config):
    r"""Constructs the dataloaders.

    First of all, we load the full pickled dataset from ``config.dataset``.
    Next, we sample ``config.train_fraction + config.val_fraction`` elements
    from it either uniformly or with probabilities proportional to squared
    amplitudes. ``config.replacement`` controls whether we sample with of
    without replacement. After that, we split the sampled elements into
    training and validation datasets.

    In total, four dataloaders are returned:

        * Training
        * Validation
        * Rest (everything except for training and validation parts)
        * All (simply the whole dataset)
    """
    dataset = training._with_file_like(config.dataset, "rb", pickle.load)
    with torch.no_grad():
        dataset = dataset[0], torch.from_numpy(dataset[1]).squeeze()
        assert config.sampling in {"uniform", "quadratic"}
        weights = None if config.sampling == "uniform" else dataset[1] ** 2
        train, rest = training.random_split(
            dataset,
            config.train_fraction + config.val_fraction,
            weights=weights,
            replacement=config.replacement,
        )
        train, val = training.random_split(
            train,
            config.train_fraction / (config.train_fraction + config.val_fraction),
            weights=None,
            replacement=False,
        )

    train_loader = make_spin_dataloader(*train, batch_size=config.train_batch_size)
    val_loader = make_spin_dataloader(*val, batch_size=config.val_batch_size)
    rest_loader = make_spin_dataloader(*rest, batch_size=config.val_batch_size)
    all_loader = make_spin_dataloader(*dataset, batch_size=config.val_batch_size)
    return collections.namedtuple("Loaders", ["training", "validation", "rest", "all"])(
        train_loader, val_loader, rest_loader, all_loader
    )


r"""Configuration options.
"""
Config = collections.namedtuple(
    "Config",
    [
        "dataset",  # Path of the pickled dataset to use for training
        "model",  # Path of the Python source file which defines the Net class
        "output",
        "train_fraction",  # Fraction of the Hilbert space basis to use for training
        "val_fraction",  # Fraction of the Hilbert space basis to use for validation
        "train_batch_size",  # Batch size of training
        "max_epochs",
        "patience",
        "optimiser",  #
        "target",  # Either 'sign' or 'amplitude'
        "sampling",
        "replacement",  # Whether to sampling with replacement
        "log_interval",  # Interval between logging training loss
        "val_batch_size",
    ],
    defaults=[
        50,  # patience
        "lambda m: torch.optim.RMSprop(m.parameters(), lr=1e-3, weight_decay=5e-4)",
        "sign",
        "uniform",  # sampling
        False,  # replacement
        10,
        1024,  # val_batch_size
    ],
)


def create_sign_trainer(model, optimizer, device, non_blocking):
    r"""Creates an engine for training the network on the sign structure.

    Cross entropy is used as loss function.
    """

    def prepare_batch(batch, device, non_blocking):
        x, y = batch
        if device:
            x = x.to(device=device, non_blocking=non_blocking)
            y = y.to(device=device, non_blocking=non_blocking)
        # Convert coefficients to classes
        y = torch.where(
            y >= 0, torch.tensor([0], device=device), torch.tensor([1], device=device)
        ).squeeze()
        return x, y

    return create_supervised_trainer(
        model,
        optimizer,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device=device,
        non_blocking=non_blocking,
        prepare_batch=prepare_batch,
    )


def overlap(predicted, expected):
    predicted = predicted.squeeze()
    expected = expected.squeeze()
    return (
        torch.abs(torch.dot(predicted, expected))
        / torch.norm(predicted)
        / torch.norm(expected)
    )


def create_amplitude_trainer(model, optimizer, device, non_blocking):
    r"""Creates an engine for training the network on wavefunction amplitudes.

    Cross entropy is used as loss function.
    """

    def prepare_batch(batch, device, non_blocking):
        x, y = batch
        if device:
            x = x.to(device=device, non_blocking=non_blocking)
            y = y.to(device=device, non_blocking=non_blocking)
        y = torch.abs(y).view([-1, 1])
        return x, y

    def loss_fn(predicted, expected):
        predicted = predicted.squeeze()
        expected = expected.squeeze()
        return (
            1
            - torch.dot(predicted, expected) ** 2
            / torch.norm(predicted) ** 2
            / torch.norm(expected) ** 2
        )

    # loss_fn = torch.nn.MSELoss()

    return create_supervised_trainer(
        model,
        optimizer,
        loss_fn=loss_fn,
        # torch.nn.MSELoss(),
        # lambda *args, **kwargs: -overlap(*args, **kwargs),
        device=device,
        non_blocking=non_blocking,
        prepare_batch=prepare_batch,
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
            self._sum += torch.dot(
                self._loss_fn(predicted, expected).squeeze(), weight
            ).item()
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
    def __init__(self, output_transform=lambda x: x):
        self._dot = None
        self._norm_predicted = None
        self._norm_expected = None
        self._called = None
        super().__init__(output_transform=output_transform)

    def reset(self):
        self._dot = 0
        self._norm_predicted = 0
        self._norm_expected = 0
        self._called = 0

    def update(self, output):
        predicted, expected = output
        predicted = predicted.squeeze()
        expected = expected.squeeze()
        assert torch.all(expected >= 0)
        self._dot += torch.dot(predicted, expected).item()
        self._norm_predicted += torch.norm(predicted).item() ** 2
        self._norm_expected += torch.norm(expected).item() ** 2
        self._called += 1

    def compute(self):
        # print(self._norm_predicted, self._norm_expected)
        if self._norm_predicted == 0 or self._norm_expected == 0:
            print(self._called, self._norm_predicted, self._norm_expected)
            raise ignite.exceptions.NotComputableError(
                "OverlapMetric must have at least one example before it can be computed."
            )
        return abs(self._dot) / math.sqrt(self._norm_predicted * self._norm_expected)


def create_sign_evaluator(model, device, non_blocking):
    def accuracy_output_transform(output):
        y_pred, y = output
        return (
            torch.max(y_pred, dim=1)[1],
            torch.where(
                y >= 0,
                torch.tensor([0], device=device),
                torch.tensor([1], device=device),
            ).squeeze(),
            (y ** 2).squeeze(),
        )

    def loss_output_transform(output):
        y_pred, y = output
        return (
            y_pred,
            torch.where(
                y >= 0,
                torch.tensor([0], device=device),
                torch.tensor([1], device=device),
            ).squeeze(),
            (y ** 2).squeeze(),
        )

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


def create_sign_evaluators(model, device, non_blocking):
    return collections.namedtuple("Evaluators", ["training", "validation", "testing"])(
        create_sign_evaluator(model, device=device, non_blocking=False),
        create_sign_evaluator(model, device=device, non_blocking=False),
        create_sign_evaluator(model, device=device, non_blocking=False),
    )


def create_amplitude_evaluator(model, device, non_blocking):
    def prepare_batch(batch, device, non_blocking):
        x, y = batch
        if device:
            x = x.to(device=device, non_blocking=non_blocking)
            y = y.to(device=device, non_blocking=non_blocking)
        y = torch.abs(y).view([-1, 1])
        return x, y

    return create_supervised_evaluator(
        model, metrics={"overlap": OverlapMetric()}, prepare_batch=prepare_batch
    )


def create_amplitude_evaluators(model, device, non_blocking):
    return collections.namedtuple("Evaluators", ["training", "validation", "testing"])(
        create_amplitude_evaluator(model, device=device, non_blocking=False),
        create_amplitude_evaluator(model, device=device, non_blocking=False),
        create_amplitude_evaluator(model, device=device, non_blocking=False),
    )


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.loaders = get_data_loaders(config)
        self.model = import_network(config.model)(
            next(iter(self.loaders.training))[0].size(1)
        )
        self.optimiser = eval(config.optimiser)(self.model)
        self.__add_engines()
        self.__add_loggers()
        self.__add_handlers()

    def __add_engines(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.config.target == "sign":
            self.trainer = create_sign_trainer(
                self.model, self.optimiser, device=device, non_blocking=False
            )
            self.evaluators = create_sign_evaluators(
                self.model, device=device, non_blocking=False
            )
        else:
            self.trainer = create_amplitude_trainer(
                self.model, self.optimiser, device=device, non_blocking=False
            )
            self.evaluators = create_amplitude_evaluators(
                self.model, device=device, non_blocking=False
            )

    def __add_loggers(self):
        self.tb_writer = SummaryWriter(log_dir=self.config.output)
        self.tqdm_writer = ignite.contrib.handlers.tqdm_logger.ProgressBar()

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
        add("validation", self.evaluators.validation, self.loaders.validation)
        add("rest", self.evaluators.testing, self.loaders.rest)
        add("all", self.evaluators.testing, self.loaders.all)

    def __add_checkpoint(self):
        if self.config.target == "sign":
            score_fn = lambda e: -e.state.metrics["cross_entropy"]
        else:
            score_fn = lambda e: e.state.metrics["overlap"]
        self.evaluators.validation.add_event_handler(
            Events.COMPLETED,
            ignite.handlers.ModelCheckpoint(
                dirname=self.config.output,
                filename_prefix="checkpoint_",
                score_function=score_fn,
            ),
            {"model": self.model},
        )

    def __add_early_stopping(self):
        if self.config.target == "sign":
            score_fn = lambda e: -e.state.metrics["cross_entropy"]
        else:
            score_fn = lambda e: e.state.metrics["overlap"]
        self.evaluators.validation.add_event_handler(
            Events.COMPLETED,
            ignite.handlers.EarlyStopping(
                patience=self.config.patience,
                score_function=score_fn,
                trainer=self.trainer,
            ),
        )

    def __call__(self):
        self.trainer.run(self.loaders.training, max_epochs=self.config.max_epochs)
        self.tb_writer.close()


def run(config):
    return Trainer(config)()


def main():
    # parser = ArgumentParser()
    # parser.add_argument(
    #     "config_file", type=argparse.FileType(mode="r"), help="path to JSON config file"
    # )
    # args = parser.parse_args()
    # config = json.load(args.config_file)

    target = "amplitude"
    if target == "sign":
        config = Config(
            dataset="dataset_1000.pickle",
            model="test_model.py",
            output="ignite/sign/1",
            target="sign",
            train_fraction=0.05,
            val_fraction=0.05,
            train_batch_size=16,
            patience=100,
            max_epochs=500,
            sampling="quadratic",
            replacement=True,
        )
    else:
        config = Config(
            dataset="dataset_1000.pickle",
            model="amplitude.py",
            output="ignite/amplitude/23",
            target="amplitude",
            optimiser="lambda m: torch.optim.RMSprop(m.parameters(), lr=1e-3, weight_decay=1e-4)",
            train_fraction=0.05,
            val_fraction=0.05,
            train_batch_size=32,
            patience=100,
            max_epochs=500,
            sampling="quadratic",
            replacement=True,
        )
    run(config)


if __name__ == "__main__":
    # cProfile.run("main()")
    main()
