#!/usr/bin/env python3

from collections import namedtuple
from enum import Enum
import glob
import math
from math import sqrt
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np

import ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import ignite.contrib.handlers.tqdm_logger

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

try:
    from typing_extensions import Final
except ImportError:
    # If you don't have `typing_extensions` installed, you can use a
    # polyfill from `torch.jit`.
    from torch.jit import Final

from tqdm import tqdm

from nqs_playground import *
import nqs_playground._C as _C
from nqs_playground.core import SpinDataset, forward_with_batches
from nqs_playground.sr import load_exact, load_optimiser

# torch.set_num_interop_threads(1)
# np.random.seed(123)
# torch.manual_seed(432)

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
        "output",
    ],
    # Some sane defaults
    defaults=[
        256,  # train_batch_size
        3,  # max_epochs
        0.80,  # train_fraction
        10,  # patience
        None,
        # lambda m: torch.optim.RMSprop(
        #     m.parameters(), lr=1e-3, weight_decay=1e-6
        # ),  # optimiser
        4096,  # val_batch_size
        None,
    ],
)


class Target(Enum):
    AMPLITUDE = 1
    SIGN = 2


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


# @torch.jit.script
def overlap_loss_fn(
    predicted: Tensor, expected: Tensor, weights: Tensor  # ] = None
) -> Tensor:
    r"""
        ∑ψφ = 10
    """
    if not torch.all(predicted >= 0):
        print("predicted: ", predicted)
    if not torch.all(expected >= 0):
        print("expected: ", expected)
    assert torch.all(predicted >= 0) and torch.all(expected >= 0)
    predicted = predicted.squeeze()
    expected = expected.squeeze()
    # if weights is None:
    #     quotient = expected / predicted
    #     return 1 - torch.sum(quotient) ** 2 / torch.norm(
    #         quotient
    #     ) ** 2 / predicted.size(0)
    # else:
    weights = weights.squeeze()
    dot = torch.sum(weights * predicted * expected)
    norm_predicted = torch.dot(weights, predicted ** 2)
    norm_expected = torch.dot(weights, expected ** 2)
    return 1 - dot ** 2 / norm_predicted / norm_expected


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, targets, weights=None):
        r = super().forward(inputs, targets)
        # print("loss: {}, {}, {} -> {}".format(inputs[:10], targets[:10], weights, r))
        # print("loss: {}, {}, {} -> {}".format(inputs.size(), targets.size(), weights, r))
        return r


@torch.jit.script
def _overlap_metric_kernel(
    predicted: Tensor, expected: Tensor, weights: Optional[Tensor] = None
) -> Tuple[float, float, float]:
    predicted = predicted.squeeze()
    expected = expected.squeeze()
    if weights is None:
        weights = torch.ones_like(predicted)
    else:
        weights = weights.squeeze()
    dot = torch.sum(weights * predicted * expected).item()
    norm_predicted = torch.dot(weights, predicted ** 2).item()
    norm_expected = torch.dot(weights, expected ** 2).item()
    return dot, norm_predicted, norm_expected


class _OverlapMetric:
    r"""A metric for computing overlap with the target state."""

    def __init__(self, output_transform=lambda x: x):
        self.output_transform = output_transform
        self._dot = None
        self._norm_predicted = None
        self._norm_expected = None

    def reset(self):
        self._dot = 0
        self._norm_predicted = 0
        self._norm_expected = 0

    def update(self, output):
        output = self.output_transform(output)
        dot, predicted, expected = _overlap_metric_kernel(*output)
        self._dot += dot
        self._norm_predicted += predicted
        self._norm_expected += expected

    def compute(self):
        return abs(self._dot) / math.sqrt(self._norm_predicted * self._norm_expected)


class _FidelityMetric(_OverlapMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        return 1 - super().compute() ** 2


class _AccuracyMetric:
    r"""A metric for computing the accuracy."""

    def __init__(self, output_transform=lambda x: x):
        self.output_transform = output_transform
        self._good = None
        self._total = None

    def reset(self):
        self._good = 0
        self._total = 0

    def update(self, output):
        predicted, expected, *weights = self.output_transform(output)
        if len(weights) > 0:
            self._good += torch.dot(
                weights[0], (predicted == expected).to(weights[0].dtype)
            ).item()
            self._total += torch.sum(weights[0])
        else:
            self._good += torch.sum(predicted == expected).item()
            self._total += predicted.size(0)

    def compute(self):
        return self._good / self._total


class _LossMetric:
    r"""An Ignite metric for computing overlap of with the target state.
    """

    def __init__(self, loss_fn, output_transform=lambda x: x):
        self.output_transform = output_transform
        self.loss_fn = loss_fn
        self._sum = None
        self._count = None

    def reset(self):
        self._sum = 0
        self._count = 0

    def update(self, output):
        n = output[0].size(0)
        self._sum += self.loss_fn(*self.output_transform(output)) * n
        self._count += n

    def compute(self):
        return self._sum / self._count


class TensorIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, *tensors, batch_size=1, shuffle=False):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        assert all(tensors[0].device == tensor.device for tensor in tensors)
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return self.tensors[0].size(0)

    @property
    def device(self):
        return self.tensors[0].device

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self), device=self.device)
            tensors = tuple(tensor[indices] for tensor in self.tensors)
        else:
            tensors = self.tensors
        return zip(*(torch.split(tensor, self.batch_size) for tensor in tensors))


class ManualTrainer:
    def __init__(self, target, model, train_dataset, config, test_dataset=None):
        self.target = target
        self.config = config
        self.train_dataset = self._prepare_dataset(train_dataset)
        self.test_dataset = self._prepare_dataset(test_dataset, test=True)
        self.model = self._prepare_model(model)
        self.debug = True

        self._tb_writer = SummaryWriter(log_dir=self.config.output)
        self._loss_fn = self._get_loss_function()
        self._optimiser = load_optimiser(self.config.optimiser, self.model.parameters())
        self._train_loader = None
        self._eval_loader = None
        self._test_loader = None
        self._eval_metrics = self._get_eval_metrics()
        self._test_metrics = self._get_test_metrics()

    def _prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        r"""Prepares the model for training. If :attr:`target` is
        ``Target.SIGN``, then there is nothing to be done. Otherwise, we
        transform the model from one predicting log amplitudes to the one
        predicting amplitudes.

        .. note:: This function is not as pure as it might look: it (possibly)
                  modifies :attr:`train_dataset` by adding weights to it.
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "model must be a torch.nn.Module; received {} instead"
                "".format(type(model))
            )
        if self.target is Target.SIGN:
            return model
        assert self.target is Target.AMPLITUDE
        # Since we're optimising overlap and model predicts log amplitudes,
        # we first transform it to predict actual amplitudes. Effectively
        # we just want to add a new layer taking the exponent, but this can
        # overflow. So instead we first determine the maximal predicted
        # value in the dataset and then rescale the output to prevent
        # overflows.
        class _ExpModel(torch.nn.Module):
            scale: Final[float]

            def __init__(self, module, scale):
                super().__init__()
                self.module = module
                self.scale = scale

            def forward(self, x):
                return torch.exp(self.module(x) - self.scale)

        with torch.no_grad():
            model.eval()
            # We assume that dataset is a tuple of tensors where the first
            # tensor contains inputs. Forward propagating the full training
            # dataset is a necessary evil here...
            xs = self.train_dataset[0]
            ys = forward_with_batches(model, xs, batch_size=self.config.val_batch_size)
            scale = torch.max(ys).item()
            # Since we're propagated all the data anyway, we might as well
            # compute something useful: sample weights (=1/p² where p is the
            # probability). These weights will later on be used to estimate
            # overlap.
            ys -= scale
            ys *= -2.0
            torch.exp_(ys)
            self.train_dataset = self.train_dataset + (ys,)
        # Store the original model "just in case"
        self.__original_model = model
        model = _ExpModel(model, scale)
        if isinstance(self.__original_model, torch.jit.ScriptModule):
            model = torch.jit.script(model)
        return model

    def _prepare_dataset(
        self, dataset: Tuple[Tensor, Tensor], test=False
    ) -> Tuple[Tensor, Tensor]:
        r"""Prepares the training dataset from Monte Carlo data. This applies
        some common transformation such as extracting signs or amplitudes.
        """
        if dataset is None:
            return dataset
        # fmt: off
        if not isinstance(dataset, (tuple, list)) or \
                not all((isinstance(t, Tensor) for t in dataset)):
            raise TypeError(
                "dataset must be a tuple of Tensors; received {} instead"
                "".format(type(dataset)))
        # fmt: on
        with torch.no_grad():
            xs, _ys = dataset
            if self.target is Target.SIGN:
                ys = (_ys < 0).to(torch.long).squeeze()
                if test:
                    return xs, ys, _ys ** 2
            else:
                assert self.target is Target.AMPLITUDE
                ys = torch.abs(_ys).squeeze()
            return xs, ys

    def _get_loss_function(self):
        return {Target.SIGN: CrossEntropyLoss(), Target.AMPLITUDE: overlap_loss_fn}[
            self.target
        ]

    def _get_eval_metrics(self):
        return {
            Target.AMPLITUDE: {"overlap_error": _FidelityMetric()},
            Target.SIGN: {
                "accuracy": _AccuracyMetric(
                    lambda t: (torch.argmax(t[0], dim=1), t[1])
                ),
                "cross_entropy": _LossMetric(loss_fn=CrossEntropyLoss()),
            },
        }[self.target]

    def _get_test_metrics(self):
        if self.test_dataset is None:
            return {}
        return {
            Target.AMPLITUDE: {"overlap_error": _FidelityMetric()},
            Target.SIGN: {
                "accuracy": _AccuracyMetric(
                    lambda t: (torch.argmax(t[0], dim=1), t[1])
                ),
                "weighted_accuracy": _AccuracyMetric(
                    lambda t: (torch.argmax(t[0], dim=1), t[1], t[2])
                ),
                "cross_entropy": _LossMetric(loss_fn=CrossEntropyLoss()),
            },
        }[self.target]

    def _run_once_on_dataset(self, global_step: int) -> int:
        self.model.train()
        for batch in self._train_loader:
            self._optimiser.zero_grad()
            inputs, *other = batch
            outputs = self.model(inputs)
            loss = self._loss_fn(outputs, *other)
            loss.backward()
            if self.debug:
                self._tb_writer.add_scalar("training/loss", loss.item(), global_step)
            self._optimiser.step()
            global_step += 1
        return global_step

    def _run_metrics_on_dataset(self, metrics, loader):
        if len(metrics) == 0:
            return {}
        self.model.eval()
        with torch.no_grad():
            for m in metrics.values():
                m.reset()
            for batch in loader:
                inputs, *other = batch
                outputs = self.model(inputs)
                for m in metrics.values():
                    m.update((outputs, *other))
            return dict((name, metric.compute()) for name, metric in metrics.items())

    def _run_eval_on_dataset(self, epoch):
        results = self._run_metrics_on_dataset(self._eval_metrics, self._eval_loader)
        for name, value in results.items():
            self._tb_writer.add_scalar("validation/{}".format(name), value, epoch)
        return results

    def _run_test_on_dataset(self, epoch):
        results = self._run_metrics_on_dataset(self._test_metrics, self._test_loader)
        for name, value in results.items():
            self._tb_writer.add_scalar("test/{}".format(name), value, epoch)
        return results

    def _when_to_eval(self):
        if self.debug:
            return set(range(self.config.max_epochs + 1))
        return set(np.linspace(0, self.config.max_epochs, 10, dtype=np.int64))

    def _when_to_test(self):
        if self.debug:
            return set(np.linspace(0, self.config.max_epochs, 5, dtype=np.int64))
        return set()

    def _internal_train(self):
        should_eval = self._when_to_eval()
        should_test = self._when_to_test()

        if 0 in should_test:
            self._run_test_on_dataset(0)
        if 0 in should_eval:
            r = self._run_eval_on_dataset(0)
            if self.target is Target.SIGN and r.get("accuracy", 0) == 1:
                return
        iteration = 0

        for epoch in range(1, self.config.max_epochs + 1):
            start = time.time()
            iteration = self._run_once_on_dataset(iteration)
            self._tb_writer.add_scalar(
                "training/time_per_epoch", time.time() - start, epoch
            )
            if epoch in should_test:
                self._run_test_on_dataset(epoch)
            if epoch in should_eval:
                r = self._run_eval_on_dataset(epoch)
                if self.target is Target.SIGN and r.get("accuracy", 0) == 1:
                    break

    def run(self):
        self._train_loader = TensorIterableDataset(
            *self.train_dataset, shuffle=True, batch_size=self.config.train_batch_size
        )
        self._eval_loader = TensorIterableDataset(
            *self.train_dataset, shuffle=False, batch_size=self.config.val_batch_size
        )
        if self.test_dataset is not None:
            self._test_loader = TensorIterableDataset(
                *self.test_dataset, shuffle=False, batch_size=self.config.val_batch_size
            )
        self._internal_train()
        self._tb_writer.flush()
        if self.target is Target.AMPLITUDE:
            # Since we're transformed the model, it's a good idea to check that
            # we've actually updated the weights in the original model as well
            state_dict = self.model.state_dict()
            for name, tensor in self.__original_model.state_dict().items():
                assert torch.all(tensor == state_dict["module." + name])


def train_amplitude(model, dataset, output, config, test_dataset):
    ManualTrainer(
        Target.AMPLITUDE,
        model,
        dataset,
        config._replace(output=output),
        test_dataset=test_dataset,
    ).run()


def train_sign(model, dataset, output, config, test_dataset):
    ManualTrainer(
        Target.SIGN,
        model,
        dataset,
        config._replace(output=output),
        test_dataset=test_dataset,
    ).run()


@torch.jit.script
def _safe_real_exp(values: Tensor, normalise: bool = True) -> Tensor:
    r"""``values`` is a 2D tensor representing logarithms of the wavefunction.
    First column is the real part of the logarithm and the second column --
    imaginary part. It is assumed that the wavefunction has one common complex
    phase. We compute the exponent of the log of the wavefunction eliminating
    this common phase. The resulting wavefunction is thus purely real.
    """
    assert values.dim() == 2 and values.size(1) == 2
    amplitude = values[:, 0]
    # This ensures that the biggest real part of the logarithm is 0 making it
    # perfectly safe to compute the exponent
    amplitude -= torch.max(amplitude)
    torch.exp_(amplitude)
    if normalise:
        amplitude /= torch.norm(amplitude)
    # The following eliminates the common phase. It might not be the fastest
    # solution, but since this code is not performance critical, it's okay.
    phase = values[:, 1]
    phase /= 3.141592653589793
    torch.round_(phase)
    torch.abs_(phase)
    phase.fmod_(2.0)
    # Combines amplitudes and signs
    return amplitude * (1.0 - 2.0 * phase)


class Runner(object):
    def __init__(self, config):
        self.config = config
        self.amplitude, self.sign = self.config.model
        self.sampling_options = SamplingOptions(
            self.config.number_samples, self.config.number_chains
        )
        self.exact = load_exact(self.config.exact)
        self.tb_writer = SummaryWriter(log_dir=self.config.output)
        self._iteration = 0

    def compute_statistics(self):
        tqdm.write("Computing statistics for {}...".format(self._iteration))
        if self.exact is None:
            return None

        self.amplitude.eval()
        self.sign.eval()
        with torch.no_grad():
            basis = self.config.hamiltonian.basis
            y = forward_with_batches(
                self.combined_state,
                torch.from_numpy(basis.states.view(np.int64)),
                batch_size=8192,
            )
            y = _safe_real_exp(y, normalise=True).cpu().numpy()

            overlap = abs(np.dot(y.conj(), self.exact))
            # Hy <- H*y
            Hy = self.config.hamiltonian(y)
            # E <- y.conj() * H * y = y.o
            energy = np.dot(y.conj(), Hy)
            Hy -= energy * y
            variance = np.linalg.norm(Hy) ** 2

            self.tb_writer.add_scalar("overlap", overlap, self._iteration)
            self.tb_writer.add_scalar("energy_real", energy.real, self._iteration)
            self.tb_writer.add_scalar("energy_imag", energy.imag, self._iteration)
            self.tb_writer.add_scalar("variance", variance, self._iteration)
            return overlap, energy, variance

    @property
    def combined_state(self):
        if not hasattr(self, "__combined_state"):
            self.__combined_state = core.combine_amplitude_and_sign(
                self.amplitude, self.sign, apply_log=False, out_dim=2
            )
        return self.__combined_state

    def apply_polynomial(self, spins: Tensor) -> Tensor:
        with torch.no_grad():
            log_ψ = self.combined_state._c._get_method("forward")
            polynomial = _C.Polynomial(
                self.config.hamiltonian, self.config.roots(self._iteration)
            )
            values = _C.apply(spins, polynomial, log_ψ)
            values = _safe_real_exp(values, normalise=True)
            return values

    def monte_carlo(self):
        tqdm.write("Monte Carlo sampling from |ψ|²...", end="")
        start = time.time()

        self.amplitude.eval()
        self.sign.eval()
        spins, _, _ = sample_some(
            self.amplitude,
            self.config.hamiltonian.basis,
            self.sampling_options,
            mode="exact",
        )
        acceptance = 1.0
        values = self.apply_polynomial(spins)

        stop = time.time()
        tqdm.write(
            " Done in {:.2f} seconds. ".format(stop - start)
            + "Acceptance {:.2f}%".format(100 * acceptance)
        )
        return spins, values

    def load_checkpoint(self, i: int):
        def load(target, model):
            pattern = os.path.join(self.config.output, str(i), target, "best_model_*")
            [filename] = glob.glob(pattern)
            model.load_state_dict(torch.load(filename))

        load("amplitude", self.amplitude)
        load("sign", self.sign)

    def prepare_test_dataset(self):
        tqdm.write("Preparing test dataset...", end="")
        start = time.time()

        basis = self.config.hamiltonian.basis
        basis.build()
        spins = torch.from_numpy(basis.states.view(np.int64)).view(-1, 1)
        spins = torch.cat(
            [spins, torch.zeros(spins.size(0), 7, dtype=torch.int64)], dim=1
        )
        values = self.apply_polynomial(spins)

        stop = time.time()
        tqdm.write(" Done in {:.2f} seconds. ".format(stop - start))
        return spins, values

    def step(self):
        if self._iteration == 0:
            self.compute_statistics()

        dataset = self.monte_carlo()
        test_dataset = None # self.prepare_test_dataset()
        train_amplitude(
            self.amplitude,
            dataset,
            "{}/{}/amplitude".format(self.config.output, self._iteration),
            self.config.amplitude(self._iteration),
            test_dataset=test_dataset,
        )
        train_sign(
            self.sign,
            dataset,
            "{}/{}/sign".format(self.config.output, self._iteration),
            self.config.sign(self._iteration),
            test_dataset=test_dataset,
        )
        # train(
        #     "amplitude",
        #     self.amplitude,
        #     dataset,
        #     basis,
        #     "{}/{}/amplitude".format(self.config.output, self._iteration),
        #     self.config.amplitude,
        # )
        # basis = self.config.hamiltonian.basis
        # train(
        #     "sign",
        #     self.sign,
        #     dataset,
        #     basis,
        #     "{}/{}/sign".format(self.config.output, self._iteration),
        #     self.config.sign,
        # )
        torch.jit.save(
            self.amplitude,
            "{}/{}/amplitude.pt".format(self.config.output, self._iteration),
        )
        torch.jit.save(
            self.sign, "{}/{}/sign.pt".format(self.config.output, self._iteration)
        )
        self._iteration += 1
        metrics = self.compute_statistics()
        self.tb_writer.flush()


def run(config: Config):
    # Running the simulation
    runner = Runner(config)
    for i in range(config.epochs):
        runner.step()
