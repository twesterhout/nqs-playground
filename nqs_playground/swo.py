#!/usr/bin/env python3

from collections import namedtuple
import glob
from math import sqrt
import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np

import ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import ignite.contrib.handlers.tqdm_logger

import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from ._C_nqs import v2 as _C  # Makes _C.v2 look like _C
from . import core
from .core import SamplingOptions
from .supervised import subtract

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
    # Some sane defaults
    defaults=[
        16,  # train_batch_size
        50,  # max_epochs
        0.80,  # train_fraction
        10,  # patience
        lambda m: torch.optim.RMSprop(
            m.parameters(), lr=1e-4, weight_decay=1e-5
        ),  # optimiser
        4096,  # val_batch_size
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


def forward_with_batches(f, xs, batch_size: int) -> torch.Tensor:
    r"""Applies ``f`` to all ``xs`` propagating no more than ``batch_size``
    samples at a time. ``xs`` is split into batches along the first dimension
    (i.e. dim=0). ``f`` must return a torch.Tensor.
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
):
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

    def make_xs_and_ys():
        assert mode in {"exact", "full"}
        basis.build()  # Initialises internal cache if not done already
        # Compute log amplitudes
        xs = basis.states
        # ys are log amplitudes on all states xs
        ys = forward_with_batches(
            lambda xs: log_ψ(_C.unpack(xs, basis.number_spins)), xs, batch_size=8192
        ).squeeze()
        if ys.dim() != 1:
            raise ValueError(
                "log_ψ should return real part of the logarithm of the "
                "wavefunction, but output tensor has dimension {}; did "
                "you by accident use sign instead of amplitude network?"
                "".format(ys.dim())
            )
        return xs, ys

    if mode == "exact":
        with torch.no_grad(), torch.jit.optimized_execution(True):
            xs, ys = make_xs_and_ys()
            probabilities = core._log_amplitudes_to_probabilities(ys)
            if len(probabilities) < (1 << 24):
                # PyTorch only supports discrete probability distributions
                # shorter than 2²⁴.
                indices = torch.multinomial(
                    probabilities,
                    num_samples=options.number_chains * options.number_samples,
                    # replacement=True is IMPORTANT because it more closely
                    # emulates the actual Monte Carlo behaviour
                    replacement=True,
                )
            else:
                # If we have more than 2²⁴ different probabilities chances are,
                # NumPy will complain about probabilities not being normalised
                # since float32 precision is not enough. The simplest
                # workaround is to convert the probabilities to float64 and
                # then renormalise then which is what we do.
                probabilities = probabilities.to(torch.float64)
                probabilities /= torch.sum(probabilities)
                indices = np.random.choice(
                    len(probabilities),
                    size=options.number_chains * options.number_samples,
                    replace=True,
                    p=probabilities,
                )
            return xs[indices], ys[indices], {"xs": xs, "ys": ys}
    elif mode == "full":
        with torch.no_grad(), torch.jit.optimized_execution(True):
            xs, ys = make_xs_and_ys()
            return xs, ys, None
    elif mode == "monte_carlo":
        raise NotImplementedError()
    else:
        raise ValueError(
            'invalid mode: {!r}; expected either "exact", "full", or '
            '"monte_carlo"'.format(mode)
        )


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
    :param optimiser: PyTorch optimiser to use.
    """
    return create_supervised_trainer(
        model, optimiser, loss_fn=torch.nn.CrossEntropyLoss()
    )


def _create_sign_evaluator(model: torch.nn.Module) -> ignite.engine.Engine:
    r"""Creates an engine for evaluating how well the network learned the sign
    structure.

    Evaluator computes two metrics: accuracy and cross entropy loss. Accuracy
    is simply the percentage of signs that the model predicted correctly. Cross
    entropy loss is also averaged over all the samples.

    :param model: PyTorch model. Given a ``torch.FloatTensor`` of
        shape ``(batch_size, num_spins)``, ``model`` must return a
        ``torch.FloatTensor`` of shape ``(batch_size, 2)``. Columns of the
        output tensor are interpreted as unnormalised probabilities of spin
        configurations having signs ``+1`` or ``-1``.
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


def _create_amplitude_trainer(
    model: torch.nn.Module, optimiser: torch.optim.Optimizer
) -> ignite.engine.Engine:
    r"""Creates an engine for training the network on wavefunction amplitudes.

    The model is trained to maximise the overlap with the target state.

    :param model: PyTorch model. Given a ``torch.FloatTensor`` of
        shape ``(batch_size, num_spins)``, ``model`` must return a
        ``torch.FloatTensor`` of shape ``(batch_size, 1)``. It is interpreted
        as **wavefunction amplitudes** (NOTE: no logarithms!).
    :param optimiser: PyTorch optimiser.
    """

    # The following loss function closely resembles something what one would
    # write if we were simply trying to compute overlap from Monte Carlo data.
    @torch.jit.script
    def loss_fn(predicted, expected):
        assert torch.all(predicted >= 0) and torch.all(expected >= 0)
        predicted = predicted.squeeze()
        expected = expected.squeeze()
        quotient = expected / predicted
        return 1 - torch.sum(quotient) ** 2 / torch.norm(
            quotient
        ) ** 2 / predicted.size(0)

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


def _create_amplitude_evaluator(model: torch.nn.Module) -> ignite.engine.Engine:
    r"""Creates an engine for evaluating how well the network learned
    wavefunction amplitude.
    """
    return create_supervised_evaluator(
        model,
        metrics={
            "weighted_fidelity": FidelityMetric(weighted=True),
            "fidelity": FidelityMetric(weighted=False),
        },
    )


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
    r"""Creates an engine for evaluating the performance of the network trained
    on either amplitude or sign of the wavefunction.

    :param target: either "amplitude" or "sign".
    :param model: PyTorch model.
    """
    make = {"sign": _create_sign_evaluator, "amplitude": _create_amplitude_evaluator}[
        target
    ]
    return namedtuple("Evaluators", ["training", "validation"])(
        make(model), make(model)
    )


class OverlapMetric(ignite.metrics.metric.Metric):
    r"""An Ignite metric for computing overlap of with the target state.
    """

    def __init__(self, output_transform=lambda x: x, weighted=True):
        self._dot = None
        self._norm_predicted = None
        self._norm_expected = None
        self._weighted = weighted
        super().__init__(output_transform=output_transform)

    def reset(self):
        self._dot = 0
        self._norm_predicted = 0
        self._norm_expected = 0

    def update(self, output):
        predicted, expected = output
        predicted = predicted.squeeze()
        expected = expected.squeeze()
        assert torch.all(predicted >= 0) and torch.all(expected >= 0)
        if self._weighted:
            self._dot += torch.dot(predicted, expected).item()
            self._norm_predicted += torch.norm(predicted).item() ** 2
            self._norm_expected += torch.norm(expected).item() ** 2
        else:
            quotient = expected / predicted
            self._dot += torch.sum(quotient).item()
            self._norm_predicted += predicted.size(0)
            self._norm_expected += torch.norm(quotient).item() ** 2

    def compute(self):
        if self._norm_expected == 0:
            raise ignite.exceptions.NotComputableError(
                "OverlapMetric must have at least one example before it can be computed."
            )
        return abs(self._dot) / sqrt(self._norm_predicted * self._norm_expected)


class FidelityMetric(OverlapMetric):
    def __init__(self, output_transform=lambda x: x, weighted=True):
        super().__init__(output_transform=output_transform, weighted=weighted)

    def compute(self):
        return 1.0 - super().compute() ** 2


class Trainer:
    r"""Class which implements supervised learning.
    """

    def __init__(
        self,
        target: str,
        model: torch.nn.Module,
        dataset: Tuple[np.ndarray, torch.Tensor],
        basis: _C.SpinBasis,
        output: str,
        config: TrainingOptions,
    ):
        r"""Initialises the trainer.

        :param target: what are we learning? Either "amplitude" or "sign".
        :param model: model to train. Must predict either signs or log
            amplitudes.
        :param dataset: a pair of inputs and outputs constituting the full
            dataset available to us for training. Trainer automatically splits
            it into training and validation datasets.
        :param basis: Hilbert space basis.
        :param output: directory where to save log files and model weights.
        :param config: training hyperparameters such as learning rate, fraction
            of the dataset to use for validation etc.
        """
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

            # Since we're optimising overlap and model predicts log amplitudes,
            # we first transform it to predict actual amplitudes. Effectively
            # we just want to add a new layer taking the exponent, but this can
            # overflow. So instead we first determine the maximal predicted
            # value in the dataset and then rescale the output to prevent
            # overflows.
            class _Model(torch.nn.Module):
                __constants__ = ["scale"]

                def __init__(self, module, scale):
                    super().__init__()
                    self.module = module
                    self.scale = scale

                def forward(self, x):
                    return torch.exp(self.module(x) - self.scale)

            with torch.no_grad(), torch.jit.optimized_execution(True):
                xs, _ = dataset
                ys = forward_with_batches(
                    lambda x: model(_C.unpack(x, self.basis.number_spins)),
                    xs,
                    batch_size=8192,
                )
                scale = float(torch.max(ys))  # - 5.0
            self.original_model = model
            self.model = _Model(model, scale)

    def __add_loaders(self, dataset: Tuple[np.ndarray, torch.Tensor]):
        with torch.no_grad():
            # Let us transform the whole dataset now once so that we don't have
            # to do it on a per-batch basis.
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
                core.make_spin_dataloader(
                    *dataset,
                    batch_size=self.config.train_batch_size,
                    drop_last=True,
                    unpack=lambda x, i: _C.unpack(x, i, self.basis.number_spins),
                )
            )
        else:
            train, val = core.random_split(
                dataset, self.config.train_fraction, weights=None, replacement=False
            )
            # Makes sure that training and validation datasets don't overlap.
            # This can potentially screw up the distribution in the validation
            # dataset if train_fraction is close to 1.
            val = subtract(val, train)
            tqdm.write(
                "Training dataset contains {} samples and validation -- {} "
                "samples".format(train[0].shape[0], val[0].shape[0])
            )
            self.loaders = Loaders(
                core.make_spin_dataloader(
                    *train,
                    batch_size=self.config.train_batch_size,
                    drop_last=True,
                    unpack=lambda x, i: _C.unpack(x, i, self.basis.number_spins),
                ),
                core.make_spin_dataloader(
                    *val,
                    batch_size=self.config.val_batch_size,
                    unpack=lambda x: _C.unpack(x, self.basis.number_spins),
                    shuffle=False,
                )
                if val[0].shape[0] > 0
                else None,
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
            return lambda e: -e.state.metrics["fidelity"]

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
        r"""Loads back weights which achieved the best score during training."""
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
            # Since we're transformed the model, it's a good idea to check that
            # we've actually updated the weights in the original model as well
            state_dict = self.model.state_dict()
            for name, tensor in self.original_model.state_dict().items():
                assert torch.all(tensor == state_dict["module." + name])


def train(target, model, dataset, basis, output, config):
    r"""Runs supervised training."""
    Trainer(target, model, dataset, basis, output, config).run()


@torch.jit.script
def _safe_real_exp(values: torch.Tensor) -> torch.Tensor:
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
        self.polynomial = _C.Polynomial(self.config.hamiltonian, self.config.roots)
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
            log_ψ = core.combine_amplitude_and_sign(
                self.amplitude, self.sign, apply_log=False, out_dim=2
            )
            y = forward_with_batches(
                lambda x: log_ψ(_C.unpack(x, basis.number_spins)),
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
            log_ψ = core.combine_amplitude_and_sign(
                self.amplitude, self.sign, apply_log=False, out_dim=2
            )
            values = forward_with_batches(
                _C.PolynomialState(
                    self.polynomial, log_ψ._c._get_method("forward"), 8192
                ),
                spins,
                batch_size=32,
            )
            values = _safe_real_exp(values)

        size = len(set(spins))
        stop = time.time()
        tqdm.write(
            " Done in {:.2f} seconds. ".format(stop - start)
            + "Acceptance {:.2f}%. ".format(100 * acceptance)
            + "Sampled {} different spin configurations".format(size)
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


def run(config: Config):
    # Running the simulation
    runner = Runner(config)
    for i in range(config.epochs):
        runner.step()
