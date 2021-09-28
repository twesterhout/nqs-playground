# from itertools import islice
from collections import namedtuple
import math
import time
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
from random import randint
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from loguru import logger
import numpy as np
import torch
from torch import Tensor
import lattice_symmetries as ls

# from ._C import MetropolisGenerator, ZanellaGenerator, zanella_choose_samples, random_spin
from .core import forward_with_batches, pad_states, as_spins_tensor, safe_exp, pack
from ._extension import lib

__all__ = [
    "SamplingOptions",
    "sample_some",
    # "sample_full",
    # "sample_exactly",
    # "sample_autoregressive",
    # "metropolis_process",
    # "sample_using_metropolis",
    # "zanella_process",
    # "sample_using_zanella",
    # "autocorr_function",
    "integrated_autocorr_time",
    "sampled_histogram",
    "are_close_l1",
]


_SamplingOptionsBase = namedtuple(
    "_SamplingOptions",
    [
        "number_samples",
        "number_chains",
        "number_discarded",
        "sweep_size",
        "mode",
        "device",
        "other",
    ],
)


class SamplingOptions(_SamplingOptionsBase):
    r"""Options for sampling spin configurations."""

    def __new__(
        cls,
        number_samples: int,
        number_chains: int = 1,
        number_discarded: Optional[int] = None,
        sweep_size: Optional[int] = None,
        mode: Optional[str] = None,
        device: Union[str, torch.device] = None,
        other: Optional[Dict[str, Any]] = None,
    ):
        r"""Create SamplingOptions.

        Parameters
        ----------
        number_samples: int
            Number of samples per Markov chain. Must be a positive integer.
            'full' sampler will ignore this parameter.
        number_chains: int, optional
            Number of independent Markov chains. Must be a positive integer.
            This parameter only makes sense for MCMC samplers such as
            Metropolis-Hastings algorithm or Zanella process. Exact samplers
            ('exact' and 'autoregressive') will just multiply `number_samples`
            by `number_chains`.
        number_discarded: int, optional
            Number of samples to discard at the beginning of each Markov chain
            (i.e. how long the thermalization procedure should be). If
            specified, must be a positive integer. Otherwise, 10% of
            `number_samples` will be used. This parameter only makes sense for
            MCMC samplers (i.e. 'exact', 'autoregressive', and 'full' samplers
            will ignore this argument).
        sweep_size: int, optional
            Sweep size, i.e. how many Markov chain steps are made until the
            next sample is saved. `sweep_size = 1` means that every sample is
            saved. `sweep_size = 5` means that per every 5 steps of the MCMC
            process we only store one sample. If not specified, the default
            value of `1` will be used. This parameter only makes sense for MCMC
            samplers (i.e. 'exact', 'autoregressive', and 'full' samplers will
            ignore this argument).
        mode: str, optional
            Which algorithm to use for sampling. Valid choices are:

              * `metropolis` -- use Metropolis-Hastings algorithm with 1- or
                2-spin flips.
              * `zanella` -- use Zanella algorithm with 2-spin flips.
              * `exact` -- exactly sample from the discrete probability
                distribution using `torch.multinomial` or
                `numpy.random.choice`. This algorithm works for small systems
                only.
              * `full` -- skip sampling altogether and just return the full
                Hilbert space basis. This algorithm works for small systems
                only.
              * `autoregressive` -- assume that the probability distribution
                has a custom `sample` method and use it.
        device: str or torch.device
            On which device to run the sampling.
        other: Dict[str, Any]
            Extra arguments for a specific sampler.
        """
        number_samples = int(number_samples)
        if number_samples <= 0:
            raise ValueError("negative number_samples: {}".format(number_samples))
        number_chains = int(number_chains)
        if number_chains <= 0:
            raise ValueError("negative number_chains: {}".format(number_chains))

        if number_discarded is not None:
            number_discarded = int(number_discarded)
            if number_discarded < 0:
                raise ValueError(
                    "invalid number_discarded: {}; expected either a non-negative "
                    "integer or None".format(number_chains)
                )
        else:
            logger.info(
                "`number_discarded` not specified when constructing SamplingOptions, "
                "1/10 of `number_samples` will be used."
            )
            number_discarded = number_samples // 10
        if sweep_size is not None:
            sweep_size = int(sweep_size)
            if sweep_size <= 0:
                raise ValueError("negative sweep_size: {}".format(sweep_size))
        else:
            sweep_size = 1
            logger.warning(
                "`sweep_size` not specified when constructing SamplingOptions, "
                "`sweep_size` will be set to 1. Make sure this is what you want!"
            )
        if device is not None and not isinstance(device, torch.device):
            device = torch.device(device)
        if other is None:
            other = dict()
        return super(SamplingOptions, cls).__new__(
            cls, number_samples, number_chains, number_discarded, sweep_size, mode, device, other
        )


def _determine_batch_size(options: SamplingOptions) -> int:
    batch_size = options.other.get("batch_size")
    if batch_size is None:
        batch_size = 8192
        logger.debug("'batch_size' not specified, will use the default value of 8192.")
    else:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(
                "invalid 'batch_size': {}; expected a positive integer".format(batch_size)
            )
    return batch_size


def _check_log_prob_shape(log_prob: Tensor, device: Optional[torch.device]) -> None:
    if log_prob.dim() != 1:
        raise ValueError(
            "log_prob_fn should return the logarithm of the probability, "
            "but output tensor has dimension {}; did you by accident use "
            "sign instead of amplitude network?"
            "".format(log_prob.dim())
        )
    if device is not None and log_prob.device != device:
        raise ValueError(
            "log_prob_fn should return tensors residing on {}; received "
            "tensors residing on {} instead; make sure options.device matches "
            "the location of log_prob_fn".format(device, log_prob.device)
        )


def sample_full(
    log_prob_fn: Callable[[Tensor], Tensor], basis: ls.SpinBasis, options: SamplingOptions
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    r"""Instead of sampling, take all basis vectors in the Hilbert space."""
    batch_size = _determine_batch_size(options)
    device = options.device
    states = torch.from_numpy(basis.states.view(np.int64))
    if device is not None:
        states = states.to(device)
    logger.debug(
        "Applying 'log_prob_fn' to all basis vectors in the Hilbert space using batch_size={}..."
        "".format(batch_size)
    )
    log_prob = forward_with_batches(log_prob_fn, states, batch_size=batch_size, device=device)
    if log_prob.dim() > 1:
        log_prob.squeeze_(dim=1)
    _check_log_prob_shape(log_prob, device)
    logger.debug("Computing weights...")
    log_prob = log_prob.unsqueeze_(dim=1)
    weights = safe_exp(log_prob, normalise=True)
    states = pad_states(states).unsqueeze_(dim=1)
    return states, log_prob, {"weights": weights}


def sample_exactly(
    log_prob_fn: Callable[[Tensor], Tensor], basis: ls.SpinBasis, options: SamplingOptions
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    r"""Sample states by explicitly constructing the discrete probability distribution.

    Number of samples is `options.number_chains * options.number_samples`, and
    `options.number_discarded` and `options.sweep_size` are ignored, since
    samples are already i.i.d.
    """
    states, log_prob, _extra = sample_full(log_prob_fn, basis, options)
    states = states.squeeze_(dim=1)
    log_prob = log_prob.squeeze_(dim=1)
    prob = _extra["weights"].squeeze_(dim=1)
    number_samples = options.number_chains * options.number_samples
    if len(prob) < (1 << 24):
        logger.debug("Using torch.multinomial to sample indices...")
        # PyTorch only supports discrete probability distributions
        # shorter than 2²⁴.
        # NOTE: replacement=True is IMPORTANT because it more closely
        # emulates the actual Monte Carlo behaviour
        indices = torch.multinomial(prob, num_samples=number_samples, replacement=True)
    else:
        logger.debug("Using numpy.random.choice to sample indices...")
        # If we have more than 2²⁴ different probabilities chances are,
        # NumPy will complain about probabilities not being normalised
        # since float32 precision is not enough. The simplest
        # workaround is to convert the probabilities to float64 and
        # then renormalise which is what we do.
        prob = prob.to(device="cpu", dtype=torch.float64)
        prob /= torch.sum(prob)
        indices = np.random.choice(len(prob), size=number_samples, replace=True, p=prob)
        indices = torch.from_numpy(indices).to(device)

    # Choose the samples
    log_prob = log_prob[indices]
    states = states[indices]
    shape = (options.number_samples, options.number_chains)
    return states.view(*shape, 8), log_prob.view(*shape), dict()


# def sample_autoregressive(model: torch.nn.Module, basis, options: SamplingOptions):
#     if not hasattr(model, "sample"):
#         raise ValueError(
#             "Model has no 'sample' method. Did you try to use a standard neural "
#             "network with 'autoregressive' sampling mode?"
#         )
#     number_samples = options.number_chains * options.number_samples
#     logger.debug("Running model.sample...")
#     states = model.sample(number_samples).to(options.device)
#     r = (states, None, None)
#     states = pack(states)
#     if states.dim() < 2:
#         states = _pad_states(states)
#     shape = (options.number_samples, options.number_chains)
#     return states.view(*shape, 8), None, {}


# def metropolis_process(
#     initial_state: Tensor,
#     _log_prob_fn: Callable[[Tensor], Tensor],
#     kernel_fn: Callable[[Tensor], Tuple[Tensor, Tensor]],
#     number_samples: int,
#     number_discarded: int,
#     sweep_size: int,
# ) -> Tuple[Tensor, Tensor, Tensor]:
#     assert number_samples >= 1
#
#     def log_prob_fn(x):
#         y = _log_prob_fn(x)
#         if y.dim() > 1:
#             y.squeeze_(dim=1)
#         return y
#
#     current_state, current_norm = initial_state
#     current_log_prob = log_prob_fn(current_state)
#     dtype = current_log_prob.dtype
#     device = current_log_prob.device
#     current_norm = current_norm.to(dtype)
#     states = current_state.new_empty((number_samples,) + current_state.size())
#     log_probs = current_log_prob.new_empty((number_samples,) + current_log_prob.size())
#     accepted = torch.zeros(current_state.size(0), dtype=torch.int64, device=device)
#
#     states[0].copy_(current_state)
#     log_probs[0].copy_(current_log_prob)
#     current_state = states[0]
#     current_log_prob = log_probs[0]
#
#     def sweep():
#         nonlocal accepted
#         for i in range(sweep_size):
#             proposed_state, proposed_norm = kernel_fn(current_state, dtype)
#             state_is_valid = proposed_norm > 0
#             proposed_log_prob = current_log_prob.new_zeros((proposed_state.size(0),))
#             proposed_log_prob[state_is_valid] = log_prob_fn(proposed_state[state_is_valid])
#             r = torch.rand(current_state.size(0), device=device, dtype=dtype)
#             r *= current_norm
#             r /= proposed_norm
#             t = r <= torch.exp_(proposed_log_prob - current_log_prob)
#             t &= state_is_valid
#             current_state[t] = proposed_state[t]
#             current_log_prob[t] = proposed_log_prob[t]
#             current_norm[t] = proposed_norm[t]
#             accepted += t
#
#     # Thermalisation
#     for i in range(number_discarded):
#         sweep()
#
#     # Reset acceptance count after thermalisation
#     accepted.fill_(0)
#     for i in range(1, number_samples):
#         states[i].copy_(current_state)
#         log_probs[i].copy_(current_log_prob)
#         current_state = states[i]
#         current_log_prob = log_probs[i]
#         sweep()
#
#     # Subtract 1 because the loop above starts at 1
#     acceptance = accepted.double() / ((number_samples - 1) * sweep_size)
#     return states, log_probs, acceptance
#
#
# def sample_using_metropolis(
#     log_prob_fn: Callable[[Tensor], Tensor], basis, options: SamplingOptions
# ):
#     initial_state = prepare_initial_state(basis, options.number_chains)
#     initial_norm = ls.batched_state_info(basis, initial_state.numpy().view(np.uint64))[2]
#     initial_norm *= initial_norm  # We need 1/N rather than 1/√N
#     initial_state = initial_state.to(options.device)
#     initial_norm = torch.from_numpy(initial_norm).to(options.device)
#     kernel_fn = MetropolisGenerator(basis)
#     t1 = time.time()
#     states, log_probs, acceptance = metropolis_process(
#         (initial_state, initial_norm),
#         log_prob_fn,
#         kernel_fn,
#         options.number_samples,
#         options.number_discarded,
#         options.sweep_size,
#     )
#     t2 = time.time()
#     info = {"acceptance_rate": torch.mean(acceptance).item(), "metropolis_process": t2 - t1}
#     return states, log_probs, info


@torch.jit.script
@torch.no_grad()
def _zanella_jump_rates(current_log_prob: Tensor, possible_log_prob: Tensor) -> Tensor:
    r"""Calculate jump rates (i.e. probabilities) for all possible states.

    Parameters
    ----------
    current_log_prob: Tensor
        A tensor of shape `(number_chains,)` with log probability of current
        state for every Markov chain.
    possible_log_prob: Tensor
        A tensor of shape `(number_chains, max_number_states)` with log
        probability for every possible new state for every chain. Because of
        symmetries, the number of possible new states may vary. We pad the
        tensor with a large negative value (whose `exp` is 0) to ensure that
        the tensor is rectangular.
    """
    (number_chains,) = current_log_prob.size()
    if possible_log_prob.size(0) != number_chains:
        raise ValueError(
            "'possible_log_prob' has wrong shape: {}; expected (number_chains={}, "
            "max_number_states)".format(possible_log_prob.size(), number_chains)
        )
    r = torch.exp_(possible_log_prob - current_log_prob.view(-1, 1))
    return torch.minimum(r, torch.scalar_tensor(1), out=r)


@torch.jit.script
@torch.no_grad()
def _sample_exponential(rates: Tensor, out: Tensor) -> Tensor:
    r"""Sample from exponential distribution with given rates."""
    if rates.size() != out.size():
        raise ValueError("'out' has wrong shape: {}; expected {}".format(out.size(), rates.size()))
    out = torch.rand(rates.size(), out=out)
    out *= -1
    torch.log1p_(out)
    out *= -1
    out /= rates
    return out
    # return out.copy_(torch.distributions.Exponential(rates).sample())


@torch.jit.script
@torch.no_grad()
def _zanella_next_state_index(rates: Tensor) -> Tensor:
    r"""Choose indices of states to which to move.

    Parameters
    ----------
    rates: Tensor
        A tensor of shape `(number_chains, max_number_states)` containing jump
        rates to possible states for every Markov chain.

    Returns
    -------
    A tensor of shape `(number_chains,)` with indices of states to which to jump.
    """
    return torch.multinomial(rates, num_samples=1).view(-1)


@torch.jit.script
@torch.no_grad()
def _zanella_update_current(possible: Tensor, indices: Tensor, out: Tensor) -> Tensor:
    r"""Pick the next state from `possible` based on `indices` and store it to `out`.

    This is equivalent to indexing `possible` along the first dimension.

    :param possible: a tensor of shape `(number_chains, max_number_states, K)`.
    :param indices: a tensor of shape `(number_chains,)`.
    :param out: a tensor of shape `(number_chains, K)`.
    """
    offsets = torch.arange(
        start=0,
        end=possible.size(0) * possible.size(1),
        step=possible.size(1),
        dtype=indices.dtype,
        device=indices.device,
    )
    offsets += indices
    torch.index_select(possible.flatten(0, 1), dim=0, index=offsets, out=out)
    return out


# @torch.no_grad()
# def _pad_log_prob(log_probs: Tensor, counts: Tensor, value: float) -> Tensor:
#     indices = torch.arange(log_probs.size(1), device=log_probs.device).view(1, -1)
#     return log_probs.masked_fill_(indices >= counts.view(-1, 1), value)


@torch.jit.script
@torch.no_grad()
def _flatten_states(states: Tensor, counts: Tensor, total_count: int) -> Tensor:
    flat_states = states.new_empty((total_count, states.size(-1)))
    offset = 0
    for i in range(counts.size(0)):
        n = counts[i].item()
        flat_states[offset : offset + n].copy_(states[i, :n], non_blocking=True)
        offset += n
    return flat_states


@torch.jit.script
@torch.no_grad()
def _unflatten_log_probs(flat_log_probs: Tensor, counts: Tensor, max_count: int) -> Tensor:
    log_probs = flat_log_probs.new_full((counts.size(0), max_count), -1e7)
    offset = 0
    for i in range(counts.size(0)):
        n = counts[i].item()
        log_probs[i, :n].copy_(flat_log_probs[offset : offset + n], non_blocking=True)
        offset += n
    return log_probs


@torch.no_grad()
def zanella_process(
    current_state: Tensor,
    _log_prob_fn: Callable[[Tensor], Tensor],
    generator_fn: Callable[[Tensor], Tuple[Tensor, List[int]]],
    number_samples: int,
    number_discarded: int,
):
    r"""

    :param current_state: State from which to start the process. The first
        dimension is the batch dimension. It corresponds to multiple chains.
    :param log_prob_fn: Function returning the logarithmic probability of a
        state. It must support batching, i.e. work with tensors similar to
        `current_state`.
    :param generator_fn:
    :param number_samples:
    """
    assert number_samples >= 1

    def log_prob_fn(x):
        y = _log_prob_fn(x)
        if y.dim() > 1:
            y.squeeze_(dim=1)
        return y

    # Device is determined by the location of initial state
    device = current_state.device
    (number_chains, configuration_size) = current_state.size()
    current_log_prob = log_prob_fn(current_state)
    # Number of chains is also deduced from current_state. It is simply
    # current_state.size(0). In the following we pre-allocate storage for
    # states and log probabilities.
    states = current_state.new_empty((number_samples,) + current_state.size())
    log_prob = current_log_prob.new_empty((number_samples,) + current_log_prob.size())
    # Weights stores weights of samples, i.e. time we spend sitting there
    weights = current_log_prob.new_empty(number_samples, current_state.size(0))
    # Store results of the first iteration. Note that current_weight is not yet
    # computed! It will be done inside the loop
    # states[0].copy_(current_state)
    # log_prob[0].copy_(current_log_prob)
    # current_state = states[0]
    # current_log_prob = log_prob[0]
    # current_weight = weights[0]

    assert number_chains > 1

    # Main loop. We keep track of the iteration manually since we want to stop
    # in the middle of the loop body rather than at the end. We also keep a
    # flag which indicates whether we are still in the thermalisation phase and
    # that samples should be discarded
    iteration = 0
    discard = True

    def _generate(_current_state):
        return generator_fn(_current_state)

    def _process(possible_states, counts, _current_log_prob, _current_weight):
        total_count = torch.sum(counts).item()
        max_count = possible_states.size(1)
        flat_possible_states = _flatten_states(possible_states, counts, total_count)
        flat_possible_log_probs = log_prob_fn(flat_possible_states)
        possible_log_probs = _unflatten_log_probs(flat_possible_log_probs, counts, max_count)
        jump_rates = _zanella_jump_rates(_current_log_prob, possible_log_probs)
        _sample_exponential(jump_rates.sum(dim=1), out=_current_weight)
        indices = _zanella_next_state_index(jump_rates)
        return indices, possible_log_probs

    m = number_chains // 2

    iterations = itertools.chain(itertools.repeat(0, number_discarded), range(number_samples - 1))

    with ThreadPoolExecutor(max_workers=1) as executor:
        _future = executor.submit(_generate, current_state[:m])
        for i in iterations:
            states[i, :m].copy_(current_state[:m], non_blocking=True)
            log_prob[i, :m].copy_(current_log_prob[:m], non_blocking=True)
            possible_states, counts = _future.result()
            _future = executor.submit(_generate, current_state[m:])
            indices, possible_log_probs = _process(possible_states, counts, current_log_prob[:m], weights[i, :m])
            _zanella_update_current(possible_states, indices, out=current_state[:m])
            _zanella_update_current(possible_log_probs, indices, out=current_log_prob[:m])

            states[i, m:].copy_(current_state[m:], non_blocking=True)
            log_prob[i, m:].copy_(current_log_prob[m:], non_blocking=True)
            possible_states, counts = _future.result()
            _future = executor.submit(_generate, current_state[:m])
            # _generate(current_state[m:])
            indices, possible_log_probs = _process(possible_states, counts, current_log_prob[m:], weights[i, m:])
            _zanella_update_current(possible_states, indices, out=current_state[m:])
            _zanella_update_current(possible_log_probs, indices, out=current_log_prob[m:])

        # Generates all states to which we could jump
        # possible_states, counts = generator_fn(current_state)
        # total_count = torch.sum(counts).item()
        # max_count = possible_states.size(1)
        # flat_possible_states = _flatten_states(possible_states, counts, total_count)
        # flat_possible_log_probs = log_prob_fn(flat_possible_states)
        # possible_log_probs = _unflatten_log_probs(flat_possible_log_probs, counts, max_count)
        # jump_rates = _zanella_jump_rates(current_log_prob, possible_log_probs)
        # Calculate for how long we have to sit in the current state
        # Note that only now have we computed all quantities for `iteration`.
        # _sample_exponential(jump_rates.sum(dim=1), out=current_weight)

        # iteration += 1
        # if discard:
        #     if iteration >= number_discarded:
        #         iteration = 0
        #         discard = False
        # else:
        #     if iteration == number_samples:
        #         return states, log_prob, weights
        #         # break
        #     current_state = states[iteration]
        #     current_log_prob = log_prob[iteration]
        #     current_weight = weights[iteration]

        # Pick the next state
        # indices = _zanella_next_state_index(jump_rates)
        # _zanella_update_current(possible_states, indices, out=current_state)
        # _zanella_update_current(possible_log_probs, indices, out=current_log_prob)
    return states, log_prob, weights


@torch.jit.script
@torch.no_grad()
def _zanella_choose_samples(weights: Tensor, number_samples: int) -> Tensor:
    if number_samples == 1:
        return torch.scalar_tensor(0, dtype=torch.int64, device=weights.device)
    if number_samples == 2:
        return torch.tensor([0, weights.size(0) - 1], dtype=torch.int64, device=weights.device)
    boundaries = weights.new_empty((weights.size(0) + 1,))
    boundaries[0] = 0
    torch.cumsum(weights, dim=0, out=boundaries[1:])
    points = torch.linspace(0, boundaries[-1], number_samples, device=weights.device)
    points[0] = torch.nextafter(torch.scalar_tensor(points[0]), torch.scalar_tensor(boundaries[-1]))
    points[-1] = torch.nextafter(
        torch.scalar_tensor(points[-1]), torch.scalar_tensor(boundaries[0])
    )
    indices = torch.bucketize(points, boundaries=boundaries, right=False) - 1
    # assert torch.all(indices >= 0)
    # assert torch.all(indices < weights.size(0))
    return indices


@torch.no_grad()
def sample_using_zanella(log_prob_fn, basis, options):
    current_state = prepare_initial_state(basis, options.number_chains)
    current_state = current_state.to(options.device)
    edges = options.other.get("edges")
    if edges is None:
        edges: List[Tuple[int, int]] = []
        for i in range(basis.number_spins - 1):
            for j in range(i + 1, basis.number_spins):
                edges.append((i, j))
    generator_fn = lib.ZanellaGenerator(basis, edges)
    sweep_size = options.sweep_size
    t1 = time.time()
    states, log_probs, weights = zanella_process(
        current_state,
        log_prob_fn,
        generator_fn,
        options.number_samples * sweep_size,
        options.number_discarded * sweep_size,
    )
    t2 = time.time()
    # return states, log_probs, weights
    final_states = states.new_empty((options.number_samples,) + states.size()[1:])
    final_log_probs = log_probs.new_empty((options.number_samples,) + log_probs.size()[1:])
    device = final_states.device
    for chain in range(weights.size(1)):
        indices = _zanella_choose_samples(weights[:, chain], options.number_samples)
        torch.index_select(states[:, chain], dim=0, index=indices, out=final_states[:, chain])
        torch.index_select(log_probs[:, chain], dim=0, index=indices, out=final_log_probs[:, chain])
    t3 = time.time()
    return final_states, final_log_probs, {"zanella_process": t2 - t1, "choose_samples": t3 - t2}


@torch.no_grad()
def sample_some(
    log_ψ: Callable[[Tensor], Tensor],
    basis: ls.SpinBasis,
    options: SamplingOptions,
    is_log_prob_fn: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Dict[str, Any]]:
    r"""Sample from |ψ(σ)|².

    Parameters
    ----------
    log_ψ: Callable
        A function which computes log(ψ(σ)) for a batch of spin configurations σ.
    basis: SpinBasis
        Hilbert space basis in which to run sampling. All returned states will
        belong to this basis.
    options: SamplingOptions
        Options specifying number of samples, algorithm etc.
    mode: str, deprecated!
        Please, specify `options.mode` instead.
    is_log_prob_fn: bool
        Whether log_ψ already specifies the probability distribution. If it
        does we do not need to square ψ.
    """
    mode = options.mode
    supported = {"exact", "full", "autoregressive", "metropolis", "zanella"}
    if not mode in supported:
        raise ValueError("invalid mode: {!r}; must be one of {}".format(mode, supported))

    if is_log_prob_fn or mode == "autoregressive":
        log_prob_fn = log_ψ
    else:

        def log_prob_fn(x):
            x = log_ψ(x)
            x *= 2
            return x

    fn = {
        "exact": sample_exactly,
        "full": sample_full,
        # "autoregressive": sample_autoregressive,
        # "metropolis": sample_using_metropolis,
        "zanella": sample_using_zanella,
    }[mode]
    return fn(log_prob_fn, basis, options)


def _random_spins_chunk(basis, size):
    spins = np.zeros((size, 8), dtype=np.uint64)
    if basis.hamming_weight is None:
        assert False
    else:
        m = basis.hamming_weight
        buf = np.empty((size, basis.number_spins), dtype=np.uint64)
        buf[:, :m] = 1
        buf[:, m:] = 0
        buf = np.random.permutation(buf.T).T
        i = 0
        word = 0
        for j in range(basis.number_spins):
            spins[:, word] |= buf[:, j] << i
            i += 1
            if i == 64:
                word += 1
    spins, _, norms = basis.batched_state_info(spins)
    spins = spins[norms > 0]
    spins = np.unique(spins, axis=0)
    return spins


def random_spins(basis: ls.SpinBasis, size: int, scale: int = 2) -> Tensor:
    spins = _random_spins_chunk(basis, scale * size)
    indices = np.random.randint(spins.shape[0], size=size)
    return spins[indices]


def prepare_initial_state(basis, batch_size: int) -> torch.Tensor:
    return torch.from_numpy(random_spins(basis, batch_size).view(np.int64))


def autocorr_function(x: np.ndarray) -> np.ndarray:
    r"""Estimate the normalised autocorrelation function of a 1D array.

    :param x:
    :return:
    """
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if x.ndim != 1:
        raise ValueError("x has wrong shape: {}; expected a 1D array".format(x.shape))
    n = 1 << math.ceil(math.log2(len(x)))
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    autocorr = np.fft.ifft(f * np.conj(f))[: len(x)].real
    autocorr /= autocorr[0]
    return autocorr


def _auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def integrated_autocorr_time(
    x: np.ndarray, c: float = 5.0, with_autocorr_fn: bool = False
) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        # NOTE: We do the computation on the CPU because it's not performance
        # critical and x might be complex which PyTorch doesn't support yet
        x = x.detach().cpu().numpy()
    f = np.zeros(x.shape[0])
    for i in range(x.shape[1]):
        f += autocorr_function(x[:, i])
    f /= x.shape[1]
    taus = 2.0 * np.cumsum(f) - 1.0
    window = _auto_window(taus, c)
    if with_autocorr_fn:
        return taus[window], f
    return taus[window]


@torch.no_grad()
def sampled_histogram(spins: Tensor, basis) -> Tensor:
    assert basis.number_spins <= 64
    if spins.dim() == 2:
        assert spins.size(1) == 8
        spins = spins[:, 0]
    device = spins.device
    spins, counts = torch.unique(spins, sorted=True, return_counts=True)
    indices = basis.batched_index(spins.cpu().numpy().view(np.uint64))
    indices = torch.from_numpy(indices.view(np.int64)).to(device)
    r = torch.zeros(basis.number_states, device=device, dtype=torch.int64)
    r[indices] += counts
    return r


@torch.no_grad()
def are_close_l1(n: int, basis, sample_fn, exact: Tensor, eps: float, options):
    r"""Use L1 norm to compare two probabilities."""
    device = exact.device
    logger.info("Sorting exact probabilities...")
    exact, order = torch.sort(exact)
    # NOTE: We do a copy here until PyTorch v1.5 which introduces searchsorted
    s = np.searchsorted(torch.cumsum(exact, dim=0).cpu().numpy(), eps / 8.0)
    ms = np.random.poisson(n, size=options.number_chains)
    logger.info("Sampling...")
    states, log_prob, info = sample_fn(options._replace(number_samples=max(ms)))
    if log_prob is not None:
        logger.info("Autocorrelation time is {}", integrated_autocorr_time(log_prob))
    if info is not None:
        logger.info("Additional info from the sampler: {}", info)
    logger.info("Computing histograms...")
    states = [sampled_histogram(states[:m, i], basis)[order] for i, m in enumerate(ms)]

    def analyze(x, k):
        v = ((x - k * exact) ** 2 - x) * exact ** (-2 / 3)
        w = exact ** (2 / 3)
        cond1 = torch.sum(v[s:-1]) > 4 * k * torch.sum(w[s:-1]) ** (1 / 2)
        cond2 = torch.sum(x[:s]) > 3 / 16 * eps * k
        return not (cond1 or cond2)

    logger.info("Analyzing histograms...")
    return [analyze(x, k) for x, k in zip(states, ms)]
