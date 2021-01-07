from loguru import logger
import numpy as np
import torch
from typing import List, Optional
import torch.utils.data

# Code for MADE was obtained by combining
#
#   * https://github.com/karpathy/pytorch-made
#   * https://github.com/EugenHotaj/pytorch-generative

from . import *

__all__ = [
    "MaskedLinear",
    "MADE",
    "NADE",
]


class MaskedLinear(torch.nn.Linear):
    """A Linear layer with masks that turn off some of the layer's weights."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask):
        self.mask.data.copy_(mask)
        
    def forward(self, x):
        return torch.nn.functional.linear(x, self.mask * self.weight, self.bias)


def _get_model_device(model: torch.nn.Module):
    for x in model.parameters(recurse=True):
        return x.device
    for x in model.buffers(recurse=True):
        return x.device


class MADE(torch.nn.Module):
    """The Masked Autoencoder Distribution Estimator (MADE) model."""

    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None, n_masks: int = 1):
        """Initializes a new MADE instance.
        Args:
            input_dim: The dimensionality of the input.
            hidden_dims: A list containing the number of units for each hidden layer.
            n_masks: The total number of distinct masks to use during training/eval.
        """
        super().__init__()
        self._input_dim = input_dim
        self._dims = [self._input_dim] + (hidden_dims or []) + [self._input_dim]
        self._n_masks = n_masks
        self._mask_seed = 0
        self._epsilon = 1e-7

        layers = []
        for i in range(len(self._dims) - 1):
            in_dim, out_dim = self._dims[i], self._dims[i + 1]
            layers.append(MaskedLinear(in_dim, out_dim))
            layers.append(torch.nn.ReLU())
        layers[-1] = torch.nn.Sigmoid()
        # layers.pop()  # remove last ReLU
        self._net = torch.nn.Sequential(*layers)

    def _create_masks(self, seed):
        """Sample a new set of autoregressive masks."""
        rng = np.random.RandomState(seed=seed)

        # Sample connectivity patterns.
        conn = [rng.permutation(self._input_dim)]
        for i, dim in enumerate(self._dims[1:-1]):
            low = 0 if i == 0 else np.min(conn[i - 1])
            high = self._input_dim - 1
            conn.append(rng.randint(low, high, size=dim))
        conn.append(np.copy(conn[0]))

        # Create masks.
        masks = [conn[i - 1][None, :] <= conn[i][:, None] for i in range(1, len(conn) - 1)]
        masks.append(conn[-2][None, :] < conn[-1][:, None])

        return [torch.from_numpy(mask.astype(np.uint8)) for mask in masks], conn[-1]

    def _forward(self, x, masks):
        # If the input is an image, flatten it during the forward pass.
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.view(original_shape[0], -1)

        layers = [layer for layer in self._net.modules() if isinstance(layer, MaskedLinear)]
        for layer, mask in zip(layers, masks):
            layer.set_mask(mask)
        return self._net(x).view(original_shape)

    def forward(self, x):
        masks, _ = self._sample_masks()
        return self._forward(x, masks)

    def _log_prob(self, sample, x_hat):
        mask = sample # (sample + 1) / 2
        log_prob = (torch.log(x_hat + self._epsilon) * mask +
                    torch.log(1 - x_hat + self._epsilon) * (1 - mask))
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob(self, sample):
        x_hat = self.forward(sample)
        return self._log_prob(sample, x_hat)

    @torch.no_grad()
    def _sample(self, n_samples, masks, ordering):
        shape = (n_samples, self._input_dim)
        conditioned_on = -torch.ones(shape)
        ordering = np.argsort(ordering)
        for dim in ordering:
            out = self._forward(conditioned_on, masks)[:, dim]
            out = torch.distributions.Bernoulli(probs=out).sample()
            assert torch.all(conditioned_on[:, dim] < 0)
            conditioned_on[:, dim] = out
        return conditioned_on

    @torch.no_grad()
    def sample(self, n_samples: int):
        masks, ordering = self._sample_masks()
        return self._sample(n_samples, masks, ordering)


class NADE(torch.nn.Module):
    """The Neural Autoregressive Distribution Estimator (NADE) model."""

    def __init__(self, input_dim, hidden_dim):
        """Initializes a new NADE instance.
        Args:
            input_dim: The dimension of the input.
            hidden_dim: The dimmension of the hidden layer. NADE only supports one
                hidden layer.
        """
        super().__init__()
        self._input_dim = input_dim
        self._epsilon = 1e-7

        # fmt: off
        self._in_W = torch.nn.Parameter(torch.zeros(hidden_dim, self._input_dim))
        self._in_b = torch.nn.Parameter(torch.zeros(hidden_dim,))
        self._h_W = torch.nn.Parameter(torch.zeros(self._input_dim, hidden_dim))
        self._h_b = torch.nn.Parameter(torch.zeros(self._input_dim,))
        # fmt: on
        torch.nn.init.kaiming_normal_(self._in_W)
        torch.nn.init.kaiming_normal_(self._h_W)

    def _forward(self, x):
        """Computes the forward pass and samples a new output.
        Returns:
            (p_hat, x_hat) where p_hat is the probability distribution over dimensions
            and x_hat is sampled from p_hat.
        """
        # If the input is an image, flatten it during the forward pass.
        original_shape = x.shape
        if len(x.shape) > 2:
            x = x.view(original_shape[0], -1)

        p_hat, x_hat = [], []
        batch_size = 1 if x is None else x.shape[0]
        # Only the bias is used to compute the first hidden unit so we must replicate it
        # to account for the batch size.
        a = self._in_b.expand(batch_size, -1)
        for i in range(self._input_dim):
            h = torch.relu(a)
            p_i = torch.sigmoid(h @ self._h_W[i : i + 1, :].t() + self._h_b[i : i + 1])
            p_hat.append(p_i)

            # Sample 'x' at dimension 'i' if it is not given.
            x_i = x[:, i : i + 1]
            x_i = torch.where(x_i < 0, torch.distributions.Bernoulli(probs=p_i).sample(), x_i)
            x_hat.append(x_i)

            # We do not need to add self._in_b[i:i+1] when computing the other hidden
            # units since it was already added when computing the first hidden unit.
            a = a + x_i @ self._in_W[:, i : i + 1].t()
        if x_hat:
            return (
                torch.cat(p_hat, dim=1).view(original_shape),
                torch.cat(x_hat, dim=1).view(original_shape),
            )
        return []

    def forward(self, x):
        """Computes the forward pass.
        Args:
            x: Either a tensor of vectors with shape (n, input_dim) or images with shape
                (n, 1, h, w) where h * w = input_dim.
        Returns:
            The result of the forward pass.
        """
        if x.dtype == torch.int64:
            x = unpack(x, self._input_dim)
            x = (x + 1) / 2
        return self._forward(x)[0]

    def _log_prob(self, sample, x_hat):
        mask = sample # (sample + 1) / 2
        log_prob = (torch.log(x_hat + self._epsilon) * mask +
                    torch.log(1 - x_hat + self._epsilon) * (1 - mask))
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob(self, x):
        if x.dtype == torch.int64:
            x = unpack(x, self._input_dim)
            x = (x + 1) / 2
        x_hat = self.forward(x)
        return self._log_prob(x, x_hat)

    @torch.no_grad()
    def _sample(self, n_samples):
        shape = (n_samples, self._input_dim)
        conditioned_on = -torch.ones(shape)
        return self._forward(conditioned_on)[1]

    @torch.no_grad()
    def sample(self, n_samples: int):
        return self._sample(n_samples)





def overlap_fn(a, b):
    a -= torch.max(a)
    b -= torch.max(b)
    a = torch.exp(0.5 * a)
    b = torch.exp(0.5 * b)
    return torch.dot(a, b) / torch.linalg.norm(a) / torch.linalg.norm(b)

def overlap_loss_fn(a, b):
    a = a.squeeze()
    b = b.squeeze()
    a = a - torch.max(a)
    b = b - torch.max(b)
    a = torch.exp(0.5 * a)
    b = torch.exp(0.5 * b)
    a_over_b = a / b
    return torch.sum(a_over_b) / torch.linalg.norm(a_over_b) / torch.sqrt(torch.scalar_tensor(b.size(0)))

def fancy_loss_fn(a, b):
    return torch.sum(torch.nn.functional.softplus(b) * torch.exp((a - b)**2))

def _negative_log_overlap(log_pred_prob, log_target_prob):
    log_pred_prob = log_pred_prob.squeeze()
    log_target_prob = log_target_prob.squeeze()
    # log_target_prob -= torch.logsumexp(log_target_prob, dim=0)
    # return 1 - overlap_loss_fn(log_pred_prob, log_target_prob)
    pred_times_target = torch.logsumexp(0.5 * (log_pred_prob - log_target_prob), dim=0)
    norm_pred = 0.5 * torch.logsumexp(log_pred_prob - log_target_prob, dim=0)
    norm_target = 0.5 * torch.log(torch.scalar_tensor(log_target_prob.size(0)))
    # print(torch.exp(pred_times_target - 0.5 * norm_pred - 0.5 * norm_target))
    return -(pred_times_target - norm_pred - norm_target)

def negative_log_overlap(a, b):
    return _negative_log_overlap(b, a)

def negative_log_overlap_uniform(log_pred_prob, log_target_prob):
    log_pred_prob = log_pred_prob.squeeze()
    log_target_prob = log_target_prob.squeeze()
    # return 1 - overlap_fn(log_pred_prob, log_target_prob)
    pred_times_target = torch.logsumexp(0.5 * (log_pred_prob + log_target_prob), dim=0)
    norm_pred = torch.logsumexp(log_pred_prob, dim=0)
    norm_target = torch.logsumexp(log_target_prob, dim=0)
    # print(torch.exp(pred_times_target - 0.5 * norm_pred - 0.5 * norm_target))
    return -(pred_times_target - 0.5 * (norm_pred + norm_target))


def pretrain_network(network, basis, ground_state):
    states = torch.from_numpy(basis.states.view(np.int64))
    states = nqs.unpack(states, basis.number_spins)
    states = (states + 1) / 2

    target = np.abs(ground_state.squeeze())**2
    target = torch.from_numpy(target)
    target = torch.log(torch.clamp(target, min=1e-10)).float()

    def log_prob_fn(x):
        return network.forward(x).squeeze(dim=1)
    loss_fn = negative_log_overlap_uniform

    logger.info("Overlap: {}", overlap_fn(log_prob_fn(states), target))

    for lr, batch_size in [(1e-2, 64)]:
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        for epoch in range(3000):
            indices = torch.randint(low=0, high=states.size(0), size=(batch_size,))
            x_batch = states[indices]
            y_batch = target[indices]
            predicted = log_prob_fn(x_batch)
            loss = loss_fn(predicted, y_batch)
            if epoch % 100 == 0:
                logger.info("{}: loss = {}", epoch, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info("Overlap: {}", overlap_fn(log_prob_fn(states), target))

    if True:
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        for epoch in range(3000):
            x_batch = states
            y_batch = target
            predicted = log_prob_fn(x_batch)
            loss = loss_fn(predicted, y_batch)
            if epoch % 100 == 0:
                logger.info("{}: loss = {}", epoch, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info("Overlap: {}", overlap_fn(log_prob_fn(states), target))

    return network



if __name__ == "__main__":
    np.random.seed(52339877)
    torch.manual_seed(9218823294)

    import lattice_symmetries as ls
    import nqs_playground as nqs

    basis = ls.SpinBasis(ls.Group([]), number_spins=10, hamming_weight=None)
    basis.build()

    # full_basis = ls.SpinBasis(ls.Group([]), number_spins=10, hamming_weight=None)
    # full_basis.build()
    # full_states = torch.from_numpy(basis.states.view(np.int64))
    # full_states = nqs.unpack(full_states, full_basis.number_spins)

    # fmt: off
    matrix = np.array([[1,  0,  0, 0],
                       [0, -1, -2, 0],
                       [0, -2, -1, 0],
                       [0,  0,  0, 1]])
    # fmt: on
    edges = [(i, (i + 1) % basis.number_spins) for i in range(basis.number_spins)]
    operator = ls.Operator(basis, [ls.Interaction(matrix, edges)])
    E, ground_state = ls.diagonalize(operator)
    logger.info("E = {}", E)
    # logger.info("v₀ = {}", ground_state[:, 0])

    # net = torch.nn.Sequential(
    #         torch.nn.Linear(10, 100),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(100, 1, bias=False)
    # )
    # pretrain_network(net, basis, ground_state)
    # exit()

    states = torch.from_numpy(basis.states.view(np.int64))
    states = nqs.unpack(states, basis.number_spins)
    states = (states + 1) / 2

    target = np.abs(ground_state.squeeze())**2
    target = torch.from_numpy(target)
    target_prob = target # torch.ones(target.size(0)) / target.size(0) # target
    target = torch.log(torch.clamp(target, min=1e-7)).float()
    target -= torch.logsumexp(target, dim=0)


    def target_log_prob_fn(xs):
        if xs.dtype == torch.float32:
            xs = nqs.pack(xs).numpy().view(np.uint64)
        else:
            xs = xs.numpy().view(np.uint64)
        if xs.ndim > 1:
            xs = xs[:, 0]
        i = ls.batched_index(basis, xs).view(np.int64)
        return target[i]

    # logger.info("P(x) = {}", target)

    
    n_masks = 1
    # net = MADE(10, [100], n_masks=n_masks)
    net = NADE(10, 100)
    log_prob_fn = net.log_prob
    # net = torch.nn.Sequential(
    #         torch.nn.Linear(10, 100),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(100, 1, bias=False)
    # )
    # def log_prob_fn(x):
    #     return net.forward(x).squeeze(dim=1)
    loss_fn = negative_log_overlap # torch.nn.MSELoss()


    predicted = torch.zeros(states.size(0))
    for i in range(n_masks):
        predicted += log_prob_fn(states)
    predicted /= n_masks
    loss = loss_fn(predicted, target)
    logger.info("Initial loss: {}", loss)
    logger.info("Overlap: {}", overlap_fn(predicted, target))
    logger.info("negative_log_overlap: {} ?= {}", -torch.log(overlap_fn(predicted, target)), negative_log_overlap_uniform(predicted, target))

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    sampling_options = nqs.SamplingOptions(number_samples=1000)
    for epoch in range(5000):
        # x_batch = net.sample(sampling_options.number_samples)
        x_batch, _, _ = nqs.sample_autoregressive(net, None, sampling_options)
        x_batch = x_batch.view(-1, x_batch.size(-1))
        # indices = torch.multinomial(
        #     target_prob,
        #     num_samples=batch_size,
        #     replacement=True,
        # )
        # x_batch = states[indices]
        y_batch = target_log_prob_fn(x_batch) # target[indices]
        assert torch.all(y_batch == target_log_prob_fn(x_batch))
        predicted = log_prob_fn(x_batch)
        loss = loss_fn(predicted, y_batch)
        if epoch % 100 == 0:
            logger.info("{}: loss = {}", epoch, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predicted = torch.zeros(states.size(0))
    for i in range(n_masks):
        predicted += log_prob_fn(states)
    predicted /= n_masks
    loss = loss_fn(predicted, target)
    logger.info("Final loss: {}", loss)
    logger.info("Overlap: {}", overlap_fn(predicted, target))
