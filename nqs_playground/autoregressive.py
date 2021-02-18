from functools import reduce
from operator import mul
import torch

from . import *


class CausalConv2d(torch.nn.Conv2d):
    def __init__(self, mask_center: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        i, o, h, w = self.weight.shape
        mask = torch.zeros((i, o, h, w))
        mask.data[:, :, : h // 2, :] = 1
        mask.data[:, :, h // 2, : w // 2 + int(not mask_center)] = 1
        self.register_buffer("mask", mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PixelCNN(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.layers = torch.nn.Sequential(
            CausalConv2d(
                mask_center=True, in_channels=1, out_channels=16, kernel_size=7, padding=3
            ),
            torch.nn.ReLU(),
            CausalConv2d(
                mask_center=False, in_channels=16, out_channels=16, kernel_size=3, padding=1
            ),
            # torch.nn.ReLU(),
            # CausalConv2d(
            #     mask_center=False, in_channels=16, out_channels=16, kernel_size=3, padding=1
            # ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1),
            torch.nn.Sigmoid(),
        )
        self._epsilon = 0 # 1e-7

    def _forward(self, x):
        return self.layers(x)

    def forward(self, x):
        assert False
        x = unpack(x, reduce(mul, self.input_shape)).view(x.size(0), 1, *self.input_shape)
        x = (x + 1) / 2
        return self._forward(x)

    def _log_prob(self, x, x_hat):
        mask = x.byte()
        # print(mask)
        log_prob = torch.where(
            mask, torch.log(self._epsilon + x_hat), torch.log1p(self._epsilon - x_hat)
        )
        # print(torch.log(self._epsilon + x_hat))
        # print(torch.log(1 + self._epsilon - x_hat))
        # print(log_prob)
        # print(x_hat)

        log_prob = log_prob.view(log_prob.size(0), -1).sum(dim=1)
        return log_prob

    def log_prob(self, x):
        x = unpack(x, reduce(mul, self.input_shape)).view(x.size(0), 1, *self.input_shape)
        x = (x + 1) / 2
        x_hat = self._forward(x)
        return self._log_prob(x, x_hat)

    @torch.no_grad()
    def sample(self, num_samples):
        shape = (num_samples, 1, *self.input_shape)
        sample = -torch.ones(shape)
        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[1]):
                out = self._forward(sample)[:, :, i, j]
                out = torch.distributions.Bernoulli(logits=out).sample().view(num_samples, 1)
                sample[:, :, i, j] = torch.where(
                    sample[:, :, i, j] < 0,
                    out,
                    sample[:, :, i, j],
                )
        sample = 2 * sample - 1
        return sample.view(num_samples, -1)
