# Copyright Tom Westerhout (c) 2018
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
#     * Neither the name of Tom Westerhout nor the names of other
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from .functional import logcosh


class Rbm(torch.nn.Module):
    r"""
    Complex Restricted Boltzmann Machine (RBM).
    """

    def __init__(self, n: int, alpha: float = 2):
        r"""
        Constructs a new RBM given the number of spins ``n``
        """
        if n < 1:
            raise ValueError("Number of spins must be positive, but got".format(n))
        super().__init__()
        self._number_spins = n

        # Extra factor 2 comes from the fact that a complex number is
        # equivalent to two real numbers.
        number_hidden = round(2 * alpha * n)
        if number_hidden % 2 != 0:
            raise ValueError(
                "Invalid α: number of hidden spins must be even, but got {}".format(
                    number_hidden
                )
            )
        self._dense = torch.nn.Linear(n, number_hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Runs the forward propagation.
        """
        return (
            logcosh(self._dense(x))
            .view(-1, self._dense.out_features // 2, 2)
            .sum(dim=1)
            .squeeze()
        )

    @property
    def number_spins(self) -> int:
        r"""
        Returns the number of spins the network expects as input.
        """
        return self._number_spins
