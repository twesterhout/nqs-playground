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
import torch.nn as nn

class Net(nn.Module):
    """
    The Neural Network used to encode the wave function.

    It is basically a function ℝⁿ -> ℝ² where n is the number of spins.
    """

    def __init__(self, n: int):
        # NOTE: Do not change the following two lines.
        super().__init__()
        self._number_spins = n
        # NOTE: Feel free to modify the rest to define your own custom
        # architecture.
        self._dense1 = nn.Linear(n, 6 * n)
        # self._dense2 = nn.Linear(17, 15)
        # self._dense3 = nn.Linear(15, 10)
        # self._dense4 = nn.Linear(10, 15)
        # self._dense5 = nn.Linear(15, 20)
        self._dense6 = nn.Linear(6 * n, 2, bias=False)
        nn.init.normal_(self._dense1.weight, mean=0, std=1e-2)
        nn.init.normal_(self._dense1.bias, mean=0, std=2e-2)
        # nn.init.normal_(self._dense2.weight, std=5e-1)
        # nn.init.normal_(self._dense2.bias, std=1e-1)
        # nn.init.normal_(self._dense3.weight, std=5e-1)
        # nn.init.normal_(self._dense3.bias, std=1e-1)
        # nn.init.normal_(self._dense4.weight, std=1e-1)
        # nn.init.normal_(self._dense4.bias, std=1e-1)
        # nn.init.normal_(self._dense5.weight, std=1e-1)
        # nn.init.normal_(self._dense5.bias, std=1e-1)
        # nn.init.normal_(self._dense6.weight, std=1e-1)

    # NOTE: Feel free to modify the body of this function. Do make sure,
    # however, that it still returns a ``torch.FloatTensor`` of size
    # ``torch.Size([2])``.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward propagation.

        :param torch.Tensor x: Spin configuration as.
        """
        x = torch.tanh(self._dense1(x))
        # x = torch.sigmoid(self._dense2(x))
        # x = torch.tanh(self._dense3(x))
        # x = torch.tanh(self._dense4(x))
        # x = torch.tanh(self._dense5(x))
        x = self._dense6(x)
        # x[0].clamp_(-20, 5)
        # logging.info(x)
        return x

    # NOTE: Leave this function as it is unless you're sure know what you're doing.
    @property
    def number_spins(self) -> int:
        """
        Returns the number of spins the network expects as input.

        This is a property which means that you should write ``x.number_spins``
        rather than ``x.number_spins()``.
        """
        return self._number_spins
