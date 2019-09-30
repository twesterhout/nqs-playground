#!/usr/bin/env python3

# Copyright Tom Westerhout (c) 2019
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

import cmath
from typing import List, Tuple
import numpy as np

from .core import _C, with_file_like

__all__ = ["Heisenberg", "read_hamiltonian"]


class Heisenberg(object):
    """
    Isotropic Heisenberg Hamiltonian on a lattice.
    """

    def __init__(self, specs: List[Tuple[float, int, int]]):
        """
        Initialises the Hamiltonian given a list of edges.
        """
        self._specs = specs
        smallest = min(map(lambda t: min(t[1:]), specs))
        largest = max(map(lambda t: max(t[1:]), specs))
        if smallest != 0:
            ValueError(
                "Invalid graph: Counting from 0, but the minimal index "
                "present is {}.".format(smallest)
            )
        self._number_spins = largest + 1

    def to_cxx(self) -> _C.Heisenberg:
        return _C.Heisenberg(self._specs)

    @property
    def number_spins(self) -> int:
        """
        :return: number of spins in the system.
        """
        return self._number_spins

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return [(i, j) for _, i, j in self._specs]


def _read_hamiltonian(stream):
    specs = []
    for (coupling, edges) in map(
        lambda x: x.strip().split(maxsplit=1),
        filter(lambda x: not x.startswith(b"#"), stream),
    ):
        coupling = float(coupling)
        # TODO: Parse the edges properly, it's not that difficult...
        edges = eval(edges)
        for i, j in edges:
            specs.append((coupling, i, j))
    return Heisenberg(specs)


def read_hamiltonian(stream) -> Heisenberg:
    """
    Reads the Hamiltonian from ``stream``. ``stream`` could be either a
    file-like object or a ``str`` file name.
    """
    return with_file_like(stream, "rb", _read_hamiltonian)
