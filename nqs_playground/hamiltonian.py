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

from . import _C_nqs
from .core import _with_file_like, WorthlessConfiguration


class Heisenberg(object):
    """
    Isotropic Heisenberg Hamiltonian on a lattice.
    """

    def __init__(self, edges: List[Tuple[int, int]], coupling: complex = 1.0):
        """
        Initialises the Hamiltonian given a list of edges.
        """
        self._graph = edges
        self._coupling = coupling
        smallest = min(map(min, edges))
        largest = max(map(max, edges))
        if smallest != 0:
            ValueError(
                "Invalid graph: Counting from 0, but the minimal index "
                "present is {}.".format(smallest)
            )
        self._number_spins = largest + 1

    def __call__(self, state, cutoff=5.5) -> np.complex64:
        """
        Calculates local energy in the given state.
        """
        spin = state.spin

        def log_quot_wf(flips: List[int]) -> complex:
            state.spin[flips] *= -1
            log_wf = state.machine.log_wf(state.spin)
            state.spin[flips] *= -1
            return log_wf - log_quot_wf.log_wf_old

        log_quot_wf.log_wf_old = state.machine.log_wf(state.spin)

        energy = 0
        for (i, j) in self._graph:
            if spin[i] == spin[j]:
                energy += 1
            else:
                assert spin[i] == -spin[j]
                x = log_quot_wf([i, j])
                if cutoff is not None and x.real > cutoff:
                    raise WorthlessConfiguration([i, j])
                energy += -1 + 2 * cmath.exp(x)
        energy = np.complex64(energy)
        return energy if not cmath.isinf(energy) else np.complex64(1e38)

    def to_cxx(self) -> _C_nqs.Heisenberg:
        return _C_nqs.Heisenberg(edges=self._graph, coupling=self._coupling)

    @property
    def number_spins(self) -> int:
        """
        :return: number of spins in the system.
        """
        return self._number_spins

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return self._graph

    @property
    def coupling(self) -> complex:
        return self._coupling


def _read_hamiltonian(stream):
    specs = []
    for (coupling, edges) in map(
        lambda x: x.strip().split(maxsplit=1),
        filter(lambda x: not x.startswith(b"#"), stream),
    ):
        coupling = float(coupling)
        # TODO: Parse the edges properly, it's not that difficult...
        edges = eval(edges)
        specs.append((coupling, edges))
    # TODO: Generalise Heisenberg to support multiple graphs with different
    # couplings
    if len(specs) != 1:
        raise NotImplementedError("Multiple couplings are not yet supported.")
    coupling, edges = specs[0]
    return Heisenberg(edges, coupling)


def read_hamiltonian(stream) -> Heisenberg:
    """
    Reads the Hamiltonian from ``stream``. ``stream`` could be either a
    file-like object or a ``str`` file name.
    """
    return _with_file_like(stream, "rb", _read_hamiltonian)
