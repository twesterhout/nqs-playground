#!/usr/bin/env python3

# Copyright Tom Westerhout (c) 2020
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

__all__ = ["diagonalise"]

import numpy as np
from slepc4py import SLEPc
from petsc4py import PETSc


def _construct_operator(hamiltonian):
    class _Context:
        def __init__(self, h):
            self._h = h

        def mult(self, _, x, y):
            x = x.getArray(readonly=True)
            y = y.getArray(readonly=False)
            self._h(x, y)

    n = hamiltonian.basis.number_states
    operator = PETSc.Mat().createPython([n, n], _Context(hamiltonian))
    operator.setOption(PETSc.Mat.Option.SYMMETRIC if hamiltonian.is_real
        else PETSc.Mat.Option.HERMITIAN, True)
    operator.setUp()
    return operator


def _construct_solver(operator, k, ncv=None):
    solver = SLEPc.EPS().create()
    solver.setOperators(operator)
    if ncv is None:
        ncv = PETSc.DECIDE
    solver.setDimensions(k, ncv)
    solver.setProblemType(SLEPc.EPS.ProblemType.HEP)
    return solver


def diagonalise(hamiltonian, k=1, **kwargs):
    hamiltonian.basis.build()
    operator = _construct_operator(hamiltonian)
    solver = _construct_solver(operator, k, **kwargs)

    print("Information :: Diagonalising the matrix using SLEPc...")
    solver.solve()

    converged = solver.getConverged()
    if converged < k:
        print("Warning :: {} eigenpairs requested, but only {} converged".format(k, converged))

    energies = np.empty(k, dtype=PETSc.ScalarType)
    states = np.empty((hamiltonian.basis.number_states, k),
        dtype=PETSc.ScalarType if hamiltonian.is_real else PETSc.ComplexType)
    _x_real, _x_imag = operator.getVecs()
    for i in range(k):
        λ = solver.getEigenpair(i, _x_real, _x_imag)
        assert λ.imag == 0
        energies[i] = λ.real
        states.real[:, i] = _x_real.getArray(readonly=True)
        if hamiltonian.is_real:
            assert np.allclose(_x_imag.getArray(readonly=True), 0)
        else:
            states.imag[:, i] = _x_imag.getArray(readonly=True)

    return energies, states
