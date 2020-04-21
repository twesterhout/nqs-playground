import os
import sys
import numpy as np
from nqs_playground import *

import nqs_playground
from nqs_playground._C import Interaction, Operator

USE_SYMMETRIES = False
if USE_SYMMETRIES:
    FILENAME_PATTERN = "ground_state_symm_1x{}.npy"
else:
    FILENAME_PATTERN = "ground_state_1x{}.npy"
CHECK_STATES = True

__matrix = lambda data: np.array(data, dtype=np.complex128)
# fmt: off
σ_0 = __matrix([ [1, 0]
               , [0, 1] ])
σ_x = __matrix([ [0, 1]
               , [1, 0] ])
σ_y = __matrix([ [0  , 1j]
               , [-1j, 0] ])
σ_z = __matrix([ [-1, 0]
               , [0 , 1] ])
# fmt: on
σ_p = σ_x + 1j * σ_y
σ_m = σ_x - 1j * σ_y


def make_basis(n: int):
    return SpinBasis([], number_spins=n, hamming_weight=n // 2)


def make_hamiltonian(basis):
    if USE_SYMMETRIES:
        raise NotImplementedError()
    n = basis.number_spins
    return Heisenberg([(1.0, i, (i + 1) % n) for i in range(n)], basis)


def make_operator(basis):
    matrix = lambda data: np.array(data, dtype=np.complex128)
    # fmt: off
    σ_x = matrix([ [0, 1]
                 , [1, 0] ])
    σ_y = matrix([ [0 , -1j]
                 , [1j,   0] ])
    σ_z = matrix([ [1,  0]
                 , [0, -1] ])
    # fmt: on
    op = np.kron(σ_x, σ_x) + np.kron(σ_y, σ_y) + np.kron(σ_z, σ_z)
    n = basis.number_spins
    edges = [(i, (i + 1) % n) for i in range(n)]
    return Operator([Interaction(op, edges)], basis)


def check_state(H, y):
    Hy = H(y)
    E = np.dot(y.conj(), Hy)
    close = np.allclose(E * y, Hy)
    return close, E


def make_exact(hamiltonian):
    filename = FILENAME_PATTERN.format(hamiltonian.basis.number_spins)
    if not os.path.exists(filename):
        print("Information :: Diagonalising...")
        energy, ground_state = diagonalise(hamiltonian, k=1)
        ground_state = ground_state.squeeze()
        np.save(filename, ground_state, allow_pickle=False)
        print(energy)
    elif CHECK_STATES:
        print("Information :: Checking...")
        ground_state = np.load(filename)
        close, E = check_state(hamiltonian, ground_state)
        if close:
            print(E)
        else:
            raise ValueError("'{}' contains an invalid eigenstate".format(filename))


def example0_quspin():
    # Example #0 from QuSpin Documentation
    # https://weinbe58.github.io/QuSpin/examples/example0.html
    from quspin.operators import hamiltonian
    from quspin.basis import spin_basis_1d

    L = 10  # system size
    Jxy = np.sqrt(2.0)  # xy interaction
    Jzz = 1.0  # zz interaction
    hz = 1.0 / np.sqrt(3.0)  # z external field

    basis = spin_basis_1d(L, pauli=True)
    J_zz = [[Jzz, i, i + 1] for i in range(L - 1)]  # OBC
    J_xy = [[Jxy / 2.0, i, i + 1] for i in range(L - 1)]  # OBC
    h_z = [[hz, i] for i in range(L)]
    static = [["+-", J_xy], ["-+", J_xy], ["zz", J_zz], ["z", h_z]]
    dynamic = []
    H_xxz = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
    E, V = H_xxz.eigh()
    # Returns the first 10 eigenstates. We reverse the order of states since
    # QuSpin stores basis.states in decreasing order and nqs_playground stores
    # them in increasing order.
    return E[:10], V[::-1, :10]


def example0_nqs_playground():
    L = 10  # system size
    Jxy = np.sqrt(2.0)  # xy interaction
    Jzz = 1.0  # zz interaction
    hz = 1.0 / np.sqrt(3.0)  # z external field

    basis = SpinBasis([], L)
    basis.build()

    # XXZ part
    xxz_op = Jxy / 2 * (np.kron(σ_p, σ_m) + np.kron(σ_m, σ_p)) + Jzz * np.kron(σ_z, σ_z)
    xxz_edges = [(i, i + 1) for i in range(L - 1)]
    # Field part. This is a trick. Since nqs_playground only supports 2-local
    # operators, we add an identity matrix.
    field_op = hz * np.kron(σ_z, σ_0)
    # we want the first index to run from 0 to L - 1. The second index is not
    # important as long as it's different from the first.
    field_edges = [(i, i + 1) for i in range(L - 1)] + [(L - 1, L - 2)]

    hamiltonian = Operator(
        [Interaction(xxz_op, xxz_edges), Interaction(field_op, field_edges)], basis
    )
    energy, ground_state = diagonalise(hamiltonian, k=10)
    return energy, ground_state


def example0_compare():
    energy_quspin, states_quspin = example0_quspin()
    energy_nqs, states_nqs = example0_nqs_playground()
    assert np.allclose(energy_quspin, energy_nqs)
    for i in range(states_quspin.shape[1]):
        assert np.allclose(states_quspin[:, i], states_nqs[:, i]) or np.allclose(
            states_quspin[:, i], -states_nqs[:, i]
        )


def run(n):
    print("Information :: Creating basis...")
    basis = make_basis(n)
    print("Information :: Building list of representatives...")
    basis.build()
    print("Information :: Creating Hamiltonian...")
    # operator = make_hamiltonian(basis)
    operator = make_operator(basis)
    make_exact(operator)


def main():
    for n in [10, 12, 14, 16]:
        run(n)


if __name__ == "__main__":
    main()
