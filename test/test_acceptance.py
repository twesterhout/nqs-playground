from math import pi as π
from math import sqrt
import numpy as np
import torch
from torch import Tensor

from nqs_playground import *
from nqs_playground.core import forward_with_batches


def make_distribution(basis):
    n = basis.number_states
    xs = torch.arange(0, n, dtype=torch.float64)
    indices = torch.randperm(n)

    gauss = lambda x, μ, σ: torch.exp(- 0.5 * ((x - μ) / σ)**2)

    def _make_p():
        ys = gauss(xs, n / 2, n / 20)
        ys /= torch.sum(ys)
        def distribution(states: Tensor) -> Tensor:
            if len(states.shape) == 2:
                states = states[:, 0]
            return ys[indices[basis.index(states)]].squeeze()
        return distribution

    def _make_E():
        ys = 6 * gauss(xs, (0.5 - 0.05) * n, n / 16) + gauss(xs, (0.5 + 0.15) * n, n / 30)
        def observable(states: Tensor) -> Tensor:
            if len(states.shape) == 2:
                states = states[:, 0]
            return ys[indices[basis.index(states)]].squeeze()
        return observable

    return _make_p(), _make_E()

def simple():
    basis = SpinBasis([], 28, 14)
    basis.build()
    all_states = torch.from_numpy(basis.states.view(np.int64))
    print(basis.number_states)
    p, E = make_distribution(basis)
    print(torch.sum(p(all_states)))
    print(torch.sum(E(all_states)))
    exact = torch.dot(p(all_states), E(all_states))
    print(exact, torch.dot(p(all_states), (E(all_states) - exact)**2))

    with torch.no_grad():
        i = np.random.choice(basis.number_states, size=basis.number_spins * 2000, p=p(all_states).numpy(), replace=True)
        print(torch.mean(E(all_states[i])), torch.var(E(all_states[i])))

    options = SamplingOptions(
        number_chains=basis.number_spins,
        number_samples=1000,
        device="cpu"
    )
    x, y, r = sample_some(lambda x: 0.5 * torch.log(p(x)), basis, options,
            mode="monte_carlo")
    print(torch.mean(E(x.view(-1, 8))), torch.var(E(x.view(-1, 8))))
    print(r)

def make_basis():
    Lx = 6
    Ly = 6
    NUMBER_SPINS  = Lx * Ly
    sites = np.arange(NUMBER_SPINS) # site labels [0,1,2,....]
    x = sites%Lx # x positions for sites
    y = sites//Lx # y positions for sites
    #
    T_x = (x+1)%Lx + Lx*y # translation along x-direction
    T_y = x +Lx*((y+1)%Ly) # translation along y-direction
    #
    P_x = x + Lx*(Ly-y-1) # reflection about x-axis
    P_y = (Lx-x-1) + Lx*y # reflection about y-axis
    #
    symmetry_group = make_group(
        [
            # Translation
            Symmetry(
                list(T_x), sector= 0 #NUMBER_SPINS // 2
            ),
            Symmetry(
                list(T_y), sector= 0 #NUMBER_SPINS // 2
            ),
            # Reflections
            Symmetry(
                list(P_x), sector= 0 #NUMBER_SPINS // 2
            ),
            Symmetry(
                list(P_y), sector= 0 #NUMBER_SPINS // 2
            ),
        ]
    )
    return SpinBasis(symmetry_group, number_spins=NUMBER_SPINS,
            hamming_weight=NUMBER_SPINS // 2)


def andrey():
    import pickle
    with open("basis_6x6.pickle", "rb") as f:
        basis = pickle.load(f)
    # with open("basis_6x6.pickle", "wb") as f:
    #     pickle.dump(basis, f)
        
    log_ψ = torch.jit.script(torch.nn.Sequential(
        torch.nn.Linear(basis.number_spins, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1, bias=False),
    ))
    log_ψ.load_state_dict(torch.load("/home/twesterh/mc/Fail2/amplitude_weights.pt"))
    options = SamplingOptions(
        number_chains=basis.number_spins,
        number_samples=1000,
        device="cpu"
    )

    ys = forward_with_batches(
            lambda x: log_ψ(unpack(x, basis.number_spins)),
            torch.from_numpy(basis.states.view(np.int64)),
            batch_size=81920).squeeze()
    ys = torch.sort(ys)[0]
    print(ys[:10])
    print(ys[-10:])
    print(torch.histc(ys).detach().numpy().tolist())

    x, y, r = sample_some(lambda x: log_ψ(unpack(x, basis.number_spins)), basis, options, mode="monte_carlo")
    print(r)


if __name__ == '__main__':
    andrey()
