from math import sqrt
import pathlib
import time
import numpy as np
import torch
from nqs_playground import *
from nqs_playground.core import Unpack
from nqs_playground._tabu_sampler import _sample_using_zanella
from nqs_playground.monte_carlo import _log_amplitudes_to_probabilities

np.random.seed(1183472)
torch.manual_seed(14346284)

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()

@torch.no_grad()
def histogram(spins, basis):
    if spins.dim() == 2:
        assert spins.size(1) == 8
        spins = spins[:, 0]
    spins = spins.cpu()

    r = torch.zeros(basis.number_states, dtype=torch.int64)
    spins, counts = torch.unique(spins, sorted=True, return_counts=True)
    r[basis.index(spins)] += counts
    return r

@torch.no_grad()
def are_close_l1(n, basis, sample, exact, eps, sweep_size=None, device=None):
    exact, order = torch.sort(exact)
    s = np.searchsorted(torch.cumsum(exact, dim=0).numpy(), eps / 8.0)
    ms = np.random.poisson(n, size=16)
    options = SamplingOptions(number_chains=len(ms), number_samples=max(ms),
        sweep_size=sweep_size, device=None)
    qs, _ = sample(options)
    qs = [histogram(qs[:m, i], basis)[order] for i, m in enumerate(ms)]

    def analyze(x, k):
        v = ((x - k * exact)**2 - x) * exact**(-2/3)
        w = exact**(2/3)
        cond1 = torch.sum(v[s:-1]) > 4 * k * torch.sum(w[s:-1])**(1/2)
        cond2 = torch.sum(x[:s]) > 3 / 16 * eps * k
        return not (cond1 or cond2)

    return [analyze(x, k) for x, k in zip(qs, ms)]


def calculate_exact_probabilities(basis, log_ψ, device="cpu"):
    xs = torch.from_numpy(basis.states.view(np.int64)).to(device)
    ys = forward_with_batches(log_ψ, xs, batch_size=8192).squeeze()
    return _log_amplitudes_to_probabilities(ys)


def test_simple():
    basis = SpinBasis([], number_spins=10, hamming_weight=5)
    basis.build()

    log_ψ = torch.nn.Sequential(
        Unpack(basis.number_spins),
        torch.nn.Linear(basis.number_spins, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )

    exact = calculate_exact_probabilities(basis, log_ψ)

    def sample_exact(options):
        return sample_some(log_ψ, basis, options, mode="monte_carlo")[:2]

    def sample_metropolis(options):
        return sample_some(log_ψ, basis, options, mode="monte_carlo")[:2]

    def sample_zanella(options):
        return _sample_using_zanella(log_ψ, basis, options, thin_rate=1e-1)

    for f in [sample_exact, sample_metropolis, sample_zanella]:
        r = are_close_l1(1000, basis, f, exact, eps=1e-5)
        print(r)
        assert sum(r) > len(r) / 2


def test_tricky():
    L = 20
    x = np.arange(L, dtype=np.int32)
    T = (x + 1) % L
    P = L - 1 - x
    G = make_group([Symmetry(T, sector=0), Symmetry(P, sector=0)])
    basis = SpinBasis(G, number_spins=L, hamming_weight=L // 2)
    basis.build()

    log_ψ = torch.nn.Sequential(
        Unpack(L),
        torch.nn.Linear(L, L // 2),
        torch.nn.Tanh(),
        torch.nn.Linear(L // 2, 1, bias=False)
    )
    exact = calculate_exact_probabilities(basis, log_ψ)

    def sample_exact(options):
        return sample_some(log_ψ, basis, options, mode="exact")[:2]

    def sample_metropolis(options):
        return sample_some(log_ψ, basis, options, mode="monte_carlo")[:2]

    def sample_zanella(options):
        return _sample_using_zanella(log_ψ, basis, options, thin_rate=2e-2)

    for f in [sample_exact, sample_metropolis, sample_zanella]:
        r = are_close_l1(2000, basis, f, exact, eps=1e-5)
        print(r)
        assert sum(r) > len(r) / 2


def test_5x5_with_symmetries():
    import pickle
    basis = with_file_like(CURRENT_DIR / "basis_5x5.pickle", "rb", pickle.load)
    log_ψ = torch.jit.load(str(CURRENT_DIR / "difficult_to_sample_5x5.pt"))
    exact = calculate_exact_probabilities(basis, log_ψ)

    def sample_exact(options):
        return sample_some(log_ψ, basis, options, mode="exact")[:2]

    def sample_metropolis(options):
        return sample_some(log_ψ, basis, options, mode="monte_carlo")[:2]

    def sample_zanella(options):
        return _sample_using_zanella(log_ψ, basis, options, thin_rate=1e0)

    for f in [ sample_exact,
               # sample_metropolis,
               sample_zanella,
             ]:
        r = are_close_l1(10000, basis, f, exact, eps=1e-2, sweep_size=100)
        print(r)
        assert sum(r) > len(r) / 2
    return

def profile_5x5_with_symmetries():
    import pickle
    import pprofile
    basis = with_file_like(CURRENT_DIR / "basis_5x5.pickle", "rb", pickle.load)
    log_ψ = torch.jit.load(str(CURRENT_DIR / "difficult_to_sample_5x5.pt"))
    options = SamplingOptions(number_chains=16, number_samples=100,
        sweep_size=100, device=None)
    prof = pprofile.Profile()
    with prof():
        # sample_some(log_ψ, basis, options, mode="exact")
        # sample_some(log_ψ, basis, options, mode="monte_carlo")
        _sample_using_zanella(log_ψ, basis, options, thin_rate=1e0)
    prof.print_stats()

# test_simple()
# test_tricky()
# test_zanella() 
test_5x5_with_symmetries()
# profile_5x5_with_symmetries()
