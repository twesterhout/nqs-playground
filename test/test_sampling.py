import lattice_symmetries as ls
from loguru import logger
import nqs_playground as nqs
import numpy as np
from scipy.stats import power_divergence, combine_pvalues, chisquare
import torch
from torch import Tensor


def prepare_even_chain(n: int):
    assert n > 2 and n % 2 == 0
    anti_symmetric = (n // 2) % 2 == 1

    # Constructing a symmetrized basis
    x = np.arange(n, dtype=np.int32)
    T = (x + 1) % n
    P = n - 1 - x
    G = ls.Group(
        [
            ls.Symmetry(T, sector=n // 2 if anti_symmetric else 0),
            ls.Symmetry(P, sector=1 if anti_symmetric else 0),
        ]
    )
    symm_basis = ls.SpinBasis(
        G, number_spins=n, hamming_weight=n // 2, spin_inversion=-1 if anti_symmetric else 1
    )
    symm_basis.build()

    # Constructing full basis
    full_basis = ls.SpinBasis(ls.Group([]), number_spins=n, hamming_weight=n // 2)
    full_basis.build()

    # Diagonalize the Hamiltonian in two bases
    terms = [nqs.heisenberg_interaction([(i, (i + 1) % n) for i in range(n)])]
    symm_H = ls.Operator(symm_basis, terms)
    symm_E, symm_ψ = ls.diagonalize(symm_H)
    symm_ψ = symm_ψ.squeeze()

    full_H = ls.Operator(full_basis, terms)
    full_E, full_ψ = ls.diagonalize(full_H)
    full_ψ = full_ψ.squeeze()

    # Check that the states are the same
    assert np.isclose(symm_E, full_E)

    def make_log_amplitude(basis, ground_state):
        log_ground_state = np.log(np.abs(ground_state))

        @torch.no_grad()
        def log_amplitude_fn(spins: torch.Tensor):
            assert spins.dim() == 1 or (spins.dim() == 2 and spins.size(1) == 8)
            if spins.dim() > 1:
                spins = spins[:, 0]
            device = spins.device
            spins = spins.cpu().numpy().view(np.uint64)
            indices = basis.batched_index(spins)
            out = torch.from_numpy(log_ground_state[indices]).view(-1, 1)
            return out.to(device)

        return log_amplitude_fn

    return {
        "full": {
            "basis": full_basis,
            "hamiltonian": full_H,
            "exact_prob": torch.from_numpy(np.abs(full_ψ) ** 2),
            "log_amplitude": make_log_amplitude(full_basis, full_ψ),
        },
        "symm": {
            "basis": symm_basis,
            "hamiltonian": symm_H,
            "exact_prob": torch.from_numpy(np.abs(symm_ψ) ** 2),
            "log_amplitude": make_log_amplitude(symm_basis, symm_ψ),
        },
    }

def test_compiles():
    ε = 1e-4
    for n in [4]:
        info = prepare_even_chain(n)
        for mode in ["full", "exact", "metropolis", "zanella"]:
            sampling_options = nqs.SamplingOptions(number_samples=10, number_chains=4, sweep_size=n, mode=mode)
            for variant in ["full", "symm"]:
                data = info[variant]
                _ = nqs.sample_some(data["log_amplitude"], data["basis"], sampling_options)


def test_via_l1_closeness():
    ε = 1e-4
    for n in [4, 6, 8]:
        info = prepare_even_chain(n)
        for mode in ["metropolis", "zanella"]:
            sampling_options = nqs.SamplingOptions(
                number_samples=1, number_chains=32, sweep_size=n
            )
            for variant in ["full", "symm"]:
                data = info[variant]
                r = nqs.are_close_l1(
                    100000,
                    data["basis"],
                    lambda o: nqs.sample_some(data["log_amplitude"], data["basis"], o, mode),
                    data["exact_prob"],
                    ε,
                    sampling_options,
                )
                logger.info("Results: {}", r)
                assert sum(r) > len(r) / 2


def test_via_chisquare():
    ε = 1e-4
    for n in [6, 4]:
        info = prepare_even_chain(n)
        options = nqs.SamplingOptions(number_samples=100000, number_chains=32, sweep_size=n)
        for mode in ["metropolis", "zanella"]:
            for variant in ["full", "symm"]:
                data = info[variant]
                expected_counts = data["exact_prob"].numpy() * options.number_samples
                expected_counts = np.round(expected_counts).astype(np.int64)

                spins, log_prob, extra = nqs.sample_some(
                    data["log_amplitude"], data["basis"], options, mode
                )
                if log_prob is not None:
                    logger.info(
                        "Autocorrelation time is {}", nqs.integrated_autocorr_time(log_prob)
                    )
                if extra is not None:
                    logger.info("Additional info from the sampler: {}", extra)

                logger.info("Computing histograms...")
                sampled_counts = [
                    nqs.sampled_histogram(spins[:, i, :], data["basis"]).numpy()
                    for i in range(spins.size(1))
                ]

                p_values = [
                    chisquare(counts, f_exp=expected_counts)[1]
                    for counts in sampled_counts
                    if np.all(counts > 5)
                ]
                logger.info("p-values: {}", p_values)
                _, combined_p_value = combine_pvalues(p_values, method="fisher")
                logger.info("combined p-value: {}", combined_p_value)
                assert combined_p_value > ε

def test_zanella_graphs():
    ε = 1e-4
    info = prepare_even_chain(6)
    graphs = [
        # None,
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
    ]
    for g in graphs:
        sampling_options = nqs.SamplingOptions(number_samples=1, number_chains=32, sweep_size=6, other={"edges": g})
        for variant in ["full", "symm"]:
            data = info[variant]
            r = nqs.are_close_l1(
                1000,
                data["basis"],
                lambda o: nqs.sample_some(data["log_amplitude"], data["basis"], o, mode="zanella"),
                data["exact_prob"],
                ε,
                sampling_options,
            )
            logger.info("Results: {}", r)
            assert sum(r) > len(r) / 2
