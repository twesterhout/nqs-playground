import json
import lattice_symmetries as ls
from loguru import logger
import nqs_playground as nqs
import numpy as np
import os
import time
import torch
from scipy import stats
import subprocess
import yaml


def get_processor_name():
    result = subprocess.run(["lscpu", "-J"], check=False, capture_output=True)
    if result.returncode != 0:
        logger.warn(
            "Failed to get processor name: {} returned error code {}: {}",
            result.args,
            result.returncode,
            result.stderr,
        )
        return None
    for obj in json.loads(result.stdout)["lscpu"]:
        if obj["field"].startswith("Model name"):
            return obj["data"]


def get_gpu_name():
    result = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True)
    if result.returncode != 0:
        logger.warn(
            "Failed to get processor name: {} returned error code {}: {}",
            result.args,
            result.returncode,
            result.stderr,
        )
        return None
    for line in filter(
        lambda s: s.startswith("GPU 0:"), result.stdout.decode("utf-8").split("\n")
    ):
        line = line.strip("GPU 0:").strip(" ")
        return line


def make_basis(number_spins: int) -> ls.SpinBasis:
    parity = (number_spins - 1) - np.arange(number_spins, dtype=np.int32)
    translation = (np.arange(number_spins, dtype=np.int32) + 1) % number_spins
    return ls.SpinBasis(
        ls.Group([ls.Symmetry(parity, sector=0), ls.Symmetry(translation, sector=0)]),
        number_spins=number_spins,
        hamming_weight=number_spins // 2,
        spin_inversion=1 if number_spins % 2 == 0 else None,
    )


def make_network(number_spins: int) -> torch.nn.Module:
    module = torch.nn.Sequential(
        nqs.Unpack(number_spins),
        torch.nn.Linear(number_spins, 64),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(64, 1, bias=False),
    )
    module = torch.jit.script(module)
    return module


def profile_one(number_spins: int, device: torch.device, number_chains: int = 10):
    basis = make_basis(number_spins)
    log_amplitude_fn = make_network(basis.number_spins)
    log_amplitude_fn.to(device=device)
    options = nqs.SamplingOptions(
        number_chains=number_chains,
        number_samples=1,
        sweep_size=1,
        number_discarded=10,
        device=device,
    )
    # chain_lengths = [1, 400, 800, 1200, 1400, 1600, 1800, 2000, 2200, 2400]
    chain_lengths = [1, 800, 1600, 2400]
    results = {}
    for method in ["zanella"]:
        line = []
        logger.debug("Measuring '{}'...", method)
        for number_samples in chain_lengths:
            t1 = time.time()
            _ = nqs.sample_some(
                log_amplitude_fn,
                basis,
                options._replace(number_samples=number_samples, mode=method),
            )
            t2 = time.time()
            line.append(t2 - t1)

        regression = stats.linregress(chain_lengths[1:], line[1:])
        results[method] = {
            "raw": line,
            "slope": regression.slope,
            "slope_err": regression.stderr,
            "intercept": regression.intercept,
            "intercept_err": regression.intercept_stderr,
        }
    return results


def profile_all(number_chains: int = 1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Running on '{}'...", device)

    header = "Date: {}\n".format(time.asctime())
    cpu = get_processor_name()
    if cpu is not None:
        header += "CPU: {}\n".format(cpu)
    gpu = get_gpu_name()
    if gpu is not None:
        header += "GPU: {}\n".format(gpu)
    header += "number_spins\tmetropolis\tmetropolis_error\tzanella\tzanella_error"

    results = []
    for system_size in [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192]:
        r = profile_one(system_size, device)
        results.append(
            (
                system_size,
                r["metropolis"]["slope"],
                r["metropolis"]["slope_err"],
                r["zanella"]["slope"],
                r["zanella"]["slope_err"],
            )
        )

    results = np.asarray(results, dtype=np.float64)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    np.savetxt(
        os.path.join(script_dir, "data", "timing_chain_{}.dat".format(number_chains)),
        results,
        header=header,
    )


def main():
    print(profile_one(65, "cuda", 2))
    # profile_all()


if __name__ == "__main__":
    main()
