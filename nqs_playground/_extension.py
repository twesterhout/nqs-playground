import os
import sys
import torch
from torch.utils.cpp_extension import load as _load

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

lib = _load(
    "_nqs_playground_cpp",
    [os.path.join(PACKAGE_DIR, "cbits", "zanella.cpp"),
     os.path.join(PACKAGE_DIR, "cbits", "accumulator.cpp"),
     os.path.join(PACKAGE_DIR, "cbits", "init.cpp")
     ],
    extra_cflags=[
        "-fvisibility=hidden",
        "-std=c++17",
        "-g",
        # "-march=native",
        # "-O3",
        # "-ftree-vectorize",
        "-fopenmp",
        "-Wall",
        "-Wextra"
    ],
    extra_include_paths=[os.path.join(sys.prefix, "include")],
    extra_ldflags=["-fopenmp", "-L" + os.path.join(sys.prefix, "lib"), "-llattice_symmetries"],
    verbose=True
)

