PyTorch-based implementation of SR and SWO for NQS.


**Contents**  
- [Installation](#installation-and-use)
  - [Conda package](#conda-package)
  - [Building from source](#building-from-source)
- [Notes](#version-semantics)
  - [Hilbert space basis](#hilbert-space-basis)
- [Appendix](#appendix)


## Installation


### Conda package

The simplest way to get started using `nqs_playground` package is to install it
using [Conda](https://docs.conda.io/en/latest/):
```sh
conda install -c twesterhout nqs_playground
```

> **Note:** I am just getting started with Conda packaging system. Currently,
> there is only one version of `nqs_playground` on [Anaconda
> Cloud](https://anaconda.org/twesterhout/nqs_playground) built against Python
> 3.7 and PyTorch 1.3.1 with GPU support. I hope to add more versions in the
> near future.

### Building from source

If you want to contribute to the development you will need to build the package
from source. Luckily, with Conda, it is very simple.

1. Clone the repository:
   ```sh
   git clone https://github.com/twesterhout/nqs-playground.git
   ```
   You do not have to initialise the submodules. CMake will do it automatically
   for you. Just ensure that you are connected to Internet when you first build
   the package.

2. Create a [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):
   ```sh
   conda env create -f devel.yml
   ```
   This will create `nqs_dev` environment using packages listed in `devel.yml`.
   This is a minimal environment necessary to compile and install the package.
   Feel free to use your custom one just ensure that packages listed in
   `devel.yml` are present.

   If you are compiling on a cluster which provides CUDA as a module, now would
   be a good time to load it. I.e. before activating the environment to ensure
   that Conda paths are searched first.

3. Activate the environment:
   ```sh
   conda activate nqs_dev
   ```

4. Compile the C++ extension:
   ```sh
   mkdir build && cd build
   cmake -GNinja -DCMAKE_BUILD_TYPE=Release ..
   # or cmake -GNinja -DCMAKE_BUILD_TYPE=Debug .. if you want the debug version
   # with all asserts etc.
   cmake --build . --target install
   ```
   This should install two files: `libnqs.so` shared library and
   `_C.cpython-...-x86_64-linux-gnu.so` Python extension into the
   `nqs_playground/` directory. You do not have to worry about depencencies --
   they are handled using git submodules and CMake, and it all happens
   automatically. Also, CMake will determine whether you have a CUDA compiler
   on your system and use it to compile GPU kernels. If you do not have CUDA,
   do not despair, the code will compile and work on CPU-only systems as well.

5. Install the Python package:
   ```sh
   python -m pip install --user -e .
   ```
   Alternatively, you can just import `nqs_playground` directly from the project
   root without using `pip install`. This works, because all the necessary
   shared library objects are inside the `nqs_playground/` directory.

And that is it!


## Notes

### Hilbert space basis

The very first thing one does when solving a quantum mechanical problem is
defining the Hilbert space. The way to do it in `nqs_playground` is by using
the `SpinBasis` constructor.

> **Technical note:** `SpinBasis` is not a class, it's a simple function which
> emulates a class. The reason is that under the hood (depending on the number
> of spins in the system) one of the following two classes is used:
> `nqs_playground._C.SmallSpinBasis` or `nqs_playground._C.BigSpinBasis`. These
> two classes have slightly different functionality. E.g. `SmallSpinBasis` can
> return all states and can be used for Exact Diagonalisation. As a user you
> probably do not care about the underlying class, so `SpinBasis` function
> automatically constructs the class most suitable for the problem.

For example, `SpinBasis([], 20)` constructs Hilbert space basis for a system of
20 spins. We can also restrict the Hilbert space to a sector of fixed
magnetisation: `SpinBasis([], number_spins=20, hamming_weight=10)` will contain
states which have zero magnetisation.

**More information** can on constructing the basis can obtained by calling
`help(nqs_playground.SpinBasis)`. `help(nqs_playground.SpinBasis([], 20))` will
give more information specific to `SmallSpinBasis`, and
`help(nqs_playground.SpinBasis([], 200))` -- specific to `BigSpinBasis`.

## Appendix
