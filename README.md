PyTorch-based implementation of SR and SWO for NQS.


**Contents**
- [Installation](#installation)
  - [Conda](#conda)
  - [Building from source](#building-from-source)


## Installation


## Conda

> **WARNING:** The version available on Conda is currently out of date.
> Please, build from source for the latest features.

The simplest way to get started using `nqs_playground` package is to install it
using [Conda](https://docs.conda.io/en/latest/):
```sh
conda install -c twesterhout nqs_playground
```


## Building from source

### CPU-only version

If you do not have access or do not wish to use a GPU you can use cpu-only
version PyToch and nqs_playground. For this, first clone the repository:

```sh
git clone https://github.com/twesterhout/nqs-playground.git
```

Now just run [`build_locally_cpu.sh`](./build_locally_cpu.sh):

```sh
./build_locally_cpu.sh
```

### Full version

TODO
