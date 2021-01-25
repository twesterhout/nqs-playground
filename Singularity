Bootstrap: docker
From: nvidia/cuda:11.0-devel-ubuntu20.04

%post -c /bin/bash
    set -e
    export LC_ALL=C
    export PYTHON_VERSION=3.8
    export PYTORCH_VERSION=1.7
    export LATTICE_SYMMETRIES_VERSION=0.4.0

    apt-get update
    apt-get install -y --no-install-recommends \
         ca-certificates \
         pkg-config \
         curl
    apt-get clean
    apt-get autoclean
    rm -rf /var/lib/apt/lists/*

    mkdir -p /workdir
    cd /workdir

    curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p /opt/conda
    rm miniconda.sh

    . /opt/conda/etc/profile.d/conda.sh
    # We only match major and minor versions of cudatoolkit
    conda install -y python=$PYTHON_VERSION cudatoolkit=${CUDA_VERSION%.*} \
        anaconda-client conda-build conda-verify \
        gcc_linux-64 gxx_linux-64 cmake ninja
    conda install -y -c pytorch pytorch=$PYTORCH_VERSION
    conda install -y -c twesterhout lattice-symmetries=$LATTICE_SYMMETRIES_VERSION
    conda clean -ya


%environment
    export LC_ALL=C
    export TERM=xterm-256color


%runscript
    exec /bin/bash
