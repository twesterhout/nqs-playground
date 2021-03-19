#!/bin/bash
#SBATCH -p gpu -n 1 -c 16
#SBATCH --time 12:00:00 --mem 60G

set -e

export PYTORCH_VERSION=v1.8.0 # 9dfbfe9 # a7cf04ec40e487286ad3f8068fa18321f3474dd2 # master
export VISION_VERSION=v0.9.0 # 9dfbfe9 # a7cf04ec40e487286ad3f8068fa18321f3474dd2 # master

module load 2020
module load GCC/9.3.0 CUDA/11.0.2-GCC-9.3.0 cuDNN/8.0.3.33-gcccuda-2020a

. $HOME/conda/etc/profile.d/conda.sh
conda activate cartesius_devel

export CFLAGS="-march=native -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O3 -ffunction-sections"
export CXXFLAGS="-fvisibility-inlines-hidden -fmessage-length=0 -march=native -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O3 -ffunction-sections"
export LDFLAGS="-Wl,-O3 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections"
export TORCH_CUDA_ARCH_LIST="3.5"
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CMAKE_PREFIX_PATH"
export USE_CUDNN=1
export USE_FBGEMM=1
export USE_KINETO=0
export USE_NUMPY=1
export BUILD_TEST=0
# export USE_MKL=0
export USE_MKLDNN=1
# export USE_MKLDNN_CBLAS=1
export MKLDNN_CPU_RUNTIME="OMP"
export USE_NNPACK=1
export USE_QNNPACK=0
export USE_XNNPACK=0
export USE_DISTRIBUTED=1
export USE_TENSORPIPE=1
export USE_GLOO=1
export USE_MPI=0
export USE_SYSTEM_NCCL=0
export BUILD_CAFFE2_OPS=1
export BUILD_CAFFE2=1
export USE_IBVERBS=0
export USE_OPENCV=0
export USE_OPENMP=1
export USE_FFMPEG=0
export USE_LEVELDB=0
export USE_LMDB=0
export USE_REDIS=0
export USE_ZSTD=0
export BLAS="MKL"
export MKL_THREADING="OMP"
export ATEN_THREADING="OMP"

export USE_STATIC_CUDNN=1
export USE_STATIC_NCCL=1

export CMAKE_ARGS="-DTORCH_CUDA_ARCH_LIST=3.5 $CMAKE_ARGS"
# export INTEL_COMPILER_DIR="/nonexistant"
# export INTEL_MKL_DIR="/nonexistant"
# export INTEL_OMP_DIR="/nonexistant"

if [ ! -e pytorch ]; then git clone https://github.com/pytorch/pytorch.git; fi
if [ ! -e vision ]; then git clone https://github.com/pytorch/vision.git; fi
# pushd pytorch
# git checkout $PYTORCH_VERSION
# git submodule sync
# git submodule update --init --recursive
# python3 -m pip install -v .
# popd

pushd vision
git checkout $VISION_VERSION
git submodule sync
git submodule update --init --recursive
python3 setup.py install # build
# python3 -m pip install -v .
popd
