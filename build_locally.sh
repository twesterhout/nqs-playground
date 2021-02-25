#!/usr/bin/env bash

set -e
set -o pipefail

USE_CUDA=0

get_repo_root() {
  git rev-parse --show-toplevel
}

activate_environment() {
  if [ $USE_CUDA -eq 0 ]; then
	  declare -r env_name="nqs_devel_cpu"
	  declare -r env_file="conda-cpu.yml"
  else
	  declare -r env_name="nqs_devel_gpu"
	  declare -r env_file="conda-gpu.yml"
  fi
  if ! conda env list | grep -q "$env_name"; then
    echo "You do not have $env_name Conda environment. Creating it..."
    conda env create --file "$env_file"
  fi
  if ! echo "$CONDA_DEFAULT_ENV" | grep -q "$env_name"; then
	echo "Activating $env_name environment..."
	if ! which activate; then
	  . $(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh
	fi
	conda activate "$env_name"
  fi
}

build_cxx_code() {
  echo "Building C++ extension code..."
  mkdir -vp build
  pushd build
  declare -r site_packages_dir=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  cmake -GNinja \
	$CMAKE_ARGS \
	-DCMAKE_CUDA_FLAGS="-cudart shared --compiler-options -march=nehalem" \
    -DCMAKE_CXX_FLAGS="$CXXFLAGS -march=nehalem" \
    -DCMAKE_C_FLAGS="$CFLAGS -march=nehalem" \
    -DCMAKE_PREFIX_PATH="$site_packages_dir/torch/share/cmake" \
    -DCMAKE_BUILD_TYPE=Release \
	-DNQS_PLAYGROUND_USE_CUDA=$USE_CUDA \
    ..
  cmake --build . --target install
  popd
  echo "Done building C++ code!"
}

install_python_package() {
  echo "Installing Python package..."
  python3 -m pip install -e .
}

print_help() {
  echo ""
  echo "Usage: ./build_locally.sh [--help] [--cuda]"
  echo ""
  echo "This script builds and installs nqs_playground locally."
  echo ""
  echo "Options:"
  echo "  --help         Display this message."
  echo "  --cuda         Compile with GPU support."
  echo ""
}

main() {
  while [ $# -gt 0 ]; do
    key="$1"
    case $key in
    --cuda)
      USE_CUDA=1
      shift
      ;;
    --help)
      print_help
      exit 0
      ;;
    *)
      echo "Error: unexpected argument '$1'"
	  print_help
	  exit 1
      ;;
    esac
  done
  cd "$(get_repo_root)"
  activate_environment
  build_cxx_code
  install_python_package
}

main "$@"
