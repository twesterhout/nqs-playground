#!/usr/bin/env bash

set -e
set -o pipefail

get_repo_root() {
  git rev-parse --show-toplevel
}

activate_environment() {
  if ! conda env list | grep -q nqs_devel_cpu; then
    echo "You do not have nqs_devel_cpu Conda environment. Creating it..."
    conda env create --file conda-cpu.yml
  fi
  echo "Activating nqs_devel_cpu environment..."
  . $(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh
  conda activate nqs_devel_cpu
}

build_cxx_code() {
  echo "Building C++ extension code..."
  mkdir -vp build
  pushd build
  declare -r site_packages_dir=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  cmake -GNinja \
    -DCMAKE_CXX_FLAGS="$CXXFLAGS -march=nehalem" \
    -DCMAKE_C_FLAGS="$CFLAGS -march=nehalem" \
    -DCMAKE_PREFIX_PATH="$site_packages_dir/torch/share/cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    ..
  cmake --build . --target install
  popd
  echo "Done building C++ code!"
}

install_python_package() {
  echo "Installing Python package..."
  python3 -m pip install -e .
}

main() {
  cd "$(get_repo_root)"
  activate_environment
  build_cxx_code
  install_python_package
}

main "$@"
