#!/bin/sh

if [ $# -ne 2 ]; then
    echo "Expected two arguments: python interpreter and target"
    exit 1
fi

PYTHON="$1"
TARGET="$2"

get_include_paths()
{
    "$PYTHON" -c "from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()), end='')" \
        | grep -v "No CUDA runtime is found"
}

get_library_paths()
{
    { "$PYTHON" -c "from torch.utils.cpp_extension import library_paths; print(';'.join(library_paths()), end='')" \
        | grep -v "No CUDA runtime is found"; } || true
}

get_abi_flags()
{
    "$PYTHON" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI), end='')" \
        | grep -v "No CUDA runtime is found"
}

case "$TARGET" in
    include) get_include_paths ;;
    lib) get_library_paths ;;
    abi) get_abi_flags ;; 
    *)
        echo "target mus be either 'include', 'lib', or 'abi'"
        exit 1
    ;;
esac
