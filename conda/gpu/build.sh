#!/bin/bash

set -ex

export CMAKE_LIBRARY_PATH=$PREFIX/lib:$PREFIX/include:$CMAKE_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$PREFIX:$SP_DIR/torch:$CMAKE_PREFIX_PATH

mkdir -p build
pushd build
rm -rf *
cmake -GNinja \
      -DCMAKE_CUDA_FLAGS="-cudart shared" \
      -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target install
popd

find nqs_playground/ -name "*.so*" -maxdepth 1 -type f | while read sofile; do
    echo "Setting rpath of $sofile to " '$ORIGIN:$ORIGIN/../torch/lib'
    patchelf --set-rpath '$ORIGIN:$ORIGIN/../torch/lib' --force-rpath $sofile
    patchelf --print-rpath $sofile
done

$PYTHON setup.py install
