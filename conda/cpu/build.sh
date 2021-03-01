#!/bin/bash

set -ex

export CMAKE_LIBRARY_PATH=$PREFIX/lib # :$CMAKE_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$PREFIX:$SP_DIR/torch # :$CMAKE_PREFIX_PATH

which cmake
cmake --version

mkdir -p build
pushd build
rm -rf *
cmake -GNinja \
      $CMAKE_ARGS \
      -DCMAKE_CXX_FLAGS="$CXXFLAGS -march=nehalem" \
      -DCMAKE_C_FLAGS="$CFLAGS -march=nehalem" \
      -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target install
popd

find nqs_playground/ -name "*.so*" -maxdepth 1 -type f | while read sofile; do
    echo "Setting rpath of $sofile to " '$ORIGIN:$ORIGIN/../torch/lib'
    patchelf --set-rpath '$ORIGIN:$ORIGIN/../torch/lib' --force-rpath $sofile
    patchelf --print-rpath $sofile
done

$PYTHON setup.py install
