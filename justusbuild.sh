#! /bin/bash
rm -rf build
mkdir build
cd build || exit
cmake ..
cmake --build . --target dc_map_stencil_test