#! /bin/bash
rm -rf build
mkdir build
cd build || exit
cmake ..
cmake --build . --target gassimulation_test_gpu