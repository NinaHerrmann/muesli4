#!/bin/bash
cd build || exit
cmake  ..
cmake --build .
