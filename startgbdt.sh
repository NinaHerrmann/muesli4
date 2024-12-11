#!/bin/bash

./build.sh
build/gbdt/gbdt_cpu -g 0 -i 10 -n 2 -d covtype -t 32 -l 3
