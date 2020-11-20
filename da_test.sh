#!/bin/bash
# nvcc examples/dm_test.cu -I include/ -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -I include/detail/ -arch=compute_50 -code=sm_50 -o build/dmtest

# in the example program we have 
# arg[1] size of the datastructure
# arg[2] number of gpus
# arg[3] cpu fraction
# arg[4] check for correctness
# arg[5] call skeletons repeatedly
for size in 8 16; do
    ./da_test_seq $size 0 1 true 1 
    for np in 1 2; do
        mpirun -np $np ./da_test_cpu $size 0 $cpu_p true 1
    	for cpu_p in 0.00 0.5; do
		mpirun -np $np ./da_test_gpu $size 1 $cpu_p true 1
        done
    done
done
