#!/bin/bash
nvcc examples/gameoflife.cu -I include/ -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -I include/detail/ -arch=compute_50 -code=sm_50 -o build/gol

printf "n;Gpus;cpu_fraction;runs1;time;runs;runtime\n"
for m_size in 1024; do
    for gpu_n in 1 2; do
        for cpu_p in 0.00; do
		for iterations in 1000 ; do
        		#for tile_width in 8 16 24 32; do
			mpirun -np 1  build/gol $m_size $m_size $gpu_n 1 $cpu_p 16 $iterations "afucking.csv"
		#	done
		done
        done
    done
done

