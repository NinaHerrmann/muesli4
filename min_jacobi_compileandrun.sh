#!/bin/bash
nvcc examples/jacobisolver.cu -I include/ -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -I include/detail/ -arch=compute_61 -code=sm_61 -o build/jacobisolver

printf "n;Gpus;cpu_fraction;runs1;time;runs;runtime\n"
for m_size in 32; do
    for gpu_n in 1; do
        for cpu_p in 0.00; do
		for np in 1; do
        		#for tile_width in 8 16 24 32; do
			build/jacobisolver $m_size $m_size $gpu_n 1 $cpu_p 16
		#	done
		done
        done
    done
done

