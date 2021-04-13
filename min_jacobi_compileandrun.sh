#!/bin/bash
nvcc examples/min_ninajacobi_shared.cu -I include/ -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -I include/detail/ -arch=compute_50 -code=sm_50 -o build/jacobi_min

printf "n;Gpus;cpu_fraction;runs1;time;runs;runtime\n"
for m_size in 10000; do
    for gpu_n in 1 2; do
        for cpu_p in 0.00; do
		for np in 1; do
        		#for tile_width in 8 16 24 32; do
			build/jacobi_min $m_size $m_size $gpu_n 1 $cpu_p 16
		#	done
		done
        done
    done
done

