#!/bin/bash
nvcc -arch=compute_75 -code=sm_75 -O3 -use_fast_math -o build/jacobi -Iinclude/  -I/opt/openmpi/include/ -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -I/opt/openmpi-4.0.3/ompi/include/ -Iinclude/detail/ -lmpi examples/ninajacobi_shared.cu

printf "n;Gpus;cpu_fraction;runs1;time;runs;runtime\n"
for m_size in 5000; do
    for gpu_n in 1 2; do
        for cpu_p in 0.05; do
		for np in 1; do
        		#for tile_width in 8 16 24 32; do
			build/jacobi $m_size $m_size $gpu_n 1 $cpu_p 16
		#	done
		done
        done
    done
done

