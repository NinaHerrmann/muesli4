#!/bin/bash
nvcc examples/gaussianblur.cu -I include/ -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -I include/detail/ -arch=compute_50 -code=sm_50 -o build/gaussian

#printf "randgentime;calctime;iterations_reachedn;Gpus;tile_width;iterations;\n"
for np in 1; do
    for gpu_n in 1; do
    	for cpu_p in 0.00; do
            for iterations in 1; do
	        for kw in 4; do
                    for tile_width in 16; do
	                mpirun -np $np build/gaussian $gpu_n 1 $cpu_p $tile_width $iterations 1 $kw
	            done
	            mpirun -np $np build/gaussian $gpu_n 1 $cpu_p 12 $iterations 0 $kw
	        done
	    done
        done
    done
done
