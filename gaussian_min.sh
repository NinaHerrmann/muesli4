#!/bin/bash
nvcc examples/gaussianblur.cu -I include/ -I/usr/lib/x86_64-linux-gnu/openmpi/include/ -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -I include/detail/ -arch=compute_50 -code=sm_50 -o build/gaussian

#printf "randgentime;calctime;iterations_reachedn;Gpus;tile_width;iterations;\n"
for np in 1; do
    for gpu_n in 1; do
        for cpu_p in 0.00; do
            for iterations in 1; do
                for kw in 2 4 6 8 10 12 14; do
                    for reps in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
                        for tile_width in 4 8; do
                            build/gaussian $gpu_n 1 $cpu_p $tile_width $iterations 1 $kw $reps
                        done
                    done
                    build/gaussian $gpu_n 1 $cpu_p 16 $iterations 0 $kw
                done
            done
        done
    done
done

for np in 1; do
    for gpu_n in 1; do
    	for cpu_p in 0.00; do
            for iterations in 1; do
	        for kw in 2 4 6 8 10 12 14; do
		    for reps in 1 2 3 4 5 6 7 8; do
                        for tile_width in 16 32; do
	                    build/gaussian $gpu_n 1 $cpu_p $tile_width $iterations 1 $kw $reps
	                done
		    done
	            build/gaussian $gpu_n 1 $cpu_p 16 $iterations 0 $kw
	        done
	    done
        done
    done
done
