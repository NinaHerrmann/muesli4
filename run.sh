#!/bin/bash
printf "n;Gpus;cpu_fraction;runs1;time;runs;runtime\n"
for m_size in 1024; do
    for gpu_n in 1 2; do
        for cpu_p in 0.25; do
		for np in 1; do
        		mpirun -np $np build/ninajacobi $m_size $m_size $gpu_n 6 $cpu_p
		done
        done
    done
done

