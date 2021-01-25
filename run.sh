#!/bin/bash
printf "n;Gpus;cpu_fraction;runs1;time;runs;runtime\n"
for m_size in 8; do
    for gpu_n in 1; do
        for cpu_p in 0.25; do
        mpirun -np 1 build/ninajacobi $m_size $m_size $gpu_n 1 $cpu_p
        done
    done
done

