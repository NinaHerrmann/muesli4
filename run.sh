#!/bin/bash
printf "n;Gpus;cpu_fraction;runs1;time;runs;runtime\n"
for m_size in 1024 2048 4096 8192; do
    for gpu_n in 1 2; do
        for cpu_p in 0.02 0.04 0.06 0.08 0.10 0.12 0.14; do
        mpirun build/jacobi $m_size $m_size $gpu_n 10 $cpu_p
        done
    done
done

