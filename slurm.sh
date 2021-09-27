#!/bin/bash

#SBATCH --export=NONE
#SBATCH --partition=gpu2080
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --time=15:00:00
#SBATCH --exclusive
#SBATCH --job-name=GaussianBlur
#SBATCH --output=/scratch/tmp/n_herr03/gaussian/muesli/errorandoutput/Gaussian.txt
#SBATCH --error=/scratch/tmp/n_herr03/gaussian/muesli/errorandoutput/Gaussian.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n_herr03@uni-muenster.de

module load intelcuda/2019a
module load CMake/3.15.3

cd /home/n/n_herr03/stenciltestgaussian/

./build.sh
export OMP_NUM_THREADS=4

# vorl�ufig, bis MPI �ber Infiniband funktioniert
export I_MPI_DEBUG=3
# export I_MPI_FABRICS=shm:ofa   nicht verf�gbar
# alternativ: Ethernet statt Infiniband:
export I_MPI_FABRICS=shm:tcp

# mpirun /home/k/kuchen/Muesli4/build/$1 $2 $3
# parameters: array dim #MPI nodes
# mpirun /home/k/kuchen/Muesli4/build/da_test 32 2

# parameters: area size (needs to be quadratic) #MPI nodes
# mpirun /home/k/kuchen/Muesli4/build/mandelbrotDA 10000 2

# parameters: #processes (= dim of DA), #throws, #MPI nodes
# mpirun /home/k/kuchen/Muesli4/build/piDA 1000 1000000 2

# parameters: #DMCols #DMRows #nGPU #nRuns #CpuPercentage

#echo  "n;Gpus;cpu_fraction;runs1;time;runs;runtime\n" >> "/scratch/tmp/n_herr03/timesplit_stenciltestgaussian.csv"
#for m_size in 1024 4096 8192 16384; do
#    for gpu_n in 1 2; do
#        for tile_width in 16; do
#                for iterations in 1000 2000 3000 4000 5000 10000; do
#                        for run in 1 2 3 4 5 6 7 8 9 10; do
#				for np in 1 2;
#                 	       mpirun -np $np /home/n/n_herr03/stenciltestgaussian/build/gameoflife $m_size $m_size $gpu_n 1 0.00 $tile_width $iterations "/scratch/tmp/n_herr03/timesplit_stenciltestgaussian.csv"
#                        done
#                done
#        done
#    done
#done

for np in 1; do
    for gpu_n in 1 2 4; do
        for cpu_p in 0.00; do
            for iterations in 1 5 10 20; do
		for kw in 2 10 20; do
                    for tile_width in 8 16 32; do
                    	mpirun -np $np /home/n/n_herr03/stenciltestgaussian/build/gaussianblur $gpu_n 20 $cpu_p $tile_width $iterations 1 $kw "/scratch/tmp/n_herr03/gaussian/muesli/SM/SM_${tile_width}_BIG.csv"
                    done
                    mpirun -np $np /home/n/n_herr03/stenciltestgaussian/build/gaussianblur $gpu_n 20 $cpu_p 12 $iterations 0 $kw "/scratch/tmp/n_herr03/gaussian/muesli/GM/GM_BIG.csv"
                done
            done
        done
    done
done


#srun nvprof --analysis-metrics -o /scratch/tmp/e_zhup01/muesli-jacobi-analysis.%p.nvprof /home/e/e_zhup01/muesli4/build/jacobi -numdevices=1
# alternativ: mpirun -np 2 <Datei>
# alternativ: srun <Datei>
