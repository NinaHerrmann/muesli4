#!/bin/bash

#SBATCH --export=NONE
#SBATCH --partition=gpu2080
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --time=15:00:00
#SBATCH --exclusive
#SBATCH --job-name=Muesli-solver
#SBATCH --outpu=/scratch/tmp/n_herr03/muesli-measurements/output_muesli.txt
#SBATCH --error=/scratch/tmp/n_herr03/muesli-measurements/error_muesli.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n_herr03@uni-muenster.de

module load intelcuda/2019a
module load CMake/3.15.3

cd /home/n/n_herr03/muesli4

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

for cpu_p in 0.02 0.04 0.06 0.08 0.10 0.12 0.14; do
    for m_size in 512 1000 5000 10000; do
        for gpu_n in 1 4; do
		for np in 1 2; do
        		mpirun -np np /home/n/n_herr03/muesli4/build/ninajacobi $m_size $m_size $gpu_n 10 $cpu_p "/scratch/tmp/n_herr03/muesli-measurements/ninajacobi.csv"
        	done
	done
    done    
done


#srun nvprof --analysis-metrics -o /scratch/tmp/e_zhup01/muesli-jacobi-analysis.%p.nvprof /home/e/e_zhup01/muesli4/build/jacobi -numdevices=1
# alternativ: mpirun -np 2 <Datei>
# alternativ: srun <Datei>
