#!/bin/bash
 
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=32G                   # how much memory is needed per node (units can be: K, M, G, T)
#SBATCH --partition broadwell
# alternative: express, normal, gpuk20, gpu2080, broadwell, knl
#SBATCH --time 00:05:00
## SBATCH --exclusive

#SBATCH --job-name Muesli2:current
#SBATCH --output /scratch/tmp/kuchen/outputCurrent.txt
#SBATCH --error /scratch/tmp/kuchen/errorCurrent.txt
#SBATCH --mail-type ALL
#SBATCH --mail-user kuchen@uni-muenster.de

cd /home/k/kuchen/Muesli2

module load intelcuda/2019a
## no need for new build, since done previously
## module load CMake/3.15.3
## ./build.sh  
export OMP_NUM_THREADS=4

mpirun /home/k/kuchen/Muesli2/build/bin/nbody_cpu

