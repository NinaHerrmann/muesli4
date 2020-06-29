#!/bin/bash
 
#SBATCH --export NONE
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:4
#SBATCH --partition gpu2080
# alternativ: gpuk20
#SBATCH --time 04:00:00
#SBATCH --exclusive

#SBATCH --job-name Muesli2-lena-GPU
#SBATCH --output /scratch/tmp/kuchen/outputGPUlena.txt
#SBATCH --error /scratch/tmp/kuchen/errorGPUlena.txt
#SBATCH --mail-type ALL
#SBATCH --mail-user kuchen@uni-muenster.de

cd ~/Muesli2
module load intelcuda/2019a
module load CMake/3.15.3

# vorläufig, bis MPI über Infiniband funktioniert
## export I_MPI_DEBUG=3
export I_MPI_FABRICS=shm:ofa
# alternativ: Ethernet statt Infiniband: export I_MPI_FABRICS=shm:tcp

mpirun ~/Muesli2/build/bin/canny_gpu
# alternativ: srun <Datei>
