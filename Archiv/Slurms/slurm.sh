#!/bin/bash
 
#SBATCH --export NONE
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
## keine GPUs SBATCH --gres=gpu:3
#SBATCH --partition express
# alternativ: gpuk20, gpu2080
#SBATCH --time 02:00:00
#SBATCH --exclusive

#SBATCH --job-name Muesli2
#SBATCH --output /scratch/tmp/kuchen/outputLena.txt
#SBATCH --error /scratch/tmp/kuchen/errorLena.txt
#SBATCH --mail-type ALL
#SBATCH --mail-user kuchen@uni-muenster.de

cd /home/k/kuchen/Muesli2
module load intelcuda/2019a
module load CMake/3.15.3

## ./build.sh   bereits erfolgt
export OMP_NUM_THREADS=4

# vorläufig, bis MPI über Infiniband funktioniert
export I_MPI_DEBUG=3
export I_MPI_FABRICS=shm:ofa
# alternativ: Ethernet statt Infiniband: export I_MPI_FABRICS=shm:tcp

mpirun /home/k/kuchen/Muesli2/build/bin/canny_cpu
# alternativ: srun <Datei>
