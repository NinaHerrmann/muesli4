#!/bin/bash
 
#SBATCH --export NONE
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
## max. number of parallel jobs:
#SBATCH --ntasks=4
#SBATCH --cpus-per-task 4
##  no GPUs
#SBATCH --partition normal
# alternativ: gpu2080, gpuk20, express
#SBATCH --time 04:00:00
#SBATCH --exclusive

#SBATCH --job-name Muesli2-CPU examples
#SBATCH --output /scratch/tmp/kuchen/outputAllCPU.txt
#SBATCH --error /scratch/tmp/kuchen/errorAllCPU.txt
#SBATCH --mail-type ALL
#SBATCH --mail-user kuchen@uni-muenster.de

cd /home/k/kuchen/Muesli2
module load intelcuda/2019a
module load CMake/3.15.3

## ./build.sh
export OMP_NUM_THREADS=4

# vorläufig, bis MPI über Infiniband funktioniert
# export I_MPI_DEBUG=3
export I_MPI_FABRICS=shm:ofa
# alternativ: Ethernet statt Infiniband: export I_MPI_FABRICS=shm:tcp

for file in /home/k/kuchen/Muesli2/build/bin/*cpu
do
  mpirun $file &
done
wait
# alternativ: srun <Datei>
