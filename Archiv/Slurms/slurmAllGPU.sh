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

#SBATCH --job-name Muesli2-examples-GPU
#SBATCH --output /scratch/tmp/kuchen/outputAllGPU.txt
#SBATCH --error /scratch/tmp/kuchen/errorAllGPU.txt
#SBATCH --mail-type ALL
#SBATCH --mail-user kuchen@uni-muenster.de

cd /home/k/kuchen/Muesli2
module load intelcuda/2019a
module load CMake/3.15.3

./build.sh
## export OMP_NUM_THREADS=4

# vorläufig, bis MPI über Infiniband funktioniert
## export I_MPI_DEBUG=3
## export I_MPI_FABRICS=shm:ofa     (nicht mehr available 22.06.2020; läuft nun auch so)
# alternativ: Ethernet statt Infiniband: export I_MPI_FABRICS=shm:tcp

for file in /home/k/kuchen/Muesli2/build/bin/*gpu
do
  mpirun $file
done
wait
# alternativ: srun <Datei>
