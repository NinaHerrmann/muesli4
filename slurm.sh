#!/bin/bash
 
#SBATCH --export NONE
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 2
#SBATCH --gres=gpu:4
#SBATCH --partition gpu2080
# alternativ: gpuk20
#SBATCH --time 00:05:00
#SBATCH --exclusive

#SBATCH --job-name Muesli-DA-Test
#SBATCH --output /scratch/tmp/kuchen/output7.txt
#SBATCH --error /scratch/tmp/kuchen/error7.txt
#SBATCH --mail-type ALL
#SBATCH --mail-user kuchen@uni-muenster.de

module load intelcuda/2019a
module load CMake/3.15.3

cd /home/k/kuchen/Muesli3

./build.sh
export OMP_NUM_THREADS=4

# vorläufig, bis MPI über Infiniband funktioniert
export I_MPI_DEBUG=3
# export I_MPI_FABRICS=shm:ofa   nicht verfügbar
# alternativ: Ethernet statt Infiniband: 
export I_MPI_FABRICS=shm:tcp

# dim #runs #gpus 
mpirun /home/k/kuchen/Muesli3/build/da_test 32 2  
# alternativ: mpirun -np 2 <Datei>
# alternativ: srun <Datei>
