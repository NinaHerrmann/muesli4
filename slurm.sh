#!/bin/bash
 
#SBATCH --export NONE
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 2
#SBATCH --gres=gpu:2
# alternativ: bis zu 4 auf gpu2080
#SBATCH --partition gpuk20
#SBATCH --time 00:05:00
#SBATCH --exclusive

#SBATCH --job-name dm_test
#SBATCH --output /scratch/tmp/kuchen/output4.txt
#SBATCH --error /scratch/tmp/kuchen/error4.txt
#SBATCH --mail-type ALL
#SBATCH --mail-user kuchen@uni-muenster.de

module load intelcuda/2019a
module load CMake/3.15.3

cd /home/k/kuchen/Muesli4

./build.sh 
export OMP_NUM_THREADS=4

# vorläufig, bis MPI über Infiniband funktioniert
export I_MPI_DEBUG=3
# export I_MPI_FABRICS=shm:ofa   nicht verfügbar
# alternativ: Ethernet statt Infiniband: 
export I_MPI_FABRICS=shm:tcp

# mpirun /home/k/kuchen/Muesli4/build/$1 $2 $3
# parameters: array dim #MPI nodes
# mpirun /home/k/kuchen/Muesli4/build/da_test 32 2

# parameters: area size (needs to be quadratic) #MPI nodes
# mpirun /home/k/kuchen/Muesli4/build/mandelbrotDA 10000 2 
 
# parameters: #processes (= dim of DA), #throws, #MPI nodes
# mpirun /home/k/kuchen/Muesli4/build/piDA 1000 1000000 2 

# parameters: #processes (= dim of DM), #throws, #MPI nodes
mpirun /home/k/kuchen/Muesli4/build/dm_test 32 2

# alternativ: mpirun -np 2 <Datei>
# alternativ: srun <Datei>
