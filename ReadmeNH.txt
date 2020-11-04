Building example(s)
===================
./build.sh

Running new examples for testing Distributed Matrices (DM)
using MPI, OpenMP and CUDA.
============================================================0======================
sbatch slurm.sh  (includes build)

  new example(s): dm_test.cu  (in directory examples)

NH, 03.08.2020
******************************************************


Currently running:
- example dm_test.cu
- mapInPlace
- mapIndexInPlace
- fold

inefficiently implemented, but working:
- getLocal()
- setLocal()

still causing problems:
- lambda as parameter of a skeleton (if __device__ __host__)
- mapIndexInPlace with object rather than struct as parameter
- map

rest not yet tested
- zipInPlace
- broadcastPartition
- permutePartition (even with lambda as argument)

NH, 29.06.2020
