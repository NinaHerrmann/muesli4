Building example(s)
===================
./build.sh

Running new examples for testing Distributed Arrays (DA) using MPI, OpenMP and CUDA
============================================================0======================
sbatch slurm.sh  (includes build)

  new example(s): da_test.cu  (in directory examples)

HK, 26.06.2020
******************************************************


Currently running:
- example da_test.cu
- mapInPlace
- mapIndexInPlace
- fold
- zipInPlace
- broadcastPartition
- permutePartition (even with lambda as argument)

inefficiently implemented, but working:
- getLocal()
- setLocal()

still causing problems:
- lambda as parameter of a skeleton (if __device__ __host__) 
- mapIndexInPlace with object rather than struct as parameter
- map

rest not yet tested 

HK, 29.06.2020

