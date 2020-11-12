# Building example(s)

===================
./build.sh

Running new examples for testing Distributed Matrices (DM)
using MPI, OpenMP and CUDA.
============================================================0======================
sbatch slurm.sh  (includes build)
new example(s): dm_test.cu  (in directory examples)

NH, 03.08.2020
******************************************************

## Datastructure

The skeletons need to be updated to a new distribution for the datastructure to be efficient. To see the proposed distribution look at the .pdf file in this repository called Dm_distribution.pdf. 

Currently running:
* mapIndexInPlace

Test:
* Map (should be fine since it does not require indices)
* mapIndex -> requires adjustments since new Index calculation is required
* MapInPlace -> CPU adjustments GPU should be fine due to download() and upload() implementation

Next:
* MapVariants
* MapStencil for Endis Masterthesis
* Fold -> Fix

inefficiently implemented, but working:
* getLocal()
* setLocal()

still causing problems:
* lambda as parameter of a skeleton (if __device__ __host__)
* mapIndexInPlace with object rather than struct as parameter
* map

rest not yet tested
* zip and all Variants
* broadcastPartition
* permutePartition (even with lambda as argument)

NH, 12.11.2020
