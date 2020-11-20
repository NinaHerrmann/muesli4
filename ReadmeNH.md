# Muesli Building example(s)

____________________________________

./build.sh

Running new examples for testing Distributed Matrices (DM)
using MPI, OpenMP and CUDA.
__________________________________________________________

sbatch slurm.sh  (includes build)
new example(s): dm_test.cu/dc_test.cu  (in directory examples)

NH, 03.08.2020
******************************************************

## Datastructure

The skeletons need to be updated to a new distribution for the datastructure to be efficient. To see the proposed distribution look at the .pdf file in this repository called Dm_distribution.pdf. 

### Currently running:
* mapIndexInPlace
* mapIndex -> requires adjustments since new Index calculation is required
* MapInPlace -> CPU adjustments GPU should be fine due to download() and upload() implementation
* fold
* zip, zipIndex, zipIndexInPlace, zipInPlace

### Test:
* mapFold

NH, 17.11.2020

# Implementation Remarks
### Next:
* mapStencil for DM
* Subsume to DS?

### inefficiently implemented, but working:
* getLocal()
* setLocal()

### still causing problems:
* lambda as parameter of a skeleton (if __device__ __host__)
* mapIndexInPlace with object rather than struct as parameter
* map

### rest not yet implemented
* broadcastPartition
* permutePartition (even with lambda as argument)

