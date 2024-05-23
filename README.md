# Mue***nster*** S*keleton* Li*brary* - Muesli 

C++ Library to execute Skeletons. 

## Setup

Muesli can generate sequential programs, parallel CPU programs, parallel (Nvidia) GPU programs, multi nodes programs, and combinations of those. 
From our experiment a mixture of GPU and CPU parallelization is not recommended. At most local PC with weak GPUs might profit from splitting calculations between CPU and GPU.
Depending on your hardware :grey_question: is optional, :exclamation: is required.

### Frameworks

G++, make, cmake and MPI are required (Although it would be possible to exclude MPI it makes the CMakeFile easier ). 
When using the GPU nvcc is required - you can try to adjust the CMakeFile for using clang++... For the CPU OpenMP is required.
As we do not have many testing environments we just list the one we are using. However, to the best of our knowledge current versions should not produce a lot of errors.


| :exclamation: g++ :exclamation:| :exclamation: make  :exclamation:  | :exclamation: cmake  :exclamation: | :exclamation: MPI (e.g. OpenMPI) :exclamation: | :grey_question: nvcc :grey_question: | :grey_question: OpenMP :grey_question: |
|---|----------------------|--------|---------------|-------|------------------------|
|11.3.0| 4.3                  | 3.22.1 | 4.1.2         | 11.8  | 4.5                    |


### Build

For the most simple use just run `./build.sh` which builds all examples in the `build/examplename/...` folder.
If CMake finds OpenMP CPU programs are generated. The same applies to CUDA and GPU programs. 
The output of CMake gives you hints if you scroll on top of all warnings ... 
```make
OpenMP found can build CPU variant.
...
Cuda found can build GPU variant.
MPI found can use mpirun to start multiple nodes.
```

Important variables are:
- CMAKE_BUILD_TYPE: Debug or Release
- CMAKE_CXX_COMPILER: the C++ compiler
e.g. On a cluster you probably want to use:
  `cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Release path/to/muesli/`

If you want to add examples just add targets to the CMakeFile `set(EXAMPLE_SRCS`.


## Run

Use the `mpirun` command to execute a program.

`mpirun -n <number of processes> /path/to/muesli/examples/executable <command line arguments>`

See the [openMPI FAQ](https://www.open-mpi.org/faq/?category=running) for more info on running MPI jobs.

Branches
--------

- *Master/Main* as semi-stable branch.
- Features are developed in *feature/* branches.
- Versions are marked with tags.

## Skeletons


| Structure                   | Map(Index,InPlace)  | Fold                | Zip(Index,InPlace) | Gather             | MapStencil            |
|-----------------------------|---------------------|---------------------|--------------------|--------------------|-----------------------|
| DA-D(*istributed*)A(*rray*) | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :black_square_button: |
| D(*istributed*)M(*atrix*)   | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:    |
| D(*istributed*)C(*ube*)     | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:    |

TODO:
- permutePartition (even with lambda as argument)
- lambda as parameter of a skeleton (if __device__ __host__)
<<<<<<< HEAD
- object rather than struct as parameter
=======
- object rather than struct as parameter
>>>>>>> 76d10f4 (README update and minor code)
