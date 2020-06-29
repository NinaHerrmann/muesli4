Muesli - Readme
======
Project of the Muenster Skeleton Library (Muesli).

[![build status](https://wiwi-gitlab.uni-muenster.de/pi-research/Muesli/badges/master/build.svg)](https://wiwi-gitlab.uni-muenster.de/pi-research/Muesli/commits/master) [![coverage report](https://wiwi-gitlab.uni-muenster.de/pi-research/Muesli/badges/master/coverage.svg)](https://wiwi-gitlab.uni-muenster.de/pi-research/Muesli/commits/master)

Setup
------
There are certain tools that must be installed to get started.

### g++,cmake
In order to build programs with Muesli a modern C++ compiler with OpenMP and C++11 support, cmake, and possibly make are required.

Install g++, cmake, and make under OpenSUSE with 
`sudo zypper in gcc-c++ cmake make`

Tested versions:

g++: 4.8.5 (higher versions were not explicitly tested, but should work)

ICC: 14 (and higher)

make: 4.0

cmake: 3.3.2 (this is also minimal required version)

### CUDA
Follow the [pre-Installation actions](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions) 2.3 and 2.4.

Download the CUDA toolkit: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
(recommended installer type: deb/rpm (network)) and follow the installation instructions.

After the installation, follow the [instructions to set up the environment](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#environment-setup).

Instead of using LD_LIBRARY_PATH, you can add CUDA libs as shared libraries.
In Ubuntu (maybe also other distros) create a file `/etc/ld.xo.conf.d/cuda.conf` that contains the path to your CUDA installation folder (e.g., /usr/local/cuda/lib64).
Then run `ldconfig`. 

If you want to, [install the samples](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#install-samples).

Tested versions:

CUDA: 7.5

**CUDA 7.5 does not work well with the latest Ubuntu LTS version 16.04. Therefore, consider installing CUDA 8.0 RC**


### MPI

Muesli works with different MPI implementations. We tested OpenMPI, MPICH, Intel MPI, and bullxmpi.

#### Install OpenMPI
##### OpenSUSE
Install the packages `openmpi` and `openmpi-devel`.

`sudo zypper in openmpi openmpi-devel`

##### Ubuntu
`sudo apt install openmpi-bin openmpi-common libopenmpi-dev libopenmpi1.10`


Build
-----
With cmake out of source builds are possible and should be used.
Create a new folder (e.g., muesli-build-release) `mkdir muesli-build-release`.
Navigate into that folder `cd muesli-build-release`.
Then run cmake.

If you feel lucky, just run `cmake path/to/muesli/source`.
Otherwise, use the `-G` flag to specifiy the cmake generator.
The documentation lists all available generators: [cmake-generators](https://cmake.org/cmake/help/v3.3/manual/cmake-generators.7.html).

The compiler can be set by overriding the `CMAKE_CXX_COMPILER` variable. Either set the `CXX` environment variable or use the `-D` flag when calling `cmake` as described in the [FAQ](https://cmake.org/Wiki/CMake_FAQ#How_do_I_use_a_different_compiler.3F).

The build type is specified via the `CMAKE_BUILD_TYPE` variable. This is also set with the `-D` flag when running cmake. Supported types are Release, Debug, RelWithDebInfo, and MinSizeRel (you probably only need the first two types).

Thus, a cmake call to generate an Eclipse project with Unix makefiles for the latest nsight release, debug mode, and the g++ 4.2 compiler looks like that:
`cmake -G "Eclipse CDT4 - Unix Makefiles" -D CMAKE_ECLIPSE_VERSION=4.4 -D CMAKE_BUILD_TYPE=Debug -D CMAKE_CXX_COMPILER=g++-4.2 path/to/muesli/`

On a cluster you probably want to use:
`cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Release path/to/muesli/`

Important variables are:
- CMAKE_BUILD_TYPE: Debug or Release
- CMAKE_CXX_COMPILER: the C++ compiler
- CMAKE_ECLIPSE_VERSION: required when a Eclipse project is generated. The current version for Nsight 7.5 is 4.4 (the Eclipse version Nsight is based on).
- CUDA_TOOL_KIT: Path to the CUDA folder. E.g. if there are two different versions installed.

Now run `make`.
This will automatically build the Sequential, CPU, Phi and GPU variant for all programs in the folder `/path/to/muesli/examples`, depending on the available compiler and software (e.g., GPU version is only built if CUDA is available).
If you just want to build the sequential variant run `make seq`.
If you just want to build the CPU variant run `make cpu`.
If you just want to build the GPU variant run `make gpu`.
If you just want to build the Xeon Phi variant run `make phi`.

Run
---
Use the `mpirun` command to execute a program.

`mpirun -n <number of processes> /path/to/muesli/examples/executable <command line arguments>`

See the [openMPI FAQ](https://www.open-mpi.org/faq/?category=running) for more info on running MPI jobs.

Taurus
-----
If you have a login for Taurus, the HPC cluster of the TU Dresden, use `ssh -t <username>@login.zih.tu-dresden.de ssh taurus.hrsk.tu-dresden.de` to log in.

Taurus uses slurm as job scheduling system. 
See the TU Dresden [HPC-wiki](https://doc.zih.tu-dresden.de/hpc-wiki/bin/view/Compendium/Slurm) on how to write job files.
The project name is *p_algcpugpu*.

Branches
--------
This project follows the *anti-gitflow* pattern [blogpost](http://endoflineblog.com/gitflow-considered-harmful).

*master* as eternal development branch.

Features are developed in feature branches starting with *feature/*. After completion, they are merged back into master and are deleted.
The same applies to release (*release/*) and hotfix (*hotfix/*) branches.

Versions are marked with tags. Tags are placed on top of a release or hotfix branch. Then the branch is merged back into master and is deleted. The version of muesli is typically X.Y.Z. A new feature increases Y, a hotfix increases Z.

### Variants
Variants are, as the name suggests, variants of Muesli, where it is not planned to merge the certain feature set back into the master branch.
Branches containing a variant start with *variant/*.

#### Cranberry
Contains a variant that includes a dynamic load balancing mechanism for simultaneous CPU-GPU execution. 

#### Strawberry
Contains a variant that includes a static load balancing mechanism for simultaneous CPU-GPU execution of data parallel skeletons.
A distributed data structure takes an additional argument that determines the calculation ratio, i.e., the percentage of elements processed by the CPU.
