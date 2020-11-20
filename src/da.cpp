/*
 * da.cu
 *
 *      Author: Herbert Kuchen <kuchen@uni-muenster.de>, S
 *              Steffen Ernsting <s.ernsting@uni-muenster.de>
 *              Nina Herrmann <nina.herrmann@uni-muenster.de>
 *
 * ---------------------------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2014-2020 Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *                     Herbert Kuchen <kuchen@uni-muenster.de.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#include <iostream>

template<typename T>
msl::DA<T>::DA(){}

// constructor creates a non-initialized DA
template<typename T>
msl::DA<T>::DA(int size)
        : DS<T>(size) {

}

// constructor creates a DA, initialized with v
template<typename T>
msl::DA<T>::DA(int size, const T &v)
        : DS<T>(size, v) {

}


// ***************************** Data-related Operations ******************************
/*
 * Can be called internally, or from the Program.
 */

// method (only) useful for debbuging
template<typename T>
void msl::DA<T>::showLocal(const std::string &descr) {
    if (!this->cpuMemoryInSync) {
        this->updateHost();
    }
    if (msl::isRootProcess()) {
        std::ostringstream s;
        if (descr.size() > 0)
            s << descr << ": ";
        s << "[";
        for (int i = 0; i < this->nLocal; i++) {
            s << this->localPartition[i] << " ";
        }
        s << "]" << std::endl;
        printf("%s", s.str().c_str());
    }
}

template<typename T>
void msl::DA<T>::show(const std::string &descr) {
    T *b = new T[this->n];
    std::ostringstream s;
    if (descr.size() > 0)
        s << descr << ": " << std::endl;
    if (!this->cpuMemoryInSync) {
        this->updateHost();
    }
    msl::allgather(this->localPartition, b, this->nLocal);

    if (msl::isRootProcess()) {
        s << "[";
        for (int i = 0; i < this->n - 1; i++) {
            s << b[i];
            s << " ";
        }
        s << b[this->n - 1] << "]" << std::endl;
        s << std::endl;
    }

    delete[] b;

    if (msl::isRootProcess()) printf("%s", s.str().c_str());
}


// SKELETONS / COMMUNICATION / BROADCAST PARTITION

template<typename T>
void msl::DA<T>::broadcastPartition(int partitionIndex) {
    if (partitionIndex < 0 || partitionIndex >= this->np) {
        throws(detail::IllegalPartitionException());
    }
    if (!this->cpuMemoryInSync)
        this->updateHost();
    msl::MSL_Broadcast(partitionIndex, this->localPartition, this->nLocal);
    this->cpuMemoryInSync = false;
    this->updateDevice();
}

// SKELETONS / COMMUNICATION / PERMUTE PARTITION

template<typename T>
template<typename Functor>
inline void msl::DA<T>::permutePartition(Functor &f) {
    int i, receiver;
    receiver = f(this->id);

    if (receiver < 0 || receiver >= Muesli::num_local_procs) {
        throws(detail::IllegalPartitionException());
    }

    int sender = UNDEFINED;

    // determine sender
    for (i = 0; i < Muesli::num_local_procs; i++) {
        // determine sender by computing invers of f
        if (f(i) == Muesli::proc_id) {
            if (sender == UNDEFINED) {
                sender = i;
            }                        // f is not bijective
            else {
                throws(detail::IllegalPermuteException());
            }
        }
    }

    // f is not bijective
    if (sender == UNDEFINED) {
        throws(detail::IllegalPermuteException());
    }

    if (receiver != Muesli::proc_id) {
        if (!this->cpuMemoryInSync)
            this->updateHost();
        T *buffer = new T[this->nLocal];
        for (i = 0; i < this->nLocal; i++) {
            buffer[i] = this->localPartition[i];
        }
        MPI_Status stat;
        MPI_Request req;
        MSL_ISend(receiver, buffer, req, this->nLocal, msl::MYTAG);
        MSL_Recv(sender, this->localPartition, stat, this->nLocal, msl::MYTAG);
        MPI_Wait(&req, &stat);
        delete[] buffer;
        this->cpuMemoryInSync = false;
        this->updateDevice();
    }
}

//*********************************** Maps ********************************


template<typename T>
template<typename MapIndexFunctor>
void msl::DA<T>::mapIndexInPlace(MapIndexFunctor &f) {
   // this->updateDevice();
#ifdef __CUDACC__
    for (int i = 0; i < Muesli::num_gpus; i++) {
      cudaSetDevice(i);
      dim3 dimBlock(Muesli::threads_per_block);
      dim3 dimGrid((this->plans[i].size+dimBlock.x)/dimBlock.x);
      detail::mapIndexKernelDA<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
          this->plans[i].d_Data, this->plans[i].d_Data, this->plans[i].nLocal, this->plans[i].first, f);
    }
#endif
    // calculate offsets for indices
    int offset = this->firstIndex;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < this->nCPU; i++) {
        this->localPartition[i] = f(i + offset, this->localPartition[i]);
    }
    // check for errors during gpu computation
    msl::syncStreams();
    this->cpuMemoryInSync = false;
}


template<typename T>
template<typename MapIndexFunctor>
void msl::DA<T>::mapIndex(MapIndexFunctor &f, DA<T> &b) {

    this->updateDevice();

    // map on GPUs
#ifdef __CUDACC__
    for (int i = 0; i < Muesli::num_gpus; i++) {
      cudaSetDevice(i);
      dim3 dimBlock(Muesli::threads_per_block);
      dim3 dimGrid((this->plans[i].size+dimBlock.x)/dimBlock.x);
      detail::mapIndexKernelDA<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                b.getExecPlans()[i].d_Data, this->plans[i].d_Data, this->plans[i].nLocal,
                this->plans[i].first, f);
    }
#endif
    // map on CPU cores
    T * blocalPartition = b.getLocalPartition();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < this->nCPU; i++) {
        this->setLocal(i, f((i + this->firstIndex), blocalPartition[i]));
    }

    // check for errors during gpu computation
    msl::syncStreams();
    this->setCpuMemoryInSync(false);
}

template<typename T>
template<typename MapStencilFunctor>
void msl::DA<T>::mapStencilInPlace(MapStencilFunctor &f, T neutral_value) {
    printf("mapStencilInPlace\n");
    throws(detail::NotYetImplementedException());
}

template<typename T>
template<typename R, typename MapStencilFunctor>
msl::DA<R> msl::DA<T>::mapStencil(MapStencilFunctor &f, T neutral_value) {
    printf("mapStencil\n");
    throws(detail::NotYetImplementedException());
}

// ************************************ zip ***************************************

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DA<T>::zipIndexInPlace(msl::DA<T2> &b, ZipIndexFunctor &f) {
    // zip on GPUs
    this->updateDevice();

#ifdef __CUDACC__
    for (int i = 0; i < Muesli::num_gpus; i++) {
      cudaSetDevice(i);
      dim3 dimBlock(Muesli::threads_per_block);
      dim3 dimGrid((this->plans[i].size+dimBlock.x)/dimBlock.x);
      detail::zipIndexKernelDA<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
          this->plans[i].d_Data, b.getExecPlans()[i].d_Data, this->plans[i].d_Data, this->plans[i].nLocal,
          this->plans[i].first, f);
    }
#endif
    // zip on CPU cores
    T2 *bPartition = b.getLocalPartition();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < this->nCPU; i++) {
        this->localPartition[i] = f(i + this->firstIndex, this->localPartition[i], bPartition[i]);
    }
    // check for errors during gpu computation
    msl::syncStreams();
    this->cpuMemoryInSync = false;
}

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DA<T>::zipIndex(msl::DA<T2> &b, msl::DA<T2> &c, ZipIndexFunctor &f) {  // should be return type DA<R>; debug type error!
    this->updateDevice();

    // zip on GPUs
#ifdef __CUDACC__
    for (int i =0; i < this->ng; i++) {
      cudaSetDevice(i);
      dim3 dimBlock(Muesli::threads_per_block);
      dim3 dimGrid((this->plans[i].size+dimBlock.x)/dimBlock.x);
      detail::zipIndexKernelDA<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
           b.getExecPlans()[i].d_Data, c.getExecPlans()[i].d_Data, this->plans[i].d_Data, this->plans[i].nLocal,
          this->plans[i].first, f);
    }
#endif
    // zip on CPU cores
    T2 *bPartition = b.getLocalPartition();
    T2 *cPartition = c.getLocalPartition();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < this->nCPU; i++) {
        this->setLocal(i, f(i + this->firstIndex, cPartition[i], bPartition[i]));
    }
    // check for errors during gpu computation
    msl::syncStreams();
    this->setCpuMemoryInSync(false);
}

template<typename T>
template<typename T2, typename T3, typename ZipFunctor>
void msl::DA<T>::zipInPlace3(DA <T2> &b, DA <T3> &c, ZipFunctor &f) {  // should be return type DA<R>; debug type error!
    // zip on GPU
#ifdef __CUDACC__
    for (int i = 0; i < this->ng; i++) {
      cudaSetDevice(i);
      dim3 dimBlock(Muesli::threads_per_block);
      dim3 dimGrid((this->plans[i].size+dimBlock.x)/dimBlock.x);
      auto bplans = b.getExecPlans();
      auto cplans = c.getExecPlans();
      detail::zipKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
          this->plans[i].d_Data, bplans[i].d_Data, cplans[i].d_Data, this->plans[i].d_Data, this->plans[i].size, f);
    }
#endif
    // zip on CPU cores
    T2 *bPartition = b.getLocalPartition();
    T3 *cPartition = c.getLocalPartition();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < this->nCPU; i++) {
        this->localPartition[i] = f(this->localPartition[i], bPartition[i], cPartition[i]);
    }
    // check for errors during gpu computation
    msl::syncStreams();
    this->cpuMemoryInSync = false;
}


