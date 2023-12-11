/*
 * dc.cpp
 *
 *      Author: Herbert Kuchen <kuchen@uni-muenster.de>,
 *              Nina Herrmann <nina.herrmann@uni-muenster.de>
 *
 * ---------------------------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2014-2020 Herbert Kuchen <kuchen@uni-muenster.de,
 *                     Nina Herrmann <nina.herrmann@uni-muenster.de>.
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
#include "muesli.h"
#include <iostream>
#include <dc.h>
#ifdef __CUDACC__
#include "map_kernels.cuh"
#endif

template<typename T>
msl::DC<T>::DC()
        :       // distributed array (resides on GPUs until deleted!)
        ncol(0), nrow(0), depth(0), // number of local elements on a node
        firstRow(0)   // first golbal row index of the DC on the local partition
{}


// constructor creates a non-initialized DC
template<typename T>
msl::DC<T>::DC(int row, int col, int depth) : ncol(col), nrow(row), depth(depth), DS<T>(row*col*depth) {
    firstRow = this->firstIndex / ncol;
    if (depth % (msl::Muesli::num_total_procs*msl::Muesli::num_gpus) != 0){
        throws(detail::InvalidCube());
        exit(0);
    }
#ifdef __CUDACC__
    initGPUs();
#endif
}

template<typename T>
msl::DC<T>::DC(int row, int col, int depth, bool rowComplete)
: ncol(col), nrow(row), depth(depth), rowComplete(rowComplete), DS<T>(row*col*depth) {

    firstRow = this->firstIndex / ncol;
#ifdef __CUDACC__
    initGPUs();
#endif
}

// constructor creates a DC, initialized with v
template<typename T>
msl::DC<T>::DC(int row, int col, int depth, const T &v)
: ncol(col), nrow(row), depth(depth), DS<T>(row*col*depth, v) {

    firstRow = this->firstIndex / ncol;
#ifdef __CUDACC__
    initGPUs();
#endif
}

template<typename T>
msl::DC<T>::DC(int row, int col, int depth, const T &v, bool rowComplete)
        : ncol(col), nrow(row), depth(depth), rowComplete(rowComplete), DS<T>(row*col*depth, v) {

    firstRow = this->firstIndex / ncol;
#ifdef __CUDACC__
    initGPUs();
#endif
}

template<typename T>
msl::DC<T>::DC(const DC <T> &other)
        : 
          nlocalRows(other.nlocalRows), ncol(other.ncol), nrow(other.nrow), depth(other.depth),
          firstRow(other.firstRow),
          rowComplete(other.rowComplete) {
    copyLocalPartition(other);

    // this->cpuMemoryInSync = true;
    // this->updateDevice();
}

template<typename T>
msl::DC<T>::DC(DC <T> &&other)
 noexcept         : nlocalRows(other.nlocalRows), ncol(other.ncol), nrow(other.nrow), depth(other.depth),
          firstRow(other.firstRow),
          rowComplete(other.rowComplete) {
    other.plans = nullptr;
    this->localPartition = other.localPartition;
    other.localPartition = nullptr;
}

template<typename T>
msl::DC<T> &msl::DC<T>::operator=(DC <T> &&other) noexcept {
    if (&other == this) {
        return *this;
    }

    this->freeLocalPartition();

    this->localPartition = other.localPartition;
    other.localPartition = nullptr;
    this->freePlans();
    this->plans = other->plans;
    other->plans = nullptr;

    this->id = other.id;
    this->n = other.n;
    this->nLocal = other->nLocal;
    nlocalRows = other.nlocalRows;
    ncol = other.ncol;
    nrow = other.nrow;
    depth = other.depth;
    this->firstIndex = other.firstIndex;
    firstRow = other.firstRow;
    this->np = other.np;
    this->cpuMemoryInSync = other.cpuMemoryInSync;
    this->gpuCopyDistributed = other.gpuCopyDistributed;
    this->ng = other.ng;
    this->nGPU = other.nGPU;
    this->indexGPU = other.indexGPU;
    rowComplete = other.rowComplete;
    return *this;
}

template<typename T>
msl::DC<T> &msl::DC<T>::operator=(const DC <T> &other) noexcept {
    if (&other == this) {
        return *this;
    }
    this->freeLocalPartition();
    copyLocalPartition(other);
    this->freePlans();
    this->plans = nullptr;
    this->plans = new GPUExecutionPlan<T>{*(other.plans)};

    this->id = other.id;
    this->n = other.n;
    this->nLocal = other->nLocal;
    nlocalRows = other.nlocalRows;
    ncol = other.ncol;
    nrow = other.nrow;
    depth = other.depth;
    this->firstIndex = other.firstIndex;
    firstRow = other.firstRow;
    this->np = other.np;
    this->cpuMemoryInSync = other.cpuMemoryInSync;
    this->gpuCopyDistributed = other.gpuCopyDistributed;
    this->ng = other.ng;
    this->nGPU = other.nGPU;
    this->indexGPU = other.indexGPU;
    rowComplete = other.rowComplete;
    return *this;
}


// template <typename T> void msl::DC<T>::swap(DC<T> &first, DC<T> &second) {}



// auxiliary method initGPUs
template<typename T>
void msl::DC<T>::initGPUs() {
#ifdef __CUDACC__
    for (int i = 0; i <this->ng; i++) {
        cudaSetDevice(i);
        this->plans[i].firstDepth = this->plans[i].first / (ncol * nrow);
        this->plans[i].firstRow = (this->plans[i].first - this->plans[i].firstDepth * (ncol*nrow)) / ncol;
        this->plans[i].firstCol = this->plans[i].first % ncol;
        this->plans[i].lastCol = (this->plans[i].first + this->plans[i].nLocal - 1) % ncol;
        this->plans[i].lastDepth = (this->plans[i].first + this->plans[i].nLocal - 1) / (ncol * nrow);
        this->plans[i].lastRow = ((this->plans[i].first + this->plans[i].nLocal - 1) - this->plans[i].lastDepth * (ncol*nrow)) / ncol;
        this->plans[i].gpuRows = this->plans[i].lastRow - this->plans[i].firstRow + 1;
        this->plans[i].gpuDepth = this->plans[i].lastDepth - this->plans[i].firstDepth + 1;
        // Error prone when not row complete.
        if (this->plans[i].gpuRows > 2) {
            this->plans[i].gpuCols = ncol;
        } else if (this->plans[i].gpuRows == 2) {
            if (this->plans[i].lastCol >= this->plans[i].firstCol) {
                this->plans[i].gpuCols = ncol;
            } else {
                this->plans[i].gpuCols = ncol - (this->plans[i].firstCol - this->plans[i].lastCol);
            }
        } else if (this->plans[i].gpuRows > 0) {
            this->plans[i].gpuCols = this->plans[i].lastCol - this->plans[i].firstCol;
        }
    }
#endif
}

// destructor removes a DC
template<typename T>
msl::DC<T>::~DC() {
}

// ***************************** auxiliary methods
// ******************************

template<typename T>
T msl::DC<T>::get3D(int row, int col, int ndepth, int gpu) const {
    int index = (row) * ncol + col + (nrow*ncol) * ndepth;
    T message;
    return this->getLocal(index);
}

template<typename T>
T msl::DC<T>::get_shared(int row, int column) const {
    int index = row * column;
    // TODO load from shared mem
    int idSource;
    T message;
    // TODO: adjust to new structure
    // element with global index is locally stored
    if (this->isLocal(index)) {
#ifdef __CUDACC__
        // element might not be up to date in cpu memory
        if (!this->cpuMemoryInSync) {
            // find GPU that stores the desired element
            int device = this->getGpuId(index);
            cudaSetDevice(device);
            // download element
            int offset = index - this->plans[device].first;
            (cudaMemcpyAsync(&message, this->plans[device].d_Data + offset,
                                              sizeof(T), cudaMemcpyDeviceToHost,
                                              Muesli::streams[device]));
        } else { // element is up to date in cpu memory
            message = this->localPartition[index - this->firstIndex];
        }
#else
        message = this->localPartition[index - this->firstIndex];
#endif
        idSource = Muesli::proc_id;
    }
        // Element with global index is not locally stored
    else {
        // Calculate id of the process that stores the element locally
        idSource = (int) (index / this->nLocal);
    }

    msl::MSL_Broadcast(idSource, &message, 1);
    return message;
}


// method (only) useful for debbuging.
template<typename T>
void msl::DC<T>::showLocal(const std::string &descr) {
    if (!this->cpuMemoryInSync) {
        this->updateHost();
    }
    if (msl::isRootProcess()) {
        std::ostringstream s;
        if (!descr.empty())
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
void msl::DC<T>::show(const std::string &descr) {
    #ifdef __CUDACC__
    cudaDeviceSynchronize();
    #endif
    T *b = new T[this->n];
    std::cout.precision(2);
    std::ostringstream s;
    if (!descr.empty())
        s << descr << ": " << std::endl;
    if (!this->cpuMemoryInSync) {
        this->updateHost();
    }


    msl::allgather(this->localPartition, b, this->nLocal);

    if (msl::isRootProcess()) {
        s << "[";
        for (int i = 0; i < this->n - 1; i++) {
            s << b[i];
            if ((i + 1) % (ncol *nrow) == 0) {
                s << "\n-------------\n " ;
            } else {
                ((i + 1) % ncol == 0) ? s << "\n " : s << " ";;
            }
        }
        s << b[this->n - 1] << "]" << std::endl;
        s << std::endl;
    }

    delete[] b;

    if (msl::isRootProcess())
        printf("%s", s.str().c_str());
}

// SKELETONS / COMMUNICATION / PERMUTE PARTITION

template<typename T>
void msl::DC<T>::permutePartition(int (*f)(int)) {
    printf("permute Partiion  (functor)\n");
    throws(detail::NotYetImplementedException());
}
template<typename T>
template<typename Functor>
inline void msl::DC<T>::permutePartition(Functor& f) {
    printf("permutePartition\n");
    throws(detail::NotYetImplementedException());

  /*int i, receiver;
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
    T* buffer = new T[this->nLocal];
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
  }*/
}


// template<typename T>
// inline void msl::DC<T>::permutePartition(int (*f)(int)) {
//  permutePartition(curry(f));
//}

template<typename T>
template<typename MapIndexFunctor>
void msl::DC<T>::mapIndexInPlace(MapIndexFunctor &f) {
#ifdef __CUDACC__

    //  int colGPU = (ncol * (1 - Muesli::cpu_fraction)) /this->ng;  // is not used, HK
    for (int i = 0; i <this->ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(8, 8, 8);
        int dimcol = ceil(this->plans[i].gpuCols / 8.0);
        int dimrow = ceil(this->plans[i].gpuRows / 8.0);
        int dimdepth = ceil(this->plans[i].gpuDepth / 8.0);
        dim3 dimGrid(dimcol, dimrow, dimdepth);
        detail::mapIndexKernelDC<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                this->plans[i].d_Data, this->plans[i].d_Data, this->plans[i].gpuRows, this->plans[i].gpuCols,
                this->plans[i].gpuDepth, this->plans[i].firstRow, this->plans[i].firstCol, this->plans[i].firstDepth, f);
    }
#endif
// all necessary calculations are performed otherwise some are skipped.
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int k = 0; k < this->nCPU; k++) {
        int l = (k + this->firstIndex) / (ncol*nrow);
        int j = ((k + this->firstIndex) - l*(ncol*nrow)) / ncol;
        int i = (k + this->firstIndex) % ncol;
        this->localPartition[k] = f(i, j, l, this->localPartition[k]);
    }
    // check for errors during gpu computation
    msl::syncStreams();

    this->cpuMemoryInSync = false;
}

template<typename T>
template<typename MapIndexFunctor>
void msl::DC<T>::mapIndex(MapIndexFunctor &f, DC <T> &b) {

#ifdef __CUDACC__

    // map on GPUs
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(8, 8, 8);
        int dimrow = ceil(this->plans[i].gpuRows / 8.0);
        int dimcol = ceil(this->plans[i].gpuCols / 8.0);
        int dimdepth = ceil(this->plans[i].gpuDepth / 8.0);
        dim3 dimGrid(dimrow, dimcol, dimdepth);
           detail::mapIndexKernelDC<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                   b.getExecPlans()[i].d_Data, this->plans[i].d_Data, this->plans[i].gpuRows, this->plans[i].gpuCols,
                this->plans[i].gpuDepth, this->plans[i].firstRow, this->plans[i].firstCol, this->plans[i].firstDepth, f);
   }
#endif

    if (this->nCPU > 0){
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int k = 0; k < this->nCPU; k++) {
        T *bPartition = b.getLocalPartition();
        int l = (k + this->firstIndex) / (ncol*nrow);
        int j = ((k + this->firstIndex) - l*(ncol*nrow)) / ncol;
        int i = (k + this->firstIndex) % ncol;
        this->localPartition[k] = f(i, j, l, bPartition[k]);
    }
}
   this->setCpuMemoryInSync(false);

    // check for errors during gpu computation
    msl::syncStreams();
}
// ************************************ zip
// ***************************************


template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DC<T>::zipIndexInPlace(DC <T2> &b, ZipIndexFunctor &f) {
    // zip on GPUs
#ifdef __CUDACC__
    for (int i = 0; i <this->ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(8, 8, 8);
        int dimrow = ceil(this->plans[i].gpuRows / 8.0);
        int dimcol = ceil(this->plans[i].gpuCols / 8.0);
        int dimdepth = ceil(this->plans[i].gpuDepth / 8.0);
        dim3 dimGrid(dimrow, dimcol, dimdepth);
        detail::zipIndexKernelDC<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                this->plans[i].d_Data, b.getExecPlans()[i].d_Data, this->plans[i].d_Data,
                f, this->plans[i].gpuRows, this->plans[i].gpuCols, this->plans[i].gpuDepth,
                this->plans[i].firstRow, this->plans[i].firstCol, this->plans[i].firstDepth);

    }
#endif
    if (this->nCPU > 0){
        T2 *bPartition = b.getLocalPartition();
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int k = 0; k < this->nCPU; k++) {
            int l = (k + this->firstIndex) / (ncol*nrow);
            int j = ((k + this->firstIndex) - l*(ncol*nrow)) / ncol;
            int i = (k + this->firstIndex) % ncol;
            this->localPartition[k] = f(i, j, l, this->localPartition[k], bPartition[k]);
        }
    }
     msl::syncStreams();
	// check for errors during gpu computation
    this->cpuMemoryInSync = false;
}
template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DC<T>::crossZipIndexInPlace(DC <T2> &b, ZipIndexFunctor &f) {
    // zip on GPUs
#ifdef __CUDACC__
    for (int i = 0; i <this->ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        //grid_size = (this->plans[i].size/block_size) + (!(Size%block_size)? 0:1);
        dim3 dimGrid((this->plans[i].size + dimBlock.x) / dimBlock.x);
        detail::crossZipInPlaceIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                this->plans[i].d_Data, b.getExecPlans()[i].d_Data,
                this->plans[i].nLocal, this->plans[i].first, f, ncol);
    }
#endif
    if (this->nCPU > 0){
        T2 *bPartition = b.getLocalPartition();
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int k = 0; k < this->nCPU; k++) {
            int i = (k + this->firstIndex) / ncol;
            int j = (k + this->firstIndex) % ncol;
            this->localPartition[k] = f(i, j, this->localPartition, bPartition);
        }
    }
    // check for errors during gpu computation
    this->cpuMemoryInSync = false;
}
template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DC<T>::zipIndex(DC <T2> &b, DC <T2> &c, ZipIndexFunctor &f) {
    // zip on GPUs
#ifdef __CUDACC__

    for (int i = 0; i <this->ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(8, 8, 8);
        int dimrow = ceil(this->plans[i].gpuRows / 8.0);
        int dimcol = ceil(this->plans[i].gpuCols / 8.0);
        int dimdepth = ceil(this->plans[i].gpuDepth / 8.0);
        dim3 dimGrid(dimrow, dimcol, dimdepth);

        detail::zipIndexKernelDC<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                b.getExecPlans()[i].d_Data, c.getExecPlans()[i].d_Data, this->plans[i].d_Data, f,
                        this->plans[i].gpuRows, this->plans[i].gpuCols, this->plans[i].gpuDepth,
                        this->plans[i].firstRow, this->plans[i].firstCol, this->plans[i].firstDepth);
    }
#endif
    // zip on CPU cores
    if (this->nCPU > 0){
        T2 *bPartition = b.getLocalPartition();
        T2 *cPartition = c.getLocalPartition();

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int k = 0; k < this->nCPU; k++) {
            int l = (k + this->firstIndex) / (ncol*nrow);
            int j = ((k + this->firstIndex) - l*(ncol*nrow)) / ncol;
            int i = (k + this->firstIndex) % ncol;
            this->localPartition[k] = f(i, j, l, cPartition[k], bPartition[k]);
        }
    }
    this->cpuMemoryInSync = false;
    msl::syncStreams();
}


template<typename T>
template<typename MapStencilFunctor, typename NeutralValueFunctor>
void msl::DC<T>::mapStencilInPlace(MapStencilFunctor &f, NeutralValueFunctor &neutral_value_functor) {
    printf("mapStencilInPlace\n");
    throws(detail::NotYetImplementedException());
}

template<typename T>
template<msl::DCMapStencilFunctor<T> f>
void msl::DC<T>::mapStencil(msl::DC<T> &result, size_t stencilSize, T neutralValue) {
#ifdef __CUDACC__
    this->updateDevice();
    syncPLCubes(stencilSize, neutralValue);
    syncPLCubesMPI(stencilSize);

    for (int i = 0; i < this->ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((this->plans[i].size + dimBlock.x - 1) / dimBlock.x);
        PLCube<T> cube = this->plCubes[i];
        detail::mapStencilKernelDC<T, f><<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(result.plans[i].d_Data, cube, result.plans[i].size);
    }
    result.setCpuMemoryInSync(false);
#else
    syncPLCubes(stencilSize, neutralValue);
    syncPLCubesMPI(stencilSize);
    const PLCube<T> cube = this->plCubes[0];
#ifdef _OPENMP
#pragma omp parallel for
#endif
    // Do we need local index?
    for (int k = 0; k < this->nLocal; k++) {
        int l = (k + this->firstIndex) / (ncol*nrow);
        int j = ((k + this->firstIndex) - l*(ncol*nrow)) / ncol;
        int i = (k + this->firstIndex) % ncol;
        // i is the column index, j is the row index, l is the depth index.
        result.localPartition[k] = f(cube, i, j, l);
    }
#endif
}

template<typename T>
void msl::DC<T>::set( int col, int row, int ldepth, T value) const {
    int index = (row * ncol) + col + (nrow * ncol * ldepth);
    if ((index >= this->firstIndex) && (index < this->firstIndex + this->nLocal)) {
        this->localPartition[index] = value;
    }
}
template<typename T>
T msl::DC<T>::get(int col, int row, int ldepth) const {
    int index = (row * ncol) + col + (nrow * ncol * ldepth);
    if ((index >= this->firstIndex) && (index < this->firstIndex + this->nLocal)) {
        return this->localPartition[index];
    }
    return 0;
}


