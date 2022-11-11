/*
 * dm.cpp
 *
 *      Author: Herbert Kuchen <kuchen@uni-muenster.de>,
 *              Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *              Nina Herrmann <nina.herrmann@uni-muenster.de>
 *
 * ---------------------------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2014-2020 Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *                     Herbert Kuchen <kuchen@uni-muenster.de,
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
#include <dm.h>


template<typename T>
msl::DM<T>::DM()
        :       // distributed array (resides on GPUs until deleted!)
        ncol(0), nrow(0),
        firstRow(0), nlocalRows(0), rowComplete(true)   // first golbal row index of the DM on the local partition
{}


// constructor creates a DM, initialized with v
template<typename T>
msl::DM<T>::DM(int row, int col, const T &v)
        : ncol(col), nrow(row), DS<T>(row*col, v) {

    if (this->nLocal% ncol == 0) {
        rowComplete = true;
    } else {
        rowComplete = false;
    }
    DMinit();
#ifdef __CUDACC__
    initGPUs();
#endif
}

template<typename T>
msl::DM<T>::DM(int row, int col, const T &v, bool rowComplete)
        : ncol(col), nrow(row), rowComplete(rowComplete), DS<T>(row*col, v) {
    DMinit();
#ifdef __CUDACC__
    initGPUs();
#endif
}

template<typename T>
msl::DM<T>::DM(const DM <T> &other)
        : ncol(other.ncol), nrow(other.nrow),
          nlocalRows(other.nlocalRows), firstRow(other.firstRow),
          rowComplete(other.rowComplete) {
    this->copyLocalPartition(other);
    this->cpuMemoryInSync = false;
    this->updateDevice() ;
}

template<typename T>
msl::DM<T>::DM(DM <T> &&other) noexcept
        : ncol(other.ncol), nrow(other.nrow),
          nlocalRows(other.nlocalRows), firstRow(other.firstRow),
          rowComplete(other.rowComplete) {
    other.plans = nullptr;
   this->localPartition = other.localPartition;
    other.localPartition = nullptr;
}

template<typename T>
msl::DM<T> &msl::DM<T>::operator=(DM <T> &&other) noexcept {
    if (&other == this) {
        return *this;
    }

    this->freeLocalPartition() ;

    this->localPartition = other.localPartition;
    other.localPartition = nullptr;
    this->freePlans() ;
    this->plans = other.plans;
    other.plans = nullptr;

    this->id = other.id;
    this->n = other.n;
    this->nLocal= other.nLocal;
    nlocalRows = other.nlocalRows;
    ncol = other.ncol;
    nrow = other.nrow;
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
msl::DM<T> &msl::DM<T>::operator=(const DM <T> &other) {
    if (&other == this) {
        return *this;
    }
    this->freeLocalPartition() ;
    this->copyLocalPartition(other);
    this->freePlans() ;
    this->plans = nullptr;
    this->plans = new GPUExecutionPlan<T>{*(other.plans)};


    this->id = other.id;
    this->n = other.n;
    this->nLocal= other.nLocal;
    nlocalRows = other.nlocalRows;
    ncol = other.ncol;
    nrow = other.nrow;
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
msl::DM<T>::~DM() = default;

// ************************* Auxiliary Methods not to be called from Program ********/

template<typename T>
void msl::DM<T>::DMinit() {

    this->n = ncol * nrow;
    if (!rowComplete) {
        // TODO not working for uneven data structures e.g. 7*7 results in 2 * 24.
        // *1.0f required to get float.
        if (ceilf((this->n * 1.0f) / this->np) - (this->n * 1.0f / this->np) != 0) {
            // If we have an even number of processes just switch roundf and roundl.
            if (this->np % 2 == 0) {
                if (this->id % 2 == 0) {
                    this->nLocal= roundf((this->n * 1.0f) / this->np);
                } else {
                    this->nLocal= roundl((this->n * 1.0f) / this->np);
                }
            } else {
                // if they are not even e.g. 5 processes 49 element just put the leftovers to first process
                // maybe not the most precise way but easy. However, not really relevant as many operation
                // do not work with uneven data structures.
                if (this->id == 0) {
                    this->nLocal= (this->n/ this->np) + ((this->n/ this->np) % this->np);
                } else {
                    this->nLocal= this->n / this->np;
                }
            }
        } else {
            this->nLocal= this->n / this->np;
        }
        nlocalRows = this->nLocal% ncol == 0 ? this->nLocal/ ncol : this->nLocal/ ncol + 1;
    } else {
        nlocalRows = nrow / this->np;
        this->nLocal= nlocalRows * ncol;
    }
    // TODO: This could result in rows being split between GPUs.
    //  Can be problematic for stencil ops.
    firstRow = this->firstIndex / ncol;
}

template<typename T>
void msl::DM<T>::initGPUs() {
#ifdef __CUDACC__
    for (int i = 0; i < this->ng; i++) {
        cudaSetDevice(i);
        this->plans[i].firstRow = this->plans[i].first / ncol;
        this->plans[i].firstCol = this->plans[i].first % ncol;
        this->plans[i].lastRow = (this->plans[i].first + this->plans[i].nLocal- 1) / ncol;
        this->plans[i].lastCol = (this->plans[i].first + this->plans[i].nLocal- 1) % ncol;
        this->plans[i].gpuRows = this->plans[i].lastRow - this->plans[i].firstRow + 1;
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


template<typename T>
T msl::DM<T>::get2D(int row, int col) const {
    int index = (row) * ncol + col;
    return this->get(index);
}

template<typename T>
void msl::DM<T>::show(const std::string &descr) {
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
    msl::allgather(this->localPartition, b, this->nLocal );

    if (msl::isRootProcess()) {
        s << "[";
        for (int i = 0; i < this->n - 1; i++) {
            s << b[i];
            ((i + 1) % ncol == 0) ? s << "\n " : s << " ";
        }
        s << b[this->n - 1] << "]" << std::endl;
        s << std::endl;
    }

    delete[] b;

    if (msl::isRootProcess())
        printf("%s", s.str().c_str());
}

// SKELETONS / COMMUNICATION / BROADCAST PARTITION

template<typename T>
void msl::DM<T>::broadcastPartition(int partitionIndex) {
    if (partitionIndex < 0 || partitionIndex >= this->np) {
        throws(detail::IllegalPartitionException());
    }
    if (!this->cpuMemoryInSync)
        this->updateDevice() ;
    msl::MSL_Broadcast(partitionIndex,this->localPartition, this->nLocal );
    this->cpuMemoryInSync = false;
    this->updateDevice() ;
}

// SKELETONS / COMMUNICATION / PERMUTE PARTITION

/*template<typename T>
template<typename Functor>
inline void msl::DM<T>::permutePartition(Functor& f) {
  int i, receiver;
  receiver = f(id);

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
      this->updateDevice() ;
    T* buffer = new T[nLocal];
    for (i = 0; i < this->nLocal ; i++) {
      buffer[i] =this->localPartition[i];
    }
    MPI_Status stat;
    MPI_Request req;
    MSL_ISend(receiver, buffer, req, this->nLocal , msl::MYTAG);
    MSL_Recv(sender,this->localPartition, stat, this->nLocal , msl::MYTAG);
    MPI_Wait(&req, &stat);
    delete[] buffer;
    this->cpuMemoryInSync = false;
    updateHost();
  }
}
*/
// template<typename T>
// inline void msl::DM<T>::permutePartition(int (*f)(int)) {
//  permutePartition(curry(f));
//}


//*********************************** Maps ********************************

template<typename T>
template<typename MapIndexFunctor>
void msl::DM<T>::mapIndexInPlace(MapIndexFunctor &f) {
    this->updateDevice() ;

#ifdef __CUDACC__

    for (int i = 0; i < this->ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((this->plans[i].size + dimBlock.x) / dimBlock.x);
        detail::mapIndexKernelDM<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                this->plans[i].d_Data, this->plans[i].d_Data, this->plans[i].nLocal, this->plans[i].first, f,
                        ncol);
    }
#endif
long elementsCPU = this->nCPU;
long firstIndexCPU = this->firstIndex;
// all necessary calculations are performed otherwise some are skipped.
#ifdef _OPENMP
#pragma omp parallel for shared(elementsCPU, firstIndexCPU, ncol, f) default(none)
#endif
    for (int k = 0; k < elementsCPU; k++) {
        int i = (k + firstIndexCPU) / ncol;
        int j = (k + firstIndexCPU) % ncol;
        this->localPartition[k] = f(i, j, this->localPartition[k]);
    }
    // check for errors during gpu computation
    msl::syncStreams();

    this->cpuMemoryInSync = false;
}

template<typename T>
template<typename MapIndexFunctor>
void msl::DM<T>::mapIndex(MapIndexFunctor &f, DM<T> &b) {
    this->updateDevice() ;
#ifdef __CUDACC__

    // map on GPUs
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((this->plans[i].size + dimBlock.x) / dimBlock.x);
        detail::mapIndexKernelDM<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                b.getExecPlans()[i].d_Data, this->plans[i].d_Data, this->plans[i].nLocal,
                        this->plans[i].first, f, ncol);
    }
#endif
    if (this->nCPU > 0) {
        //T2 *bPartition = b.getLocalPartition();
        long elementsCPU = this->nCPU;
        long firstIndexCPU = this->firstIndex;
#ifdef _OPENMP
#pragma omp parallel for shared(elementsCPU, firstIndexCPU, ncol, f, b) default(none)
#endif
        for (int k = 0; k < elementsCPU; k++) {
            int j = (k + firstIndexCPU) / ncol;
            int i = (k + firstIndexCPU) % ncol;
            this->localPartition[k] = f(i, j, b.localPartition[k]);
        }
    }

    // check for errors during gpu computation
    msl::syncStreams();
    this->setCpuMemoryInSync(false);
}

// ************************************ zip *************************************

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DM<T>::zipIndexInPlace(DM <T2> &b, ZipIndexFunctor &f) {
    this->updateDevice() ;
#ifdef __CUDACC__
    for (int i = 0; i < this->ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((this->plans[i].size + dimBlock.x) / dimBlock.x);
        detail::zipIndexKernelDM<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                this->plans[i].d_Data, b.getExecPlans()[i].d_Data, this->plans[i].d_Data,
                        this->plans[i].nLocal, this->plans[i].first, f, ncol);
    }
#endif
    if (this->nCPU > 0) {
        T2 *bPartition = b.getLocalPartition();
        long elementsCPU = this->nCPU;
        long firstIndexCPU = this->firstIndex;
#ifdef _OPENMP
#pragma omp parallel for shared(elementsCPU, firstIndexCPU, ncol, f, bPartition) default(none)
#endif
        for (int k = 0; k < elementsCPU; k++) {
            int i = (k + firstIndexCPU) / ncol;
            int j = (k + firstIndexCPU) % ncol;
            this->localPartition[k] = f(i, j,this->localPartition[k], bPartition[k]);
        }
    }
    // check for errors during gpu computation
    this->cpuMemoryInSync = false;
    msl::syncStreams();

}

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DM<T>::zipIndex(DM <T2> &b, DM<T2> &c, ZipIndexFunctor &f) {
    this->updateDevice() ;

    // zip on GPUs
#ifdef __CUDACC__

    for (int i = 0; i < this->ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((this->plans[i].size + dimBlock.x) / dimBlock.x);
        detail::zipIndexKernelDM<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                b.getExecPlans()[i].d_Data, c.getExecPlans()[i].d_Data, this->plans[i].d_Data,
                this->plans[i].nLocal, this->plans[i].first, f, ncol);
    }
#endif
    if (this->nCPU > 0) {
        // zip on CPU cores
        T2 *bPartition = b.getLocalPartition();
        T2 *cPartition = c.getLocalPartition();
        long elementsCPU = this->nCPU;
        long firstIndexCPU = this->firstIndex;
#ifdef _OPENMP
#pragma omp parallel for shared(elementsCPU, firstIndexCPU, ncol, f, bPartition, cPartition) default(none)
#endif
        for (int k = 0; k < elementsCPU; k++) {
            int i = (k + firstIndexCPU) / ncol;
            int j = (k + firstIndexCPU) % ncol;
            this->localPartition[k] = f(i, j, cPartition[k], bPartition[k]);
        }
    }
    // check for errors during gpu computation
    msl::syncStreams();
    this->setCpuMemoryInSync(false);
}

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DM<T>::crossZipIndexInPlace(DM <T2> &b, ZipIndexFunctor &f) {
    this->updateDevice() ;

    // zip on GPUs
#ifdef __CUDACC__
for (int i = 0; i < this->ng; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    //grid_size = (this->plans[i].size/block_size) + (!(Size%block_size)? 0:1);
    dim3 dimGrid((this->plans[i].size + dimBlock.x) / dimBlock.x);
    detail::crossZipInPlaceIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
            this->plans[i].d_Data, b.getExecPlans()[i].d_Data,
            this->plans[i].nLocal, this->plans[i].first, f, ncol);
}
#endif
if (this->nCPU > 0) {
    T2 *bPartition = b.getLocalPartition();
    long elementsCPU = this->nCPU;
    long firstIndexCPU = this->firstIndex;
// all necessary calculations are performed otherwise some are skipped.
#ifdef _OPENMP
#pragma omp parallel for shared(elementsCPU, firstIndexCPU, ncol, f, bPartition) default(none)
#endif
    for (int k = 0; k < elementsCPU; k++) {
        int i = (k + firstIndexCPU) / ncol;
        int j = (k + firstIndexCPU) % ncol;
        this->localPartition[k] = f(i, j,this->localPartition[k], bPartition);
    }
}
// check for errors during gpu computation
this->cpuMemoryInSync = false;
}


#ifdef __CUDACC__
template<typename T>
void msl::DM<T>::downloadupperpart(int paddingsize) {
    cudaSetDevice(0);

    // updateDevice data from device
    (cudaMemcpyAsync(this->plans[0].h_Data, this->plans[0].d_Data,
                                      paddingsize * sizeof(T), cudaMemcpyDeviceToHost,
                                      Muesli::streams[0]));

    // wait until updateDevice is finished

}
#endif
#ifdef __CUDACC__
template<typename T>
void msl::DM<T>::downloadlowerpart(int paddingsize) {
    int gpu = Muesli::num_gpus - 1;
    cudaSetDevice(gpu);

    // updateDevice data from device
    (cudaMemcpyAsync(this->plans[gpu].h_Data + (this->plans[gpu].nLocal- paddingsize), this->plans[gpu].d_Data +
    (this->plans[gpu].nLocal-paddingsize), paddingsize * sizeof(T), cudaMemcpyDeviceToHost,
                     Muesli::streams[gpu]));

    // wait until updateDevice is finished
}
#endif

template<typename T>
template<typename MapStencilFunctor, typename NeutralValueFunctor>
void msl::DM<T>::mapStencilInPlace(MapStencilFunctor &f, NeutralValueFunctor &neutral_value_functor) {
    printf("mapStencilInPlace\n");
    throws(detail::NotYetImplementedException());
}
/*template<typename T>
void msl::DM<T>::matchloadall(int paddingsize) {
#ifdef __CUDACC__
    int gpu = Muesli::num_gpus - 1;
    cudaSetDevice(gpu);

    // updateDevice data from device
    (cudaMemcpyAsync(plans[gpu].h_Data + (plans[gpu].this->nLocal- paddingsize), plans[gpu].d_Data + (plans[gpu].nLocal-paddingsize),
                     paddingsize * sizeof(T), cudaMemcpyDeviceToHost,
                     Muesli::streams[gpu]));

    // wait until updateDevice is finished
#endif
}
template<typename T>
void msl::DM<T>::updatecenter(T * destination, T* source) {
#ifdef __CUDACC__
    int gpu = Muesli::num_gpus - 1;
    cudaSetDevice(gpu);

    // wait until updateDevice is finished
#endif
}*/
/*
#ifdef __CUDACC__

template<typename T>
template<typename T2, typename MapStencilFunctor, typename NeutralValueFunctor>
void msl::DM<T>::mapStencilMM(DM<T2> &result, MapStencilFunctor &f,
                                   NeutralValueFunctor &neutral_value_functor) {
    double t = MPI_Wtime();

    if (!rowComplete) {
        std::cout << "The matrix must be distributed between nodes in full rows to "
                     "use the map stencil skeleton\n";
        fail_exit();
    }
    if (Muesli::debug) {
        if(f.getSharedMemory()){
            //printf("Shared Memory is set to %s\n\n", f.getSharedMemory() ? "true" : "false");
        }
        if(!f.getSharedMemory()){
            //printf("Using GM");
        }
    }
    // Obtain stencil size.
    int stencil_size = f.getStencilSize();

    int padding_size = (stencil_size * ncol) + (2 * stencil_size);
    int col_size = (stencil_size * ncol);
    int kw = 2 * stencil_size;
    if (!plinitMM) {
        // TODO bigger stencils than 1
        padding_stencil = new T[(padding_size) * 4];
        d_dm = std::vector<T *>(Muesli::num_gpus);
        all_data = std::vector<T *>(Muesli::num_gpus);
        vplm = std::vector<PLMatrix < T> * > (Muesli::num_gpus);
        for (int i = 0; i < Muesli::num_gpus; i++) {
            cudaSetDevice(i);
            (cudaMalloc(&d_dm[i], padding_size * 4 * sizeof(T)));
            (cudaMalloc(&all_data[i], ((this->plans[i].gpuRows+kw) * (this->plans[i].gpuCols+kw)) * sizeof(T)));
            //printf("\nSize: %d\n\n", ((padding_size * 4) + (plans[0].gpuRows * plans[0].gpuCols)));
        }
    }
    int rowoffset = plans[0].gpuCols + (2*stencil_size);
    int coloffset = plans[0].gpuRows + (2*stencil_size);
    // Fill first and last GPU in total with NVF
    if (!plinitMM) {
        for (int j = 0; j < Muesli::num_gpus; j++) {

            // In case it is the last GPU and the last process take the nvf
            if (j == (Muesli::num_gpus - 1) && Muesli::proc_id == (Muesli::num_local_procs - 1)) {
#pragma omp parallel for
                for (int i = padding_size; i < padding_size * 2; i++) {
                    int offset = (nlocalRows + stencil_size) * ncol + ((padding_size * 2) - i);
                    padding_stencil[i] =
                            neutral_value_functor(offset / ncol + firstRow - stencil_size, offset % ncol);
                }
                cudaMemcpyAsync(d_dm[j] + padding_size, padding_stencil + padding_size,
                                padding_size * sizeof(T), cudaMemcpyHostToDevice, Muesli::streams[j]);
            }
            // In case it is the first GPU and the first process take the nvf
            if (j == 0 && Muesli::proc_id == 0) {
#pragma omp parallel for
                for (int i = 0; i < padding_size; i++) {
                    padding_stencil[i] =
                            neutral_value_functor(i / ncol - stencil_size, i % ncol);
                }
                cudaMemcpyAsync(d_dm[j], padding_stencil,
                                padding_size * sizeof(T), cudaMemcpyHostToDevice, Muesli::streams[j]);
                dim3 fillthreads(Muesli::threads_per_block);
                dim3 fillblocks((coloffset) / fillthreads.x);
                // int paddingoffset, int gpuRows, int ss
                detail::fillsides<<<fillblocks,fillthreads, 0, Muesli::streams[j]>>>(all_data[j], rowoffset, plans[j].gpuCols, stencil_size);
            }
        }
    }


    MPI_Status stat;
    MPI_Request req;

    if (msl::Muesli::num_total_procs > 1) {
        // updateDevice the data from the GPU which needs to be send to other process
        updateDeviceupperpart(col_size);
        updateDevicelowerpart(col_size);
        if (Muesli::proc_id < Muesli::num_local_procs - 1) {
            T *buffer =this->localPartition + (nlocalRows - stencil_size) * ncol;
            MSL_ISend(Muesli::proc_id + 1, buffer, req, col_size, msl::MYTAG);
        }

        // Blocking receive.
        // If it is not the first process receive the bottom of the previous process and copy it to the top.
        if (Muesli::proc_id > 0) {
            MSL_Recv(Muesli::proc_id - 1, padding_stencil + (stencil_size), stat, col_size,
                     msl::MYTAG);
        }
        // Wait for completion.
        if (Muesli::proc_id < Muesli::num_local_procs - 1) {
            MPI_Wait(&req, &stat);
        }
        // Bottom up (send first stencil_rows to predecessor):
        // Non-blocking send.
        // If it is not the first process send top.
        if (Muesli::proc_id > 0) {
            MSL_ISend(Muesli::proc_id - 1,this->localPartition, req, col_size,
                      msl::MYTAG);
        }
        // Blocking receive.
        // If it is not the last process receive the top of the following process and copy it to the bottom.
        if (Muesli::proc_id < Muesli::num_local_procs - 1) {
            T *buffer = padding_stencil + padding_size + (stencil_size);
            MSL_Recv(Muesli::proc_id + 1, buffer, stat, col_size, msl::MYTAG);
        }
        // Wait for completion.
        if (Muesli::proc_id > 0) {
            MPI_Wait(&req, &stat);
        }
    }
    if (!plinitMM) {
        for (int i = 0; i < stencil_size; i++) {
            // If it was not initialized we need to fill all corners --> start of top
            padding_stencil[0 + i] =
                    neutral_value_functor(-stencil_size, (-stencil_size) + i);
            // If it was not initialized we need to fill all corners -->  end of top
            padding_stencil[stencil_size + col_size + i] =
                    neutral_value_functor(-stencil_size, stencil_size + col_size + i);
            // If it was not initialized we need to fill corners --> start of bottom
            padding_stencil[padding_size + i] =
                    neutral_value_functor((-nrow - stencil_size) + i, (-stencil_size) + i);
            // If it was not initialized we need to fill all corners --> end of bottom
            padding_stencil[padding_size + stencil_size + col_size + i] =
                    neutral_value_functor((-nrow - stencil_size) + i, stencil_size + col_size + i);
        }
    }
    // overall rows and column, gpu rows and columns
    msl::PLMatrix<T> plm(nrow, ncol, plans[0].gpuRows, plans[0].gpuCols, stencil_size, f.getTileWidth(), Muesli::reps);

    // TODO copy first GPU row tothis->localPartition
    int tile_width = f.getTileWidth();

    float milliseconds = 0.0;
    // NVF Values only need to be copied once

    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);

        // TODO adjust to process (copy gotten stencil)
        // If it is the first GPU copy first part from received paddingstencil
        if (i == 0) {
            cudaMemcpy(d_dm[i], padding_stencil,
                       padding_size * sizeof(T), cudaMemcpyHostToDevice);
        } else {
            // If it is not the first GPU the top is always copied from the previous GPU.
            cudaMemcpy(d_dm[i] + stencil_size, plans[i - 1].d_Data + (this->plans[i].this->nLocal- (stencil_size * ncol)),
                       stencil_size * ncol * sizeof(T), cudaMemcpyDeviceToDevice);
        }
        // If it is the last GPU copy data from received padding_stencil.
        if (i == (Muesli::num_gpus - 1)) {
            cudaMemcpy(d_dm[i] + padding_size, padding_stencil + padding_size,
                       padding_size * sizeof(T), cudaMemcpyHostToDevice);

        } else {
            // If it is not the last GPU the bottom is always copied from the following GPU
            cudaMemcpy(d_dm[i] + padding_size + stencil_size, plans[i + 1].d_Data,
                       padding_size * sizeof(T), cudaMemcpyDeviceToDevice);
        }
        for (int k = 0; k < padding_size; k++) {
            padding_stencil[k + 2 * padding_size] =
                    neutral_value_functor(k, -1);
        }
        for (int k = 0; k < padding_size; k++) {
            padding_stencil[k + 3 * padding_size] =
                    neutral_value_functor(k, nrow + 1);
        }

        cudaMemcpyAsync(d_dm[i] + (2 * padding_size), padding_stencil + (2 * padding_size),
                        2 * padding_size * sizeof(T), cudaMemcpyHostToDevice, Muesli::streams[i]);
        cudaMemcpyAsync(all_data[i], padding_stencil,
                        padding_size * sizeof(T), cudaMemcpyHostToDevice, Muesli::streams[i]);
        cudaMemcpyAsync(all_data[i]+((this->plans[i].gpuRows+stencil_size)*(this->plans[i].gpuCols+kw)), padding_stencil + padding_size,
                        padding_size * sizeof(T), cudaMemcpyHostToDevice, Muesli::streams[i]);
       dim3 fillthreads(tile_width, tile_width);
       dim3 fillblocks((this->plans[i].gpuRows + fillthreads.x - 1) / fillthreads.x, (this->plans[i].gpuCols + fillthreads.y - 1) / fillthreads.y);
       // int paddingoffset, int gpuRows, int ss
       if (Muesli::debug) {
           gpuErrchk(cudaPeekAtLastError());
           gpuErrchk(cudaDeviceSynchronize());
       }
       detail::fillcore<<<fillblocks,fillthreads, 0, Muesli::streams[i]>>>(all_data[i], this->plans[i].d_Data, stencil_size * (this->plans[i].gpuCols + (2*stencil_size)), this->plans[i].gpuCols, stencil_size);
        if (Muesli::debug) {
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }
        plm.addDevicePtr(all_data[i]);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < msl::Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        plm.setFirstRowGPU(this->plans[i].firstRow);
        cudaMalloc((void **) &vplm[i], sizeof(PLMatrix < T > ));
        (cudaMemcpyAsync(vplm[i], &plm, sizeof(PLMatrix < T > ),
                                cudaMemcpyHostToDevice, Muesli::streams[i]));
        plm.update();
    }
    cudaDeviceSynchronize();

    // Map stencil
    */
/*int smem_size = (tile_width + 2 * stencil_size) *
                    (tile_width + 2 * stencil_size) * sizeof(T) * 2;*//*

    int smem_size = (tile_width + 2 * stencil_size) *
                    (tile_width + 2 * stencil_size) * sizeof(T) * 2;

    for (int i = 0; i < Muesli::num_gpus; i++) {
        f.init(this->plans[i].gpuRows, this->plans[i].gpuCols, this->plans[i].firstRow,
               this->plans[i].firstCol);
        f.notify();

        cudaSetDevice(i);
        int divisor = 1;
        if (f.getSharedMemory()) {
            dim3 dimBlock(tile_width, tile_width);
            int kw = stencil_size * 2;
            if (this->plans[i].gpuRows % tile_width != 0){
                //printf("\nRight now number of rows must be dividable by tile width\n");
            }
            if (this->plans[i].gpuCols % tile_width != 0) {
                //printf("\nRight now number of columns must be dividable by tile width\n");
            }
            // cudaDeviceProp prop;
            // cudaGetDeviceProperties(&prop, i);
            // 68 for Palma. -> each SM can start one block TODO for opt.
            // int sms = prop.multiProcessorCount;
            // float smpp = prop.sharedMemPerBlock;
            // We assume that this is an even number
            //printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
            if (Muesli::debug) {
                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }
            divisor = Muesli::reps;
            // printf("\ncurrent size %d Previous size %d %d\n", ((divisor * tile_width) + kw) * (tile_width + kw) * sizeof(T), (tile_width + 2 * stencil_size) *
            // (tile_width + 2 * stencil_size) * sizeof(T) * 2, stencil_size);
            dim3 dimGrid(((this->plans[i].gpuRows))/divisor/dimBlock.x,
                         (this->plans[i].gpuCols + dimBlock.y - 1) / dimBlock.y);
            smem_size = ((divisor * tile_width) + kw) * (tile_width + kw) * sizeof(T);
            //printf("\n %d %d; %d %d \n\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
            //printf("Rows %d Cols %d %d %d %d %d \n", this->plans[i].gpuRows, this->plans[i].gpuCols, dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
            detail::mapStencilMMKernel<<<dimGrid, dimBlock, smem_size, Muesli::streams[i]>>>(
                    result.getExecPlans()[i].d_Data, this->plans[i].gpuRows, this->plans[i].gpuCols, this->plans[i].firstCol, this->plans[i].firstRow, vplm[i],all_data[i], f, tile_width, divisor, kw);
            if (Muesli::debug) {
                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }
        }
        if (!f.getSharedMemory()){
            //dim3 dimBlock(Muesli::threads_per_block);
            //dim3 dimGrid((this->plans[i].size + dimBlock.x) / dimBlock.x);
            dim3 dimBlock(tile_width, tile_width);

            divisor = Muesli::reps;

            dim3 dimGrid(((this->plans[i].gpuRows))/divisor,
                         (this->plans[i].gpuCols + dimBlock.y - 1) / dimBlock.y);
            detail::mapStencilGlobalMem_rep<<<dimGrid, dimBlock, smem_size, Muesli::streams[i]>>>(
                    result.getExecPlans()[i].d_Data, this->plans[i], vplm[i], f, i, divisor, tile_width);
            if (Muesli::debug) {
                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }
        }
        cudaDeviceSynchronize();
    }

    f.notify();
    if (this->nCPU != 0) {
        if (Muesli::debug)
            printf("Calculating %d Elements on the CPU ... \n", this->nCPU);
        #pragma omp parallel for
        for (int i = 0; i < this->nCPU; i++) {
            // TODO CPU PLM Matrix
            //result.setLocal(i, f(i / ncol + firstRow, i % ncol,this->localPartition, nrow, ncol));
        }
    }

    plinitMM = true;
}
#endif*/


// **************************************************************************************************
// ************************************ "Communication" *********************************************
// **************************************************************************************************
/* rotateRows
* 1 1 1 1          2 2 2 2           4 4 4 4
* 2 2 2 2    -1    3 3 3 3      2    1 1 1 1
* 3 3 3 3  ------> 4 4 4 4   ------> 2 2 2 2
* 4 4 4 4          1 1 1 1           3 3 3 3
 */
template<typename T>
void msl::DM<T>::rotateRows(int a) {
#ifdef __CUDACC__
#endif
    bool negative = a < 0;
    int howmuch = a;
    if (negative) {
        howmuch = -1 * a;
    }
    if (howmuch == nrow || a == 0) {
        // if the number to rotate is equal to rows data stays the same.
        return;
    }
    if (!rowComplete) {
        if (msl::isRootProcess()) {
            throws(detail::RotateRowCompleteNotImplementedException());
        }
        return;
    }
    if (howmuch > nlocalRows) {
        if (msl::isRootProcess()) {
            throws(detail::RotateRowManyNotImplementedException());
        }
        return;
    }
    // easy approach put all to cpu.
    T *switchPartition = new T[howmuch * ncol];
    this->updateDevice() ;
    /* Depending on negative we need to write the "lower" or "upper" elements into the buffer
     * and write rows up or downwards.
    */
    T *doublePartition = new T[this->nLocal];

    for (int i = 0; i < this->nLocal ; i++) {
        doublePartition[i] =this->localPartition[i];
    }
    if (negative) {
        for (int i = 0; i < howmuch * ncol; i++) {
            switchPartition[i] =this->localPartition[i];
        }
    } else {
        for (int i = 0; i < howmuch * ncol; i++) {
            switchPartition[i] =this->localPartition[this->nLocal- (howmuch * ncol) + i];
        }
    }
    // TODO switch between the MPI Processes
    T *buffer = new T[howmuch * ncol];

    if (msl::Muesli::num_total_procs > 1) {
        MPI_Status stat;
        MPI_Request req;
        int send_size = howmuch * ncol;
        int send_procid = 0;
        int rec_procid = 0;
        if (negative) {
            if (Muesli::proc_id == 0) {
                send_procid = msl::Muesli::num_total_procs - 1;
            } else {
                send_procid = Muesli::proc_id - 1;
            }
            MSL_ISend(send_procid, switchPartition, req, send_size, msl::MYTAG);
        } else {
            if (Muesli::proc_id == msl::Muesli::num_total_procs - 1) {
                send_procid = 0;
            } else {
                send_procid = Muesli::proc_id + 1;
            }
            MSL_ISend(send_procid, switchPartition, req, send_size, msl::MYTAG);
        }

        // Blocking receive.
        if (negative) {
            if (Muesli::proc_id == msl::Muesli::num_total_procs - 1) {
                rec_procid = 0;
            } else {
                rec_procid = Muesli::proc_id + 1;
            }
            MSL_Recv(rec_procid, buffer, stat, send_size, msl::MYTAG);
        } else {
            if (Muesli::proc_id == 0) {
                rec_procid = msl::Muesli::num_total_procs - 1;
            } else {
                rec_procid = Muesli::proc_id - 1;
            }

            MSL_Recv(rec_procid, buffer, stat, send_size, msl::MYTAG);
        }

        // Wait for completion.
        if (Muesli::proc_id < Muesli::num_local_procs - 1) {
            MPI_Wait(&req, &stat);
        }
        for (int j = 0; j < howmuch * ncol; j++) {
            switchPartition[j] = buffer[j];
        }
    } else {
        if (negative) {
            for (int j = 0; j < howmuch; j++) {
                for (int i = 0; i < ncol; i++) {
                    switchPartition[i + (j * ncol)] = doublePartition[i + (j * ncol)];
                }
            }
        } else {
            for (int j = 0; j < howmuch; j++) {
                for (int i = 0; i < ncol; i++) {
                    switchPartition[i + (j * ncol)] = doublePartition[i + (((nlocalRows - howmuch) + j) * ncol)];
                }
            }
        }
    }
    // TODO Depending on negative we need to write the "lower" or upper elements into the free space written to the buffer.
    for (int k = 0; k < nlocalRows; k++) {
        if (negative) {
            if (k >= nlocalRows - howmuch) {
                for (int i = 0; i < ncol; i++) {
                   this->localPartition[i + (k * ncol)] = switchPartition[i + ((k - (nlocalRows - howmuch)) * ncol)];
                }
            } else {
                for (int i = 0; i < ncol; i++) {
                   this->localPartition[i + (k * ncol)] = doublePartition[i + ((k + howmuch) * ncol)];
                }
                //take row from local partition
            }
        } else {
            if (k < howmuch) {
                for (int i = 0; i < ncol; i++) {
                    //printf("%d;", switchPartition[i + (k * ncol)]);
                   this->localPartition[i + (k * ncol)] = switchPartition[i + (k * ncol)];
                }
                // take the row from switch
            } else {
                for (int i = 0; i < ncol; i++) {
                   this->localPartition[i + (k * ncol)] = doublePartition[i + ((k - howmuch) * ncol)];
                }
            }
        }
    }

    this->updateDevice() ;
}

/* rotateCols
 * A rowwise distribution is assumed. Otherwise not working.
 * 1 2 3 4    -1    2 3 4 1      2    4 1 2 3
 * 5 6 7 8  ------> 6 7 8 5   ------> 8 5 6 7
 */
template<typename T>
void msl::DM<T>::rotateCols(int a) {
    bool negative = a < 0;
    int howmuch = a;
    if (negative) {
        howmuch = -1 * a;
    }
    if (howmuch >= ncol) {
        howmuch = howmuch % ncol;
        a = a % ncol;
    }
    if (howmuch == ncol || a == 0) {
        // if the number to rotate is equal to rows data stays the same.
        return;
    }
    // easy approach put all to cpu.
    T *doublePartition = new T[this->nLocal];
    this->updateDevice() ;
    for (int i = 0; i < this->nLocal ; i++) {
        doublePartition[i] =this->localPartition[i];
    }
    // easy case iterate where we are.
    if (rowComplete) {
        for (int i = 0; i < nrow; i++) {
            for (int j = 0; j < ncol; j++) {
                int colelement = (j + a);
                if (colelement < 0) {
                    colelement = ncol + colelement;
                }
                if (colelement >= ncol) {
                    int newcolelement = colelement - ncol;
                    colelement = newcolelement;
                }
               this->localPartition[(i * ncol) + j] = doublePartition[(i * ncol) + (colelement)];
            }
        }
    } else {
        if (msl::isRootProcess()) {
            throws(detail::RotateColCompleteNotImplementedException());
        }
        return;
    }
    this->updateDevice() ;

}

template<typename T>
template<typename T2, typename MapStencilFunctor, typename NeutralValueFunctor>
void msl::DM<T>::mapStencilMM(msl::DM<T2> &result, MapStencilFunctor &f, NeutralValueFunctor &neutral_value_functor) {

}
