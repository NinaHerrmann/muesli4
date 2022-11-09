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
#include <chrono>
#include <iostream>
#include <dm.h>


template<typename T>
msl::DM<T>::DM()
        :       // distributed array (resides on GPUs until deleted!)
        n(0), // number of elements of distributed array
        ncol(0), nrow(0), nLocal(0), // number of local elements on a node
        np(0),             // number of (MPI-) nodes (= Muesli::num_local_procs)
        id(0),             // id of local node among all nodes (= Muesli::proc_id)
        localPartition(0), // local partition of the DM
        cpuMemoryInSync(false), // is GPU memory in sync with CPU?
        firstIndex(0), // first global index of the DM on the local partition
        firstRow(0),   // first golbal row index of the DM on the local partition
        plans(0),      // GPU execution plans
        dist(Distribution::DIST), // distribution of DM: DIST (distributed) or
        // COPY (for now: always DIST)
        gpuCopyDistributed(0),     // is GPU copy distributed? (for now: always "false")
        // new: for combined usage of CPU and GPUs on every MPI-node
        ng(0),      // number of GPUs per node (= Muesli::num_gpus)
        nGPU(0),    // number of elements per GPU (all the same!)
        nCPU(0),    // number of elements on CPU = nLocal - ng*nGPU
        indexGPU(0) // number of elements on CPU = nLocal - ng*nGPU
{}

// constructor creates a non-initialized DM
template<typename T>
msl::DM<T>::DM(int row, int col) : n(col * row), ncol(col), nrow(row) {
    nLocal = n / Muesli::num_total_procs;
    if (nLocal % ncol == 0) {
        rowComplete = true;
    } else {
        rowComplete = false;
    }
    init();
#ifdef __CUDACC__

    (cudaMallocHost(&localPartition, nLocal * sizeof(T)));
    initGPUs();
#else
    localPartition = new T[nLocal];
#endif
    cpuMemoryInSync = false;
}

template<typename T>
msl::DM<T>::DM(int row, int col, bool rowComplete)
        : n(col * row), ncol(col), nrow(row), rowComplete(rowComplete) {
    init();
#ifdef __CUDACC__
    (cudaMallocHost(&localPartition, nLocal * sizeof(T)));
    initGPUs();
#else
    localPartition = new T[nLocal];
#endif
    cpuMemoryInSync = false;
}

// constructor creates a DM, initialized with v
template<typename T>
msl::DM<T>::DM(int row, int col, const T &v)
        : n(col * row), ncol(col), nrow(row) {
    nLocal = n / Muesli::num_total_procs;
    if (nLocal % ncol == 0) {
        rowComplete = true;
    } else {
        rowComplete = false;
    }
    init();

#ifdef __CUDACC__
    // TODO die CPU Elemente brauchen wir nicht unbedingt.
    (cudaMallocHost(&localPartition, nLocal * sizeof(T)));
    initGPUs();
#else
    localPartition = new T[nLocal];
#endif
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        localPartition[i] = v;
    }
    cpuMemoryInSync = true;
    updateDevice();
}

template<typename T>
msl::DM<T>::DM(int row, int col, const T &v, bool rowComplete)
        : n(col * row), ncol(col), nrow(row), rowComplete(rowComplete) {
    init();

#ifdef __CUDACC__
    // TODO die CPU Elemente brauchen wir nicht unbedingt.
    (cudaMallocHost(&localPartition, nLocal * sizeof(T)));
    initGPUs();
#else
    localPartition = new T[nLocal];
#endif
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++)
        localPartition[i] = v;
    cpuMemoryInSync = true;
    updateDevice();
}

template<typename T>
msl::DM<T>::DM(const DM <T> &other)
        : id(other.id), n(other.n), ncol(other.ncol), nrow(other.nrow), nLocal(other.nLocal),
          nlocalRows(other.nlocalRows), firstIndex(other.firstIndex), firstRow(other.firstRow),
          np(other.np), cpuMemoryInSync(other.cpuMemoryInSync), plans{new GPUExecutionPlan<T>{
                *(other.plans)}}, gpuCopyDistributed(other.gpuCopyDistributed),
          ng(other.ng), nGPU(other.nGPU), nCPU(other.nCPU), indexGPU(other.indexGPU),
          rowComplete(other.rowComplete) {
    copyLocalPartition(other);

    cpuMemoryInSync = false;
    updateDevice();
}

template<typename T>
msl::DM<T>::DM(DM <T> &&other)
        : id(other.id), n(other.n), ncol(other.ncol), nrow(other.nrow), nLocal(other.nLocal),
          nlocalRows(other.nlocalRows), firstIndex(other.firstIndex), firstRow(other.firstRow), np(other.np),
          cpuMemoryInSync(other.cpuMemoryInSync), plans{other.plans},
          gpuCopyDistributed(other.gpuCopyDistributed), ng(other.ng),
          nGPU(other.nGPU), nCPU(other.nCPU), indexGPU(other.indexGPU),
          rowComplete(other.rowComplete) {
    other.plans = nullptr;
    localPartition = other.localPartition;
    other.localPartition = nullptr;
}

template<typename T>
msl::DM<T> &msl::DM<T>::operator=(DM <T> &&other) {
    if (&other == this) {
        return *this;
    }

    freeLocalPartition();

    localPartition = other.localPartition;
    other.localPartition = nullptr;
    freePlans();
    plans = other.plans;
    other.plans = nullptr;

    id = other.id;
    n = other.n;
    nLocal = other.nLocal;
    nlocalRows = other.nlocalRows;
    ncol = other.ncol;
    nrow = other.nrow;
    firstIndex = other.firstIndex;
    firstRow = other.firstRow;
    np = other.np;
    cpuMemoryInSync = other.cpuMemoryInSync;
    gpuCopyDistributed = other.gpuCopyDistributed;
    ng = other.ng;
    nGPU = other.nGPU;
    indexGPU = other.indexGPU;
    rowComplete = other.rowComplete;
    return *this;
}

template<typename T>
msl::DM<T> &msl::DM<T>::operator=(const DM <T> &other) {
    if (&other == this) {
        return *this;
    }
    freeLocalPartition();
    copyLocalPartition(other);
    freePlans();
    plans = nullptr;
    plans = new GPUExecutionPlan<T>{*(other.plans)};

    id = other.id;
    n = other.n;
    nLocal = other.nLocal;
    nlocalRows = other.nlocalRows;
    ncol = other.ncol;
    nrow = other.nrow;
    firstIndex = other.firstIndex;
    firstRow = other.firstRow;
    np = other.np;
    cpuMemoryInSync = other.cpuMemoryInSync;
    gpuCopyDistributed = other.gpuCopyDistributed;
    ng = other.ng;
    nGPU = other.nGPU;
    indexGPU = other.indexGPU;
    rowComplete = other.rowComplete;
    return *this;
}

template<typename T>
msl::DM<T>::~DM() {
#ifdef __CUDACC__
(cudaFreeHost(localPartition));
if (plans) {
    for (int i = 0; i < ng; i++) {
        if (plans[i].d_Data != 0) {
            cudaSetDevice(i);
            (cudaFree(plans[i].d_Data));
        }
    }
    delete[] plans;
}
#else
delete[] localPartition;
#endif
}

// ************************* Auxiliary Methods not to be called from Program ********/
template<typename T>
void msl::DM<T>::copyLocalPartition(const DM <T> &other) {
#ifdef __CUDACC__
    (cudaMallocHost(&localPartition, nLocal * sizeof(T)));

#else
    localPartition = new T[nLocal];
#endif
    for (int i = 0; i < nLocal; i++)
        localPartition[i] = other.localPartition[i];
}

template<typename T>
void msl::DM<T>::freeLocalPartition() {
#ifdef __CUDACC__
    (cudaFreeHost(localPartition));
#else
    delete[] localPartition;
    localPartition = nullptr;
#endif
}

template<typename T>
void msl::DM<T>::freePlans() {
#ifdef __CUDACC__
    if (plans) {
        for (int i = 0; i < ng; i++) {
            if (plans[i].d_Data != 0) {
                cudaSetDevice(i);
                (cudaFree(plans[i].d_Data));
            }
        }
        delete[] plans;
    }
#endif
}

template<typename T>
void msl::DM<T>::init() {
    if (Muesli::proc_entrance == UNDEFINED) {
        throws(detail::MissingInitializationException());
    }
    id = Muesli::proc_id;
    np = Muesli::num_total_procs;
    ng = Muesli::num_gpus;
    n = ncol * nrow;
    if (!rowComplete) {
        // TODO not working for uneven data structures e.g. 7*7 results in 2 * 24.
        // *1.0f required to get float.
        if (ceilf((n * 1.0f) / np) - (n * 1.0f / np) != 0) {
            // If we have an even number of processes just switch roundf and roundl.
            if (np % 2 == 0) {
                if (id % 2 == 0) {
                    nLocal = roundf((n * 1.0f) / np);
                } else {
                    nLocal = roundl((n * 1.0f) / np);
                }
            } else {
                // if they are not even e.g. 5 processes 49 element just put the leftovers to first process
                // maybe not the most precise way but easy. However, not really relevant as many operation
                // do not work with uneven data structures.
                if (id == 0) {
                    nLocal = (n / np) + ((n / np) % np);
                } else {
                    nLocal = n / np;
                }
                printf("ID \t %d nLocal Rows \t %d \n", id, nLocal);
            }
        } else {
            nLocal = n / np;
        }
        nlocalRows = nLocal % ncol == 0 ? nLocal / ncol : nLocal / ncol + 1;
    } else {
        nlocalRows = nrow / np;
        nLocal = nlocalRows * ncol;
    }
    // TODO: This could result in rows being split between GPUs.
    //  Can be problematic for stencil ops.
#ifdef __CUDACC__
    nGPU = ng > 0 ? nLocal * (1.0 - Muesli::cpu_fraction) / ng : 0;
    nCPU = nLocal - nGPU * ng; // [0, nCPU-1] elements will be handled by the CPU.
    indexGPU = nCPU;
#else
    // If GPU not used all Elements to CPU.
    nGPU = 0;
    nCPU = nLocal;
    indexGPU = nLocal;
#endif
    firstIndex = id * nLocal;
    firstRow = firstIndex / ncol;
}

template<typename T>
void msl::DM<T>::initGPUs() {
#ifdef __CUDACC__
    plans = new GPUExecutionPlan<T>[ng];
    int gpuBase = indexGPU;
    for (int i = 0; i < ng; i++) {
        cudaSetDevice(i);
        plans[i].size = nGPU;
        plans[i].nLocal = plans[i].size;
        plans[i].bytes = plans[i].size * sizeof(T);
        plans[i].first = gpuBase + firstIndex;
        plans[i].firstRow = plans[i].first / ncol;
        plans[i].firstCol = plans[i].first % ncol;
        plans[i].lastRow = (plans[i].first + plans[i].nLocal - 1) / ncol;
        plans[i].lastCol = (plans[i].first + plans[i].nLocal - 1) % ncol;
        plans[i].gpuRows = plans[i].lastRow - plans[i].firstRow + 1;
        if (plans[i].gpuRows > 2) {
            plans[i].gpuCols = ncol;
        } else if (plans[i].gpuRows == 2) {
            if (plans[i].lastCol >= plans[i].firstCol) {
                plans[i].gpuCols = ncol;
            } else {
                plans[i].gpuCols = ncol - (plans[i].firstCol - plans[i].lastCol);
            }
        } else if (plans[i].gpuRows > 0) {
            plans[i].gpuCols = plans[i].lastCol - plans[i].firstCol;
        }
        plans[i].h_Data = localPartition + gpuBase;
        (cudaMalloc(&plans[i].d_Data, plans[i].bytes));
        gpuBase += plans[i].size;
    }
#endif
}

template<typename T>
T *msl::DM<T>::getLocalPartition() {
    if (!cpuMemoryInSync)
        updateHost();
    return localPartition;
}

template<typename T>
void msl::DM<T>::fill(const T &element) {
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        localPartition[i] = element;
    }
    cpuMemoryInSync = true;
    updateDevice();
}

template<typename T>
void msl::DM<T>::fill(T *const elements) {
    localPartition = elements;
    cpuMemoryInSync = true;
    updateDevice();
}
template<typename T>
T msl::DM<T>::get(int index) const {
    int idSource;
    T message;

    if (isLocal(index)) {
        if (index < firstIndex + nCPU || cpuMemoryInSync) {
            message = localPartition[index - firstIndex];
        } else {
#ifdef __CUDACC__
            int device = getGpuId(index);
            cudaSetDevice(device);
            int offset = index - plans[device].first;
            (cudaMemcpyAsync(&message, plans[device].d_Data + offset,
                             sizeof(T), cudaMemcpyDeviceToHost,
                             Muesli::streams[device]));
#endif
        }
        idSource = Muesli::proc_id;
    }
        // Element with global index is not locally stored
    else {
        // Calculate id of the process that stores the element locally
        idSource = (int) (index / nLocal);
    }

    msl::MSL_Broadcast(idSource, &message, 1);
    return message;
}

template<typename T>
T msl::DM<T>::get2D(int row, int col) const {
    int index = (row) * ncol + col;
    return get(index);
}

template<typename T>
int msl::DM<T>::getSize() const { return n; }

template<typename T>
int msl::DM<T>::getLocalSize() const { return nLocal; }

template<typename T>
int msl::DM<T>::getFirstIndex() const {
    return firstIndex;
}

template<typename T>
void msl::DM<T>::setCpuMemoryInSync(bool b) {
    cpuMemoryInSync = b;
}

template<typename T>
bool msl::DM<T>::isLocal(int index) const {
    return (index >= firstIndex) && (index < firstIndex + nLocal);
}

template<typename T>
T msl::DM<T>::getLocal(int localIndex) {
    if (localIndex >= nLocal)
        throws(detail::NonLocalAccessException());
    if ((!cpuMemoryInSync) && (localIndex >= nCPU)) {
#ifdef __CUDACC__
        int device = getGpuId(localIndex);
        cudaSetDevice(device);
        int offset = localIndex - plans[device].first;
        T message;
        (cudaMemcpyAsync(&message, plans[device].d_Data + offset,
                         sizeof(T), cudaMemcpyDeviceToHost,
                         Muesli::streams[device]));
        return message;
#endif
    }
    return localPartition[localIndex];
}

template<typename T>
T &msl::DM<T>::operator[](int index) {
    return get(index);
}

template<typename T>
void msl::DM<T>::setLocal(int localIndex, const T &v) {
    if (localIndex < nCPU) {
        localPartition[localIndex] = v;
    } else if (localIndex >= nLocal)
        throws(detail::NonLocalAccessException());
    else { // localIndex refers to a GPU
#ifdef __CUDACC__
        int gpuId = (localIndex - nCPU) / nGPU;
        int idx = localIndex - nCPU - gpuId * nGPU;
        cudaSetDevice(gpuId);
        (cudaMemcpy(&(plans[gpuId].d_Data[idx]), &v, sizeof(T),
                                     cudaMemcpyHostToDevice));
#endif
    }
}

template<typename T>
void msl::DM<T>::set(int globalIndex, const T &v) {
    if ((globalIndex >= firstIndex) && (globalIndex < firstIndex + nLocal)) {
        setLocal(globalIndex - firstIndex, v);
    }
    // TODO: Set global
}

template<typename T>
GPUExecutionPlan<T> *msl::DM<T>::getExecPlans() {
    return plans;
}

template<typename T>
void msl::DM<T>::updateDevice() {
#ifdef __CUDACC__
    if (cpuMemoryInSync) {
        for (int i = 0; i < ng; i++) {
            cudaSetDevice(i);
            // upload data to host
            (cudaMemcpyAsync(plans[i].d_Data, plans[i].h_Data,
                                              plans[i].bytes, cudaMemcpyHostToDevice,
                                              Muesli::streams[i]));
        }

        for (int i = 0; i < ng; i++) {
            (cudaStreamSynchronize(Muesli::streams[i]));
        }
    }
#endif
}

template<typename T>
void msl::DM<T>::updateHost() {
#ifdef __CUDACC__
    if (!cpuMemoryInSync) {
        for (int i = 0; i < ng; i++) {
            cudaSetDevice(i);
            // updateDevice data from device
            (cudaMemcpyAsync(plans[i].h_Data, plans[i].d_Data,
                                              plans[i].bytes, cudaMemcpyDeviceToHost,
                                              Muesli::streams[i]));
        }

        // wait until updateDevice is finished
        for (int i = 0; i < ng; i++) {
            (cudaStreamSynchronize(Muesli::streams[i]));
        }
        cpuMemoryInSync = true;
    }
#endif
}

template<typename T>
int msl::DM<T>::getGpuId(int index) const {
    return (index - firstIndex - nCPU) / nGPU;
}


template<typename T>
void msl::DM<T>::showLocal(const std::string &descr) {
    if (!cpuMemoryInSync) {
        updateHost();
    }
    if (msl::isRootProcess()) {
        std::ostringstream s;
        if (descr.size() > 0)
            s << descr << ": ";
        s << "[";
        for (int i = 0; i < nLocal; i++) {
            s << localPartition[i] << " ";
        }
        s << "]" << std::endl;
        printf("%s", s.str().c_str());
    }
}

template<typename T>
void msl::DM<T>::show(const std::string &descr) {
#ifdef __CUDACC__
    cudaDeviceSynchronize();
#endif
    T *b = new T[n];
    std::cout.precision(2);
    std::ostringstream s;
    if (descr.size() > 0)
        s << descr << ": " << std::endl;
    if (!cpuMemoryInSync) {
        updateHost();
    }
    msl::allgather(localPartition, b, nLocal);

    if (msl::isRootProcess()) {
        s << "[";
        for (int i = 0; i < n - 1; i++) {
            s << b[i];
            ((i + 1) % ncol == 0) ? s << "\n " : s << " ";;
        }
        s << b[n - 1] << "]" << std::endl;
        s << std::endl;
    }

    delete[] b;

    if (msl::isRootProcess())
        printf("%s", s.str().c_str());
}

// SKELETONS / COMMUNICATION / BROADCAST PARTITION

template<typename T>
void msl::DM<T>::broadcastPartition(int partitionIndex) {
    if (partitionIndex < 0 || partitionIndex >= np) {
        throws(detail::IllegalPartitionException());
    }
    if (!cpuMemoryInSync)
        updateDevice();
    msl::MSL_Broadcast(partitionIndex, localPartition, nLocal);
    cpuMemoryInSync = false;
    updateDevice();
}

// SKELETONS / COMMUNICATION / GATHER
template<typename T>
T *msl::DM<T>::gather() {
    T *b = new T[n];
    std::cout.precision(2);
    std::ostringstream s;
#ifdef __CUDACC__
    if (!cpuMemoryInSync) {
        updateHost();
    }
#endif
    msl::allgather(localPartition, b, nLocal);
    return b;
}

template<typename T>
void msl::DM<T>::gather(msl::DM<T> &da) {
    size_t rec_bytes = nLocal * sizeof(T); // --> received per process
    if (msl::isRootProcess()) {
        MPI_Gather(localPartition, rec_bytes, MPI_BYTE, localPartition, rec_bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(localPartition, rec_bytes, MPI_BYTE, NULL, 0, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
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
    if (!cpuMemoryInSync)
      updateDevice();
    T* buffer = new T[nLocal];
    for (i = 0; i < nLocal; i++) {
      buffer[i] = localPartition[i];
    }
    MPI_Status stat;
    MPI_Request req;
    MSL_ISend(receiver, buffer, req, nLocal, msl::MYTAG);
    MSL_Recv(sender, localPartition, stat, nLocal, msl::MYTAG);
    MPI_Wait(&req, &stat);
    delete[] buffer;
    cpuMemoryInSync = false;
    updateHost();
  }
}
*/
// template<typename T>
// inline void msl::DM<T>::permutePartition(int (*f)(int)) {
//  permutePartition(curry(f));
//}

template<typename T>
void msl::DM<T>::freeDevice() {
#ifdef __CUDACC__
    if (!cpuMemoryInSync) {
        for (int i = 0; i < ng; i++) {
            if(plans[i].d_Data == 0) {
            continue;
            }
            cudaFree(plans[i].d_Data);
            plans[i].d_Data = 0;
        }
        cpuMemoryInSync = true;
    }
#endif
}


//*********************************** Maps ********************************

template<typename T>
template<typename MapFunctor>
void msl::DM<T>::mapInPlace(MapFunctor &f) {
    updateDevice();

#ifdef __CUDACC__
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        detail::mapKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                plans[i].d_Data, plans[i].d_Data, plans[i].size,
                        f); // in, out, #bytes, function
    }
#endif

#pragma omp parallel for
    for (int k = 0; k < nCPU; k++) {
        localPartition[k] = f(localPartition[k]);
    }
    msl::syncStreams();
    cpuMemoryInSync = false;
}

template<typename T>
template<typename MapIndexFunctor>
void msl::DM<T>::mapIndexInPlace(MapIndexFunctor &f) {
    updateDevice();

#ifdef __CUDACC__

    for (int i = 0; i < ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        detail::mapIndexKernelDM<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                plans[i].d_Data, plans[i].d_Data, plans[i].nLocal, plans[i].first, f,
                        ncol);
    }
#endif
// all necessary calculations are performed otherwise some are skipped.
#pragma omp parallel for
    for (int k = 0; k < nCPU; k++) {
        int i = (k + firstIndex) / ncol;
        int j = (k + firstIndex) % ncol;
        localPartition[k] = f(i, j, localPartition[k]);
    }
    // check for errors during gpu computation
    msl::syncStreams();

    cpuMemoryInSync = false;
}
template<typename T>
template<typename F>
msl::DM<T> msl::DM<T>::mapComp(F &f) {        // preliminary simplification in order to avoid type error
    DM<T> result(nrow, ncol);
    updateDevice();
#ifdef __CUDACC__

    // map
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        detail::mapKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                plans[i].d_Data,  result.getExecPlans()[i].d_Data, plans[i].size, f);
    }
#endif
#pragma omp parallel for
    for (int k = 0; k < nCPU; k++) {
        result.setLocal(k, f(localPartition[k]));
    }

    // check for errors during gpu computation
    msl::syncStreams();
    result.setCpuMemoryInSync(false);
    return result;
}
template<typename T>
template<typename F>
void msl::DM<T>::map(F &f, DM<T> &result) {        // preliminary simplification in order to avoid type error
    updateDevice();
#ifdef __CUDACC__

    // map
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        detail::mapKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                plans[i].d_Data,  result.getExecPlans()[i].d_Data, plans[i].size, f);
    }
#endif
#pragma omp parallel for
    for (int k = 0; k < nCPU; k++) {
        result.setLocal(k, f(localPartition[k]));
    }

    // check for errors during gpu computation
    msl::syncStreams();
    result.setCpuMemoryInSync(false);
}

template<typename T>
template<typename MapIndexFunctor>
void msl::DM<T>::mapIndex(MapIndexFunctor &f, DM<T> &result) {
    updateDevice();
#ifdef __CUDACC__

    // map on GPUs
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        detail::mapIndexKernelDM<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].nLocal,
                        plans[i].first, f, ncol);
    }
#endif
#pragma omp parallel for
    for (int k = 0; k < nCPU; k++) {
        int j = (k + firstIndex) / ncol;
        int i = (k + firstIndex) % ncol;
        result.setLocal(k, f(i, j, localPartition[k]));
    }

    // check for errors during gpu computation
    msl::syncStreams();
    result.setCpuMemoryInSync(false);
}

// ************************************ zip *************************************
template<typename T>
template<typename T2, typename ZipFunctor>
void msl::DM<T>::zipInPlace(DM <T2> &b, ZipFunctor &f) {
    // zip on GPU
    updateDevice();
#ifdef __CUDACC__

    for (int i = 0; i < ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        auto bplans = b.getExecPlans();
        detail::zipKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                plans[i].d_Data, bplans[i].d_Data, plans[i].d_Data, plans[i].size, f);
    }
#endif
    T2 *bPartition = b.getLocalPartition();
#pragma omp parallel for
    for (int k = 0; k < nCPU; k++) {
        localPartition[k] = f(localPartition[k], bPartition[k]);
    }

    // check for errors during gpu computation
    msl::syncStreams();
    cpuMemoryInSync = false;
}

template<typename T>
template<typename T2, typename ZipFunctor>
void msl::DM<T>::zip(DM<T2> &b, DM<T> &result,
                ZipFunctor &f) { // should have result type DA<R>; debug
    updateDevice();

    // zip on GPUs
#ifdef __CUDACC__

    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        detail::zipKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                plans[i].d_Data, b.getExecPlans()[i].d_Data,
                        result.getExecPlans()[i].d_Data, plans[i].size, f);
    }
#endif
    // zip on CPU cores
    T2 *bPartition = b.getLocalPartition();
#pragma omp parallel for
    for (int k = 0; k < nCPU; k++) {
        result.setLocal(k, f(localPartition[k], bPartition[k]));
    }
    // check for errors during gpu computation
    msl::syncStreams();
    result.setCpuMemoryInSync(false);
}

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DM<T>::zipIndexInPlace(DM <T2> &b, ZipIndexFunctor &f) {
    updateDevice();
#ifdef __CUDACC__
    for (int i = 0; i < ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        detail::zipIndexKernelDM<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                plans[i].d_Data, b.getExecPlans()[i].d_Data, plans[i].d_Data,
                        plans[i].nLocal, plans[i].first, f, ncol);
    }
#endif
    if (nCPU > 0) {
        T2 *bPartition = b.getLocalPartition();
#pragma omp parallel for
        for (int k = 0; k < nCPU; k++) {
            int i = (k + firstIndex) / ncol;
            int j = (k + firstIndex) % ncol;
            localPartition[k] = f(i, j, localPartition[k], bPartition[k]);
        }
    }
    // check for errors during gpu computation
    cpuMemoryInSync = false;
    msl::syncStreams();

}

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DM<T>::zipIndex(DM <T2> &b, DM<T2> &result, ZipIndexFunctor &f) {
    updateDevice();

    // zip on GPUs
#ifdef __CUDACC__

    for (int i = 0; i < ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        detail::zipIndexKernelDM<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                plans[i].d_Data, b.getExecPlans()[i].d_Data,
                        result.getExecPlans()[i].d_Data, plans[i].nLocal, plans[i].first, f,
                        ncol);
    }
#endif
    // zip on CPU cores
    T2 *bPartition = b.getLocalPartition();
#pragma omp parallel for
    for (int k = 0; k < nCPU; k++) {
        int i = (k + firstIndex) / ncol;
        int j = (k + firstIndex) % ncol;
        result.setLocal(k, f(i, j, localPartition[k], bPartition[k]));
    }
    // check for errors during gpu computation
    msl::syncStreams();
    result.setCpuMemoryInSync(false);
}

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DM<T>::crossZipIndexInPlace(DM <T2> &b, ZipIndexFunctor &f) {
    updateDevice();

    // zip on GPUs
#ifdef __CUDACC__
for (int i = 0; i < ng; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    //grid_size = (plans[i].size/block_size) + (!(Size%block_size)? 0:1);
    dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
    detail::crossZipInPlaceIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
            plans[i].d_Data, b.getExecPlans()[i].d_Data,
            plans[i].nLocal, plans[i].first, f, ncol);
}
#endif
if (nCPU > 0) {
    T2 *bPartition = b.getLocalPartition();
#pragma omp parallel for
    for (int k = 0; k < nCPU; k++) {
        int i = (k + firstIndex) / ncol;
        int j = (k + firstIndex) % ncol;
        localPartition[k] = f(i, j, localPartition[k], bPartition);
    }
}
// check for errors during gpu computation
cpuMemoryInSync = false;
}

template<typename T>
template<typename T2, typename T3, typename ZipFunctor>
void msl::DM<T>::zipInPlace3(DM <T2> &b, DM <T3> &c, ZipFunctor &f) {
updateDevice();
#ifdef __CUDACC__

    for (int i = 0; i < ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        auto bplans = b.getExecPlans();
        auto cplans = c.getExecPlans();
        detail::zipKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                plans[i].d_Data, bplans[i].d_Data, cplans[i].d_Data, plans[i].d_Data,
                        plans[i].nLocal, f);
    }
#endif
    T2 *bPartition = b.getLocalPartition();
    T3 *cPartition = c.getLocalPartition();
#pragma omp parallel for
    for (int k = 0; k < nCPU; k++) {
        localPartition[k] = f(localPartition[k], bPartition[k], cPartition[k]);
    }

    // check for errors during gpu computation
    msl::syncStreams();
    cpuMemoryInSync = false;
}


// *********** fold *********************************************
#ifdef __CUDACC__
template<typename T>
template<typename FoldFunctor>
T msl::DM<T>::fold(FoldFunctor &f, bool final_fold_on_cpu) {
    updateHost();

    std::vector<int> blocks(Muesli::num_gpus);
    std::vector<int> threads(Muesli::num_gpus);
    T *gpu_results = new T[Muesli::num_gpus];
    int maxThreads = 1024; // preliminary
    int maxBlocks = 1024;  // preliminary
    for (int i = 0; i < Muesli::num_gpus; i++) {
        threads[i] = maxThreads;
        gpu_results[i] = 0;
    }
    T *local_results = new T[np];
    T **d_odata = new T *[Muesli::num_gpus];
    updateDevice();

    //
    // Step 1: local fold
    //

    // prearrangement: calculate threads, blocks, etc.; allocate device memory
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        threads[i] = (plans[i].size < maxThreads)
                ? static_cast<int>(detail::nextPow2((plans[i].size + 1) / 2))
                     : maxThreads;
        blocks[i] = plans[i].size / threads[i];
        if (blocks[i] > maxBlocks) {
            blocks[i] = maxBlocks;
        }
        (cudaMalloc((void **) &d_odata[i], blocks[i] * sizeof(T)));
    }

    // fold on gpus: step 1
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        detail::reduce<T, FoldFunctor>(plans[i].size, plans[i].d_Data, d_odata[i],
                                       threads[i], blocks[i], f, Muesli::streams[i],
                                       i);
    }
    // fold local elements on CPU (overlap with GPU computations)
    // TODO: openmp has parallel reduce operators.
    T cpu_result = 0;

    if (nCPU > 0){
        cpu_result = localPartition[0];
        for (int k = 1; k < nCPU; k++) {
            cpu_result = f(cpu_result, localPartition[k]);
        }
    }

    msl::syncStreams();

    // fold on gpus: step 2
    for (int i = 0; i < Muesli::num_gpus; i++) {
        if (blocks[i] > 1) {
            cudaSetDevice(i);
            int threads = (static_cast<int>(detail::nextPow2(blocks[i])) == blocks[i])
                          ? blocks[i]
                          : detail::nextPow2(blocks[i]) / 2;
            detail::reduce<T, FoldFunctor>(blocks[i], d_odata[i], d_odata[i], threads,
                                           1, f, Muesli::streams[i], i);
        }
    }
    msl::syncStreams();

    // copy final sum from device to host
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        (cudaMemcpyAsync(&gpu_results[i], d_odata[i], sizeof(T),
                                          cudaMemcpyDeviceToHost,
                                          Muesli::streams[i]));
    }
    msl::syncStreams();

    //
    // Step 2: global fold
    //

    T final_result, result;
    if (final_fold_on_cpu) {
        // calculate local result for all GPUs and CPU
        T tmp = cpu_result;
        for (int i = 0; i < Muesli::num_gpus; i++) {
            tmp = f(tmp, gpu_results[i]);
        }

        // gather all local results
        msl::allgather(&tmp, local_results, 1);

        // calculate global result from local results
        result = local_results[0];
        for (int i = 1; i < np; i++) {
            result = f(result, local_results[i]);
        }
        final_result = result;
    } else {

        T local_result;
        T *d_gpu_results;
        if (Muesli::num_gpus > 1) { // if there is more than 1 GPU
            cudaSetDevice(0);         // calculate local result on device 0

            // updateHost data
            (
                    cudaMalloc((void **) &d_gpu_results, Muesli::num_gpus * sizeof(T)));
            (cudaMemcpyAsync(
                    d_gpu_results, gpu_results, Muesli::num_gpus * sizeof(T),
                    cudaMemcpyHostToDevice, Muesli::streams[0]));
            (cudaStreamSynchronize(Muesli::streams[0]));

            // final (local) fold
            detail::reduce<T, FoldFunctor>(Muesli::num_gpus, d_gpu_results,
                                           d_gpu_results, Muesli::num_gpus, 1, f,
                                           Muesli::streams[0], 0);
            (cudaStreamSynchronize(Muesli::streams[0]));

            // copy result from device to host
            (cudaMemcpyAsync(&local_result, d_gpu_results, sizeof(T),
                                              cudaMemcpyDeviceToHost,
                                              Muesli::streams[0]));
            (cudaStreamSynchronize(Muesli::streams[0]));
            (cudaFree(d_gpu_results));
        } else {
            local_result = gpu_results[0];
        }

        if (np > 1) {
            // gather all local results
            msl::allgather(&local_result, local_results, 1);

            // calculate global result from local results
            // updateHost data
            (cudaMalloc((void **) &d_gpu_results, np * sizeof(T)));
            (cudaMemcpyAsync(d_gpu_results, local_results,
                                              np * sizeof(T), cudaMemcpyHostToDevice,
                                              Muesli::streams[0]));

            // final fold
            detail::reduce<T, FoldFunctor>(np, d_gpu_results, d_gpu_results, np, 1, f,
                                           Muesli::streams[0], 0);
            (cudaStreamSynchronize(Muesli::streams[0]));

            // copy final result from device to host
            (cudaMemcpyAsync(&final_result, d_gpu_results, sizeof(T),
                                              cudaMemcpyDeviceToHost,
                                              Muesli::streams[0]));
            (cudaStreamSynchronize(Muesli::streams[0]));
            cudaFree(d_gpu_results);
        } else {
            final_result = local_result;
        }
    }

    // Cleanup
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(Muesli::streams[i]);
        cudaFree(d_odata[i]);
    }
    delete[] gpu_results;
    delete[] d_odata;
    delete[] local_results;

    return final_result;
}
#else

template<typename T>
template<typename FoldFunctor>
T msl::DM<T>::fold(FoldFunctor &f, bool final_fold_on_cpu) {
    updateHost();
    T localresult = 0;
#pragma omp parallel for shared(localPartition) reduction(+: localresult)
    for (int i = 0; i < nLocal; i++) {
        localresult = f(localresult, localPartition[i]);
    }
    T *local_results = new T[np];
    T tmp = localresult;

    msl::allgather(&tmp, local_results, 1);
    T global_result = local_results[0];
#pragma omp parallel for shared(local_results) reduction(+: global_result)
    for (int i = 1; i < np; i++) {
        global_result = f(global_result, local_results[i]);
    }
    // TODO MPI global result
    return global_result;
}

#endif



#ifdef __CUDACC__
template<typename T>
void msl::DM<T>::downloadupperpart(int paddingsize) {
    cudaSetDevice(0);

    // updateDevice data from device
    (cudaMemcpyAsync(plans[0].h_Data, plans[0].d_Data,
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
    (cudaMemcpyAsync(plans[gpu].h_Data + (plans[gpu].nLocal - paddingsize), plans[gpu].d_Data + (plans[gpu].nLocal-paddingsize),
                     paddingsize * sizeof(T), cudaMemcpyDeviceToHost,
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
    (cudaMemcpyAsync(plans[gpu].h_Data + (plans[gpu].nLocal - paddingsize), plans[gpu].d_Data + (plans[gpu].nLocal-paddingsize),
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

    // TODO

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
            (cudaMalloc(&all_data[i], ((plans[i].gpuRows+kw) * (plans[i].gpuCols+kw)) * sizeof(T)));
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
            T *buffer = localPartition + (nlocalRows - stencil_size) * ncol;
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
            MSL_ISend(Muesli::proc_id - 1, localPartition, req, col_size,
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

    // TODO copy first GPU row to localPartition
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
            cudaMemcpy(d_dm[i] + stencil_size, plans[i - 1].d_Data + (plans[i].nLocal - (stencil_size * ncol)),
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
        cudaMemcpyAsync(all_data[i]+((plans[i].gpuRows+stencil_size)*(plans[i].gpuCols+kw)), padding_stencil + padding_size,
                        padding_size * sizeof(T), cudaMemcpyHostToDevice, Muesli::streams[i]);
       dim3 fillthreads(tile_width, tile_width);
       dim3 fillblocks((plans[i].gpuRows + fillthreads.x - 1) / fillthreads.x, (plans[i].gpuCols + fillthreads.y - 1) / fillthreads.y);
       // int paddingoffset, int gpuRows, int ss
       if (Muesli::debug) {
           gpuErrchk(cudaPeekAtLastError());
           gpuErrchk(cudaDeviceSynchronize());
       }
       detail::fillcore<<<fillblocks,fillthreads, 0, Muesli::streams[i]>>>(all_data[i], plans[i].d_Data, stencil_size * (plans[i].gpuCols + (2*stencil_size)), plans[i].gpuCols, stencil_size);
        if (Muesli::debug) {
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }
        plm.addDevicePtr(all_data[i]);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < msl::Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        plm.setFirstRowGPU(plans[i].firstRow);
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
        f.init(plans[i].gpuRows, plans[i].gpuCols, plans[i].firstRow,
               plans[i].firstCol);
        f.notify();

        cudaSetDevice(i);
        int divisor = 1;
        if (f.getSharedMemory()) {
            dim3 dimBlock(tile_width, tile_width);
            int kw = stencil_size * 2;
            if (plans[i].gpuRows % tile_width != 0){
                //printf("\nRight now number of rows must be dividable by tile width\n");
            }
            if (plans[i].gpuCols % tile_width != 0) {
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
            dim3 dimGrid(((plans[i].gpuRows))/divisor/dimBlock.x,
                         (plans[i].gpuCols + dimBlock.y - 1) / dimBlock.y);
            smem_size = ((divisor * tile_width) + kw) * (tile_width + kw) * sizeof(T);
            //printf("\n %d %d; %d %d \n\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
            //printf("Rows %d Cols %d %d %d %d %d \n", plans[i].gpuRows, plans[i].gpuCols, dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
            detail::mapStencilMMKernel<<<dimGrid, dimBlock, smem_size, Muesli::streams[i]>>>(
                    result.getExecPlans()[i].d_Data, plans[i].gpuRows, plans[i].gpuCols, plans[i].firstCol, plans[i].firstRow, vplm[i],all_data[i], f, tile_width, divisor, kw);
            if (Muesli::debug) {
                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }
        }
        if (!f.getSharedMemory()){
            //dim3 dimBlock(Muesli::threads_per_block);
            //dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
            dim3 dimBlock(tile_width, tile_width);

            divisor = Muesli::reps;

            dim3 dimGrid(((plans[i].gpuRows))/divisor,
                         (plans[i].gpuCols + dimBlock.y - 1) / dimBlock.y);
            detail::mapStencilGlobalMem_rep<<<dimGrid, dimBlock, smem_size, Muesli::streams[i]>>>(
                    result.getExecPlans()[i].d_Data, plans[i], vplm[i], f, i, divisor, tile_width);
            if (Muesli::debug) {
                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }
        }
        cudaDeviceSynchronize();
    }

    f.notify();
    if (nCPU != 0) {
        if (Muesli::debug)
            printf("Calculating %d Elements on the CPU ... \n", nCPU);
        #pragma omp parallel for
        for (int i = 0; i < nCPU; i++) {
            // TODO CPU PLM Matrix
            //result.setLocal(i, f(i / ncol + firstRow, i % ncol, localPartition, nrow, ncol));
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
    updateDevice();
    /* Depending on negative we need to write the "lower" or "upper" elements into the buffer
     * and write rows up or downwards.
    */
    T *doublePartition = new T[nLocal];

    for (int i = 0; i < nLocal; i++) {
        doublePartition[i] = localPartition[i];
    }
    if (negative) {
        for (int i = 0; i < howmuch * ncol; i++) {
            switchPartition[i] = localPartition[i];
        }
    } else {
        for (int i = 0; i < howmuch * ncol; i++) {
            switchPartition[i] = localPartition[nLocal - (howmuch * ncol) + i];
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
                    localPartition[i + (k * ncol)] = switchPartition[i + ((k - (nlocalRows - howmuch)) * ncol)];
                }
            } else {
                for (int i = 0; i < ncol; i++) {
                    localPartition[i + (k * ncol)] = doublePartition[i + ((k + howmuch) * ncol)];
                }
                //take row from local partition
            }
        } else {
            if (k < howmuch) {
                for (int i = 0; i < ncol; i++) {
                    //printf("%d;", switchPartition[i + (k * ncol)]);
                    localPartition[i + (k * ncol)] = switchPartition[i + (k * ncol)];
                }
                // take the row from switch
            } else {
                for (int i = 0; i < ncol; i++) {
                    localPartition[i + (k * ncol)] = doublePartition[i + ((k - howmuch) * ncol)];
                }
            }
        }
    }

    updateDevice();
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
    T *doublePartition = new T[nLocal];
    updateDevice();
    for (int i = 0; i < nLocal; i++) {
        doublePartition[i] = localPartition[i];
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
                localPartition[(i * ncol) + j] = doublePartition[(i * ncol) + (colelement)];
            }
        }
    } else {
        if (msl::isRootProcess()) {
            throws(detail::RotateColCompleteNotImplementedException());
        }
        return;
    }
    updateDevice();

}