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
#include <chrono>
#include <iostream>
#include <dc.h>


template<typename T>
msl::DC<T>::DC()
        :       // distributed array (resides on GPUs until deleted!)
        n(0), // number of elements of distributed array
        ncol(0), nrow(0), depth(0), nLocal(0), // number of local elements on a node
        np(0),             // number of (MPI-) nodes (= Muesli::num_local_procs)
        id(0),             // id of local node among all nodes (= Muesli::proc_id)
        localPartition(0), // local partition of the DC
        cpuMemoryInSync(false), // is GPU memory in sync with CPU?
        firstIndex(0), // first global index of the DC on the local partition
        firstRow(0),   // first golbal row index of the DC on the local partition
        plans(0),      // GPU execution plans
        dist(Distribution::DIST), // distribution of DC: DIST (distributed) or
        // COPY (for now: always DIST)
        gpuCopyDistributed(0),     // is GPU copy distributed? (for now: always "false")
        // new: for combined usage of CPU and GPUs on every MPI-node
        ng(0),      // number of GPUs per node (= Muesli::num_gpus)
        nGPU(0),    // number of elements per GPU (all the same!)
        nCPU(0),    // number of elements on CPU = nLocal - ng*nGPU
        indexGPU(0) // number of elements on CPU = nLocal - ng*nGPU
{}

// constructor creates a non-initialized DC
template<typename T>
msl::DC<T>::DC(int row, int col, int depth) : n(col * row), ncol(col), nrow(row), depth(depth) {
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
msl::DC<T>::DC(int row, int col, int depth, bool rowComplete)
: ncol(col), nrow(row), depth(depth), n(col * row * depth), rowComplete(rowComplete) {
    init();

#ifdef __CUDACC__
    (cudaMallocHost(&localPartition, nLocal * sizeof(T)));
    initGPUs();
#else
    localPartition = new T[nLocal];
#endif
    cpuMemoryInSync = false;
}

// constructor creates a DC, initialized with v
template<typename T>
msl::DC<T>::DC(int row, int col, int depth, const T &v)
: ncol(col), nrow(row), depth(depth), n(col * row * depth) {
    init();
    localPartition = new T[nLocal];

#ifdef __CUDACC__
    // TODO die CPU Elemente brauchen wir nicht unbedingt.
    (cudaMallocHost(&localPartition, nLocal * sizeof(T)));
    initGPUs();
#endif
#pragma omp parallel for
    for (int i = 0; i < nrow*ncol*depth; i++){
        localPartition[i] = v;
    }
#ifdef __CUDACC__
    initGPUs();
#endif
    upload();
}

template<typename T>
msl::DC<T>::DC(int row, int col, int depth, const T &v, bool rowComplete)
        : ncol(col), nrow(row), n(col * row * depth), rowComplete(rowComplete) {
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

    upload();
}

template<typename T>
msl::DC<T>::DC(const DC <T> &other)
        : id(other.id), n(other.n), nLocal(other.nLocal),
          nlocalRows(other.nlocalRows), ncol(other.ncol), nrow(other.nrow), depth(other.depth),
          firstIndex(other.firstIndex), firstRow(other.firstRow), np(other.np),
          cpuMemoryInSync(other.cpuMemoryInSync), plans{new GPUExecutionPlan<T>{
                *(other.plans)}},
          gpuCopyDistributed(other.gpuCopyDistributed), ng(other.ng),
          nGPU(other.nGPU), nCPU(other.nCPU), indexGPU(other.indexGPU),
          rowComplete(other.rowComplete) {
    copyLocalPartition(other);

    // cpuMemoryInSync = true;
    // upload();
}

template<typename T>
msl::DC<T>::DC(DC <T> &&other)
        : id(other.id), n(other.n), nLocal(other.nLocal), depth(other.depth),
          nlocalRows(other.nlocalRows), ncol(other.ncol), nrow(other.nrow),
          firstIndex(other.firstIndex), firstRow(other.firstRow), np(other.np),
          cpuMemoryInSync(other.cpuMemoryInSync), plans{other.plans},
          gpuCopyDistributed(other.gpuCopyDistributed), ng(other.ng),
          nGPU(other.nGPU), nCPU(other.nCPU), indexGPU(other.indexGPU),
          rowComplete(other.rowComplete) {
    other.plans = nullptr;
    localPartition = other.localPartition;
    other.localPartition = nullptr;
}

template<typename T>
msl::DC<T> &msl::DC<T>::operator=(DC <T> &&other) {
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
    depth = other.depth;
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
msl::DC<T> &msl::DC<T>::operator=(const DC <T> &other) {
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
    depth = other.depth;
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
void msl::DC<T>::copyLocalPartition(const DC <T> &other) {
#ifdef __CUDACC__
    (cudaMallocHost(&localPartition, nLocal * sizeof(T)));

#else
    localPartition = new T[nLocal];
#endif
    for (int i = 0; i < nLocal; i++)
        localPartition[i] = other.localPartition[i];
}

template<typename T>
void msl::DC<T>::freeLocalPartition() {
#ifdef __CUDACC__
    (cudaFreeHost(localPartition));
#else
    delete[] localPartition;
    localPartition = nullptr;
#endif
}

template<typename T>
void msl::DC<T>::freePlans() {
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
// template <typename T> void msl::DC<T>::swap(DC<T> &first, DC<T> &second) {}

// auxiliary method init()
template<typename T>
void msl::DC<T>::init() {
    if (Muesli::proc_entrance == UNDEFINED) {
        throws(detail::MissingInitializationException());
    }
    id = Muesli::proc_id;
    np = Muesli::num_total_procs;
    ng = Muesli::num_gpus;
    n = ncol * nrow * depth;
    if (!rowComplete) {
        nLocal = n / np;
    } else {
        // TODO not tested.
        auto nLocalRows = nrow / np;
        if (id == np - 1 && nrow % np != 0) {
            nLocalRows = nrow - (nLocalRows * np);
        }
        nLocal = nLocalRows * ncol;
    }

    nlocalRows = nLocal % ncol == 0 ? nLocal / ncol : nLocal / ncol + 1;
    //printf("Inside Init DC %d nLocal %d ncol %d nrow %d depth\n\n", nLocal, ncol, nrow, depth);

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

// auxiliary method initGPUs
template<typename T>
void msl::DC<T>::initGPUs() {
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

// destructor removes a DC
template<typename T>
msl::DC<T>::~DC() {
// printf("TODO: Destroy Datastructure\n");
#ifdef __CUDACC__
    (cudaFreeHost(localPartition));
    /*if (plans) {
        for (int i = 0; i < ng; i++) {
            if (plans[i].d_Data != 0) {
                cudaSetDevice(i);
                (cudaFree(plans[i].d_Data));
            }
        }
        delete[] plans;
    }*/
#else
    delete[] localPartition;
#endif
}

// ***************************** auxiliary methods
// ******************************
template<typename T>
T *msl::DC<T>::getLocalPartition() {
    if (!cpuMemoryInSync)
        download();
    return localPartition;
}

template<typename T>
void msl::DC<T>::setLocalPartition(T *elements) {
    localPartition = elements;
    initGPUs();
    upload();
    return;
}
template<typename T>
void msl::DC<T>::fill(const T &element) {
    #pragma omp parallel for
    for (int i = 0; i<nrow*ncol*depth; i++){
        localPartition[i] = element;
    }
    initGPUs();
    upload();
    return;
}
template<typename T>
T msl::DC<T>::get(int index) const {
    int idSource;
    T message;
    // TODO: adjust to new structure
    // element with global index is locally stored
    if (index < indexGPU) {
        return localPartition[index];
    }
    if (isLocal(index)) {
#ifdef __CUDACC__
        // element might not be up to date in cpu memory
        //cpuMemoryInSync = false;
        if (!cpuMemoryInSync) {
            // find GPU that stores the desired element
            int device = getGpuId(index);
            cudaSetDevice(device);
            // download element
            int offset = index - plans[device].first;
            (cudaMemcpyAsync(&message, plans[device].d_Data + offset,
                                              sizeof(T), cudaMemcpyDeviceToHost,
                                              Muesli::streams[device]));
        } else { // element is up to date in cpu memory
            message = localPartition[index - firstIndex];
        }
#else
        message = localPartition[index - firstIndex];
#endif
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
T msl::DC<T>::get3D(int row, int col, int depth, int gpu) const {
    int index = (row) * ncol + col + (nrow*ncol) * depth;
    T message;
#ifdef __CUDA_ARCH__
    // TODO calculate from row col? --> SM?
    int offset = index - plans[gpu].first;
    message = *plans[gpu].d_Data + offset;
#else
        message = localPartition[index - firstIndex];
#endif

    return message;
}
template<typename T>
T msl::DC<T>::get_shared(int row, int column) const {
    int index = row * column;
    // TODO load from shared mem
    int idSource;
    T message;
    // TODO: adjust to new structure
    // element with global index is locally stored
    if (isLocal(index)) {
#ifdef __CUDACC__
        // element might not be up to date in cpu memory
        if (!cpuMemoryInSync) {
            // find GPU that stores the desired element
            int device = getGpuId(index);
            cudaSetDevice(device);
            // download element
            int offset = index - plans[device].first;
            (cudaMemcpyAsync(&message, plans[device].d_Data + offset,
                                              sizeof(T), cudaMemcpyDeviceToHost,
                                              Muesli::streams[device]));
        } else { // element is up to date in cpu memory
            message = localPartition[index - firstIndex];
        }
#else
        message = localPartition[index - firstIndex];
#endif
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
int msl::DC<T>::getSize() const { return n; }

template<typename T>
int msl::DC<T>::getLocalSize() const { return nLocal; }

template<typename T>
int msl::DC<T>::getFirstIndex() const {
    return firstIndex;
}

template<typename T>
void msl::DC<T>::setCpuMemoryInSync(bool b) {
    cpuMemoryInSync = b;
}

template<typename T>
bool msl::DC<T>::isLocal(int index) const {
    return (index >= firstIndex) && (index < firstIndex + nLocal);
}

template<typename T>
T msl::DC<T>::getLocal(int localIndex) {
    if (localIndex >= nLocal)
        throws(detail::NonLocalAccessException());
    if ((!cpuMemoryInSync) && (localIndex >= nCPU)) {
        download();}
    return localPartition[localIndex];
}

template<typename T>
T& msl::DC<T>::operator[](int index) {
    return localPartition[index];
}
// TODO:adjust to new structure
template<typename T>
void msl::DC<T>::setLocal(int localIndex, const T &v) {


    if (localIndex < nCPU) {
        localPartition[localIndex] = v;
    } else if (localIndex >= nLocal)
        throws(detail::NonLocalAccessException());
    else { // localIndex refers to a GPU
           #ifdef __CUDACC__
        // approach 1: easy, but inefficient
        // if (!cpuMemoryInSync)
        //  download();
        // localPartition[localIndex] = v;
        // upload(); //easy, but inefficient
        // approach 2: more efficient, but also more complicated:
        // TODO adjust to new!
        int gpuId = (localIndex - nCPU) / nGPU;
        int idx = localIndex - nCPU - gpuId * nGPU;
        //printf("gpuid %d idx %d \n", (localIndex - nCPU) / nGPU, localIndex - nCPU - gpuId * nGPU);
        //printf("setLocal: localIndex: %i, gpuId: %i, idx: %i - %d - %d\n", localIndex, gpuId, idx, nCPU, nGPU); // debug
        cudaSetDevice(gpuId);
        (cudaMemcpy(&(plans[gpuId].d_Data[idx]), &v, sizeof(T),
                                     cudaMemcpyHostToDevice));
           #endif
    }
}

template<typename T>
void msl::DC<T>::set(int globalIndex, const T &v) {
    if ((globalIndex >= firstIndex) && (globalIndex < firstIndex + nLocal)) {
        setLocal(globalIndex - firstIndex, v);
    }
}

template<typename T>
GPUExecutionPlan<T> *msl::DC<T>::getExecPlans() {
    // std::vector<GPUExecutionPlan<T> > ret(plans, plans + Muesli::num_gpus);
    return plans;
}

template<typename T>
void msl::DC<T>::upload() {
    std::vector<T *> dev_pointers;
#ifdef __CUDACC__
    //if (cpuMemoryInSync) {
        for (int i = 0; i < ng; i++) {
            cudaSetDevice(i);
            // upload data
            (cudaMemcpyAsync(plans[i].d_Data, plans[i].h_Data,
                                              plans[i].bytes, cudaMemcpyHostToDevice,
                                              Muesli::streams[i]));
        }

        for (int i = 0; i < ng; i++) {
            (cudaStreamSynchronize(Muesli::streams[i]));
        }
        cpuMemoryInSync = false;
    //}
#endif
    return;
}

template<typename T>
void msl::DC<T>::download() {
#ifdef __CUDACC__
    cpuMemoryInSync = false;
    if (!cpuMemoryInSync) {
        for (int i = 0; i < ng; i++) {
            cudaSetDevice(i);
            // download data from device
            (cudaMemcpyAsync(plans[i].h_Data, plans[i].d_Data,
                                              plans[i].bytes, cudaMemcpyDeviceToHost,
                                              Muesli::streams[i]));
        }

        // wait until download is finished
        for (int i = 0; i < ng; i++) {
            (cudaStreamSynchronize(Muesli::streams[i]));
        }

        cpuMemoryInSync = true;

    }
    //CUDA_CHECK_RETURN(cudaMemcpyAsync(&localPartition[64], plans[0].d_Data, plans[0].bytes, cudaMemcpyDeviceToHost, Muesli::streams[0]));
    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );

#endif
}

template<typename T>
int msl::DC<T>::getGpuId(int index) const {
    return (index - firstIndex - nCPU) / nGPU;
}

template<typename T>
int msl::DC<T>::getFirstGpuRow() const {
    return indexGPU / ncol;
}

// method (only) useful for debbuging
template<typename T>
void msl::DC<T>::showLocal(const std::string &descr) {
    if (!cpuMemoryInSync) {
        download();
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
void msl::DC<T>::show(const std::string &descr) {
    #ifdef __CUDACC__
    cudaDeviceSynchronize();
    #endif
    T *b = new T[n];
    std::cout.precision(2);
    std::ostringstream s;
    if (descr.size() > 0)
        s << descr << ": " << std::endl;
    if (!cpuMemoryInSync) {
        download();
    } else {
        //std::cout << "CPU in sync" << std::endl;
    }
    // localPartition, plans[0].h_Data[i]
/*    int gpuBase = indexGPU;
      for(int i = 0; i < ng; i++){
        localPartition+gpuBase = &plans[i].h_Data;
        gpuBase += plans[i].size;
    }*/

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
// SKELETONS / COMMUNICATION / GATHER
template<typename T>
T* msl::DC<T>::gather() {
    T *b = new T[n];
    std::cout.precision(2);
    std::ostringstream s;

    if (!cpuMemoryInSync) {
        download();
    } else {
        //std::cout << "CPU in sync" << std::endl;
    }
    msl::allgather(localPartition, b, nLocal);

    return b;
}
template<typename T>
void msl::DC<T>::gather(msl::DC<T> &da) {
    printf("gather\n");

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
inline void msl::DC<T>::permutePartition(Functor& f) {
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
      download();
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
    upload();
  }
}
*/
// template<typename T>
// inline void msl::DC<T>::permutePartition(int (*f)(int)) {
//  permutePartition(curry(f));
//}

template<typename T>
void msl::DC<T>::freeDevice() {
#ifdef __CUDACC__
    if (!cpuMemoryInSync) {
        for (int i = 0; i < ng; i++) {
            //  if(plans[i].d_Data == 0) {
            //    continue;
            //  }
            cudaFree(plans[i].d_Data);
            plans[i].d_Data = 0;
        }
        cpuMemoryInSync = true;
    }
#endif
}
template<typename T>
void msl::DC<T>::printTime() {//t10 %.2f;t7 %.2f;t8 %.2f;t9 %.2f; - t10, t7, t8, t9,
    printf("\nt0 %.2f;t1 %.2f;t2 %.2f;t3 %.2f;t4 %.2f;t5 %.2f;t6 %.2f; t7%.2f; sum %.2f;",
           t0, t1, t2, t3, t4, t5, t6,t7, t0+t4+t1+t2+t3+t5+t6+t7+t8+t9);
}

//*********************************** Maps ********************************

template<typename T>
template<typename MapFunctor>
void msl::DC<T>::mapInPlace(MapFunctor &f) {
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
void msl::DC<T>::mapIndexInPlace(MapIndexFunctor &f) {
#ifdef __CUDACC__

    //  int colGPU = (ncol * (1 - Muesli::cpu_fraction)) / ng;  // is not used, HK
    for (int i = 0; i < ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        detail::mapIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
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
msl::DC<T> msl::DC<T>::map(
        F &f) {        // preliminary simplification in order to avoid type error
    DC <T> result(nrow, ncol, depth); // should be: DC<R>
#ifdef __CUDACC__

    // map
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        detail::mapKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                //          plans[i].d_Data, plans[i].d_Data, plans[i].size, f); // in,
                //          out, #bytes, function
                plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].size, f);
    }
#endif
#pragma omp parallel for
    for (int k = 0; k < nCPU; k++) {
        result.setLocal(k, f(localPartition[k]));
    }

    // check for errors during gpu computation
    msl::syncStreams();
    result.setCpuMemoryInSync(false);
    result.download();

    return result;
}

template<typename T>
template<typename MapIndexFunctor>
msl::DC<T> msl::DC<T>::mapIndex(MapIndexFunctor &f) {
    DC <T> result(nrow, ncol, depth);
    // TODO : CUDA Kernel
#ifdef __CUDACC__

    // map on GPUs
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        detail::mapIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].nLocal,
                        plans[i].first, f, ncol);
    }
#endif
#pragma omp parallel for
    for (int k = 0; k < nCPU; k++) {
        int l = (k + firstIndex) / (ncol*nrow);
        int j = ((k + firstIndex) - l*(ncol*nrow)) / ncol;
        int i = (k + firstIndex) % ncol;
        result.setLocal(k, f(i, j, l, localPartition[k]));
    }

    // check for errors during gpu computation
    msl::syncStreams();
    result.setCpuMemoryInSync(false);
    result.download();

    return result;
}
// ************************************ zip
// ***************************************
template<typename T>
template<typename T2, typename ZipFunctor>
void msl::DC<T>::zipInPlace(DC <T2> &b, ZipFunctor &f) {
    // zip on GPU
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
msl::DC<T>
msl::DC<T>::zip(DC <T2> &b,
                ZipFunctor &f) { // should have result type DA<R>; debug
    DC <T> result(nrow, ncol);
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
    return result;
}

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DC<T>::zipIndexInPlace(DC <T2> &b, ZipIndexFunctor &f) {
    // zip on GPUs
#ifdef __CUDACC__
    for (int i = 0; i < ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);

        detail::zipIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                plans[i].d_Data, b.getExecPlans()[i].d_Data, plans[i].d_Data,
                        plans[i].nLocal, plans[i].first, f, ncol);
    }
#endif
    if (nCPU > 0){
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
}
template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DC<T>::crossZipIndexInPlace(DC <T2> &b, ZipIndexFunctor &f) {
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
    if (nCPU > 0){
        T2 *bPartition = b.getLocalPartition();
#pragma omp parallel for
        for (int k = 0; k < nCPU; k++) {
            int i = (k + firstIndex) / ncol;
            int j = (k + firstIndex) % ncol;
            localPartition[k] = f(i, j, localPartition, bPartition);
        }
    }
    // check for errors during gpu computation
    cpuMemoryInSync = false;
}
template<typename T>
template<typename T2, typename ZipIndexFunctor>
msl::DC<T> msl::DC<T>::zipIndex(DC <T2> &b, ZipIndexFunctor &f) {
    DC <T> result(nrow, ncol);
    // zip on GPUs
#ifdef __CUDACC__

    for (int i = 0; i < ng; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        detail::zipIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
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
    return result;
}

template<typename T>
template<typename T2, typename T3, typename ZipFunctor>
void msl::DC<T>::zipInPlace3(DC <T2> &b, DC <T3> &c, ZipFunctor &f) {
    // zip on GPU
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

// template <typename T>
// template <typename T2, typename T3, typename T4, typename ZipFunctor>
// void msl::DC<T>::zipInPlaceAAM(DA<T2> &b, DA<T3> &c, DC<T4> &d, ZipFunctor
// &f) {
//   // zip on GPU
//   for (int i = 0; i < ng; i++) {
//     cudaSetDevice(i);
//     dim3 dimBlock(Muesli::threads_per_block);
//     dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
//     auto bplans = b.getExecPlans();
//     auto cplans = c.getExecPlans();
//     auto dplans = d.getExecPlans();
//     detail::zipKernelAAM<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
//         plans[i].d_Data, bplans[i].d_Data, cplans[i].d_Data,
//         d.plans[i].d_Data, plans[i].d_Data, plans[i].nLocal, plans[i].first,
//         bplans[i].first, f, ncol);
//   }

//   T2 *bPartition = b.getLocalPartition();
//   T3 *cPartition = c.getLocalPartition();
//   T4 *dPartition = d.getLocalPartition();
//   int bfirst = b.getFirstIndex();
// #pragma omp parallel for
//   for (int k = 0; k < nCPU; k++) {
//     int i = ((k + firstIndex) / ncol) - bfirst;
//     //    printf("k=%d, i=%d, firstIndex=%d, ncol=%d, localPartition[k]=%d,
//     //            bPartition[i]=%d, cPartition[i]=%d, dPartition[k]=%d,
//     //            bfirst=%d\n", k, i, firstIndex, ncol, localPartition[k],
//     //            bPartition[i], cPartition[i], dPartition[k], bfirst); //
//     debug localPartition[k] =
//         f(localPartition[k], bPartition[i], cPartition[i], dPartition[k]);
//   }

//   // check for errors during gpu computation
//   msl::syncStreams();
//   cpuMemoryInSync = false;
// }

// *********** fold *********************************************
#ifdef __CUDACC__
template<typename T>
template<typename FoldFunctor>
T msl::DC<T>::fold(FoldFunctor &f, bool final_fold_on_cpu) {
    if (!cpuMemoryInSync) {
        download();
    }
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
    upload();

    //
    // Step 1: local fold
    //

    // prearrangement: calculate threads, blocks, etc.; allocate device memory
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        threads[i] = (plans[i].size < maxThreads)
                     ? detail::nextPow2((plans[i].size + 1) / 2)
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
    T cpu_result = localPartition[0];
    for (int k = 1; k < nCPU; k++) {
        cpu_result = f(cpu_result, localPartition[k]);
    }

    msl::syncStreams();

    // fold on gpus: step 2
    for (int i = 0; i < Muesli::num_gpus; i++) {
        if (blocks[i] > 1) {
            cudaSetDevice(i);
            int threads = (detail::nextPow2(blocks[i]) == blocks[i])
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

            // upload data
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
            // upload data
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
T msl::DC<T>::fold(FoldFunctor &f, bool final_fold_on_cpu) {
    if (!cpuMemoryInSync) {
        download();
    }
    T localresult = 0;

#pragma omp parallel for shared(localPartition) reduction(+: localresult)
    for (int i = 0; i < nLocal; i++) {
        localresult = f(localresult, localPartition[i]);
    }

    // TODO MPI global result
    // T* globalResults = new T[np];
    return localresult;
}
#endif