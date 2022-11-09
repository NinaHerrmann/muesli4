/*
 * ds.cpp
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
#include <ds.h>
#include "cuda_runtime.h"
#include "cuda.h"

template<typename T>
msl::DS<T>::DS()
        :       // distributed array (resides on GPUs until deleted!)
        n(0), // number of elements of distributed array
        dim(0), // number of elements of distributed array
        nLocal(0), // number of local elements on a node
        np(0),             // number of (MPI-) nodes (= Muesli::num_local_procs)
        id(0),             // id of local node among all nodes (= Muesli::proc_id)
        localPartition(0), // local partition of the DS
        cpuMemoryInSync(false), // is GPU memory in sync with CPU?
        firstIndex(0), // first global index of the DS on the local partition
        plans(0),      // GPU execution plans
        dist(Distribution::DIST), // distribution of DS: DIST (distributed) or
        // COPY (for now: always DIST)
        gpuCopyDistributed(0),     // is GPU copy distributed? (for now: always "false")
        // new: for combined usage of CPU and GPUs on every MPI-node
        ng(0),      // number of GPUs per node (= Muesli::num_gpus)
        nGPU(0),    // number of elements per GPU (all the same!)
        nCPU(0),    // number of elements on CPU = nLocal - ng*nGPU
        indexGPU(0) // number of elements on CPU = nLocal - ng*nGPU
{}

// constructor creates a non-initialized DS
template<typename T>
msl::DS<T>::DS(int size, int dimensions) : n(size), dim(dimensions) {
    init();
#ifdef __CUDACC__
    (cudaMallocHost(&localPartition, nLocal * sizeof(T)));
    initGPUs();
#else
    localPartition = new T[nLocal];
#endif
    cpuMemoryInSync = false;
}


// constructor creates a DS, initialized with v
template<typename T>
msl::DS<T>::DS(int elements, int dimensions, const T &v)
: n(elements), dim(dimensions) {
    init();
    localPartition = new T[nLocal];

#ifdef __CUDACC__
    // TODO die CPU Elemente brauchen wir nicht unbedingt.
    (cudaMallocHost(&localPartition, nLocal * sizeof(T)));
#endif
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++){
        localPartition[i] = v;
    }
    cpuMemoryInSync = true;
#ifdef __CUDACC__
    initGPUs();
#endif
    updateDevice();
}

template<typename T>
msl::DS<T>::DS(const DS <T> &other)
        : id(other.id), n(other.n), nLocal(other.nLocal), dim(other.dim),
          firstIndex(other.firstIndex), np(other.np),
          cpuMemoryInSync(other.cpuMemoryInSync), plans{new GPUExecutionPlan<T>{
                *(other.plans)}},
          gpuCopyDistributed(other.gpuCopyDistributed), ng(other.ng),
          nGPU(other.nGPU), nCPU(other.nCPU), indexGPU(other.indexGPU){
    copyLocalPartition(other);

    // cpuMemoryInSync = true;
    // updateDevice();
}

template<typename T>
msl::DS<T>::DS(DS <T> &&other)
    : id(other.id), n(other.n), nLocal(other.nLocal), dim(other.dim),
    firstIndex(other.firstIndex), np(other.np),
    cpuMemoryInSync(other.cpuMemoryInSync), plans{new GPUExecutionPlan<T>{
    *(other.plans)}},
    gpuCopyDistributed(other.gpuCopyDistributed), ng(other.ng),
    nGPU(other.nGPU), nCPU(other.nCPU), indexGPU(other.indexGPU){
    other.plans = nullptr;
    localPartition = other.localPartition;
    other.localPartition = nullptr;
}

template<typename T>
msl::DS<T> &msl::DS<T>::operator=(DS <T> &&other) {
    printf("=\n");
    throws(detail::NotYetImplementedException());
}

template<typename T>
msl::DS<T> &msl::DS<T>::operator=(const DS <T> &other) {
    printf("=\n");
    throws(detail::NotYetImplementedException());
}

template<typename T>
void msl::DS<T>::copyLocalPartition(const DS <T> &other) {
#ifdef __CUDACC__
    (cudaMallocHost(&localPartition, nLocal * sizeof(T)));
#else
    localPartition = new T[nLocal];
#endif
    for (int i = 0; i < nLocal; i++)
        localPartition[i] = other.localPartition[i];

    cpuMemoryInSync = true;
}

template<typename T>
void msl::DS<T>::freeLocalPartition() {
#ifdef __CUDACC__
    (cudaFreeHost(localPartition));
#else
    delete[] localPartition;
    localPartition = nullptr;
#endif
}

template<typename T>
void msl::DS<T>::freePlans() {
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
// template <typename T> void msl::DS<T>::swap(DS<T> &first, DS<T> &second) {}

// auxiliary method init()
template<typename T>
void msl::DS<T>::init() {
    if (Muesli::proc_entrance == UNDEFINED) {
        throws(detail::MissingInitializationException());
    }

    id = Muesli::proc_id;
    np = Muesli::num_total_procs;
    ng = Muesli::num_gpus;
    nLocal = n / np;

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
}

// auxiliary method initGPUs
template<typename T>
void msl::DS<T>::initGPUs() {
#ifdef __CUDACC__
    plans = new GPUExecutionPlan<T>[ng];
    int gpuBase = indexGPU;
    for (int i = 0; i < ng; i++) {
        cudaSetDevice(i);
        plans[i].size = nGPU;
        plans[i].nLocal = plans[i].size;
        plans[i].bytes = plans[i].size * sizeof(T);
        plans[i].first = gpuBase + firstIndex;
        plans[i].h_Data = localPartition + gpuBase;
        size_t total; size_t free;
        cuMemGetInfo(&free, &total);
        if (plans[i].bytes > free) {
            throws(detail::DeviceOutOfMemory());
            exit(0);
        }
        (cudaMalloc(&plans[i].d_Data, plans[i].bytes));
        gpuBase += plans[i].size;
    }
#endif
}

// destructor removes a DS
template<typename T>
msl::DS<T>::~DS() {
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

// ***************************** auxiliary methods
// ******************************
template<typename T>
T *msl::DS<T>::getLocalPartition() {
    updateHost();
    return localPartition;
}

template<typename T>
void msl::DS<T>::setLocalPartition(T *elements) {
    localPartition = elements;
    initGPUs();
    updateDevice();
    return;
}
template<typename T>
void msl::DS<T>::fill(const T &element) {
    #pragma omp parallel for default(none) shared(element)
    for (int i = 0; i<nLocal; i++){
        localPartition[i] = element;
    }
    cpuMemoryInSync = true;
#ifdef __CUDACC__
    initGPUs();
    updateDevice();
#endif
}
template<typename T>
T msl::DS<T>::get(int index) const {
    int idSource;
    T message;
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
        }
#else
        message = localPartition[index - firstIndex];
#endif
        idSource = Muesli::proc_id;
    }
    else {
        idSource = (int) (index / nLocal);
    }
    msl::MSL_Broadcast(idSource, &message, 1);
    return message;
}

template<typename T>
T msl::DS<T>::get_shared(int row, int column) const {
    printf("get_shared\n");
    throws(detail::NotYetImplementedException());
}
template<typename T>
int msl::DS<T>::getSize() const { return n; }

template<typename T>
int msl::DS<T>::getLocalSize() const { return nLocal; }

template<typename T>
int msl::DS<T>::getFirstIndex() const {
    return firstIndex;
}

template<typename T>
void msl::DS<T>::setCpuMemoryInSync(bool b) {
    cpuMemoryInSync = b;
}

template<typename T>
bool msl::DS<T>::isLocal(int index) const {
    return (index >= firstIndex) && (index < firstIndex + nLocal);
}

template<typename T>
T msl::DS<T>::getLocal(int localIndex) {
    if (localIndex >= nLocal)
        throws(detail::NonLocalAccessException());
    if ((!cpuMemoryInSync) && (localIndex >= nCPU)) {
        updateHost();}
    return localPartition[localIndex];
}
/*
template<typename T>
T& msl::DS<T>::operator[](int index) {
    return static_cast<int &>(static_cast<int &>(get(index)));
}*/
// TODO:adjust to new structure
template<typename T>
void msl::DS<T>::setLocal(int localIndex, const T &v) {
    if (localIndex < nCPU) {
        localPartition[localIndex] = v;
    } else if (localIndex >= nLocal) {
        throws(detail::NonLocalAccessException());
    } else {
#ifdef __CUDACC__
        int gpuId = (localIndex - nCPU) / nGPU;
        int idx = localIndex - nCPU - gpuId * nGPU;
        cudaSetDevice(gpuId);
        (cudaMemcpy(&(plans[gpuId].d_Data[idx]), &v, sizeof(T), cudaMemcpyHostToDevice));
#endif
    }
}

template<typename T>
void msl::DS<T>::set(int globalIndex, const T &v) {
    if ((globalIndex >= firstIndex) && (globalIndex < firstIndex + nLocal)) {
        setLocal(globalIndex - firstIndex, v);
    }
    // TODO: Set global
}

template<typename T>
GPUExecutionPlan<T> *msl::DS<T>::getExecPlans() {
    return plans;
}

template<typename T>
void msl::DS<T>::updateDevice() {
    std::vector<T *> dev_pointers;
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
        cpuMemoryInSync = false;
    }
#endif
}

template<typename T>
void msl::DS<T>::updateHost() {
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
#endif
}

template<typename T>
int msl::DS<T>::getGpuId(int index) const {
    return (index - firstIndex - nCPU) / nGPU;
}

// method (only) useful for debbuging
template<typename T>
void msl::DS<T>::showLocal(const std::string &descr) {
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
void msl::DS<T>::show(const std::string &descr) {
    T *b = new T[n];
    std::ostringstream s;
    if (!descr.empty())
        s << descr << ": " << std::endl;
    if (!cpuMemoryInSync) {
        updateHost();
    }
    msl::allgather(localPartition, b, nLocal);

    if (msl::isRootProcess()) {
        s << "[";
        for (int i = 0; i < n - 1; i++) {
            s << b[i];
            s << " ";
        }
        s << b[n - 1] << "]" << std::endl;
        s << std::endl;
    }

    delete[] b;

    if (msl::isRootProcess()) printf("%s", s.str().c_str());
}

// SKELETONS / COMMUNICATION / BROADCAST PARTITION
// SKELETONS / COMMUNICATION / GATHER
template<typename T>
T* msl::DS<T>::gather() {
    if (msl::Muesli::num_total_procs > 1) {
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
    } else {
#ifdef __CUDACC__
        if (!cpuMemoryInSync) {
            updateHost();
        }
#endif
        return localPartition;
    }
}
template<typename T>
void msl::DS<T>::gather(msl::DS<T> &da) {
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
inline void msl::DS<T>::permutePartition(Functor& f) {
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
      updateHost();
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
    updateDevice();
  }
}
*/
// template<typename T>
// inline void msl::DS<T>::permutePartition(int (*f)(int)) {
//  permutePartition(curry(f));
//}

template<typename T>
void msl::DS<T>::freeDevice() {
#ifdef __CUDACC__
    if (!cpuMemoryInSync) {
        for (int i = 0; i < ng; i++) {
            cudaFree(plans[i].d_Data);
            plans[i].d_Data = 0;
        }
        cpuMemoryInSync = true;
    }
#endif
}

//*********************************** Maps ********************************
template<typename T>
int msl::DS<T>::getnCPU(){
    return nCPU;
}
template<typename T>
template<typename MapFunctor>
void msl::DS<T>::mapInPlace(MapFunctor &f) {

#ifdef __CUDACC__
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        detail::mapKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                plans[i].d_Data, plans[i].d_Data, plans[i].size,
                f);
    }
#endif
    if (nCPU > 0) {
#pragma omp parallel for
        for (int k = 0; k < nCPU; k++) {
            localPartition[k] = f(localPartition[k]);
        }
    }
    cpuMemoryInSync = false;
}

template<typename T>
template<typename MapIndexFunctor>
void msl::DS<T>::mapIndexInPlace(MapIndexFunctor &f) {
#ifdef __CUDACC__
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        dim3 dimBlock(Muesli::threads_per_block);
        dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
        detail::mapKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                plans[i].d_Data, plans[i].d_Data, plans[i].size,
                f);
    }
#endif
    if (nCPU > 0) {
#pragma omp parallel for
        for (int k = 0; k < nCPU; k++) {
            localPartition[k] = f(localPartition[k]);
        }
    }
    cpuMemoryInSync = false;}

template<typename T>
template<typename F>
void msl::DS<T>::map(F &f, DS<T> &b) {        // preliminary simplification in order to avoid type error

    updateDevice();

    // map
#ifdef __CUDACC__
    for (int i = 0; i < Muesli::num_gpus; i++) {
      cudaSetDevice(i);
      dim3 dimBlock(Muesli::threads_per_block);
      dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
      detail::mapKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
              b.getExecPlans()[i].d_Data, plans[i].d_Data, plans[i].size, f);
    }
#endif
if (nCPU > 0) {

#pragma omp parallel for
    for (int i = 0; i < nCPU; i++) {
        localPartition[i] = f(b.getLocal(i));
    }
}
    setCpuMemoryInSync(false);
}


template<typename T>
template<typename MapIndexFunctor>
void msl::DS<T>::mapIndex(MapIndexFunctor &f, DS <T> &b) {
    if (getSize() != b.getSize()) {
        throws(detail::NotSameSizeException());
    }
    updateDevice();
    if (dim == 1) {
#ifdef __CUDACC__
        for (int i = 0; i < Muesli::num_gpus; i++) {
      cudaSetDevice(i);
      dim3 dimBlock(Muesli::threads_per_block);
      dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
      detail::mapIndexKernelDA<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                b.getExecPlans()[i].d_Data, plans[i].d_Data, plans[i].nLocal,
                plans[i].first, f);
    }
#endif
        // map on CPU cores
#pragma omp parallel for
        for (int i = 0; i < nCPU; i++) {
            b.setLocal(i, f((i + firstIndex), localPartition[i]));
        }
    } else if (dim == 2){

    }
    msl::syncStreams();
    b.setCpuMemoryInSync(false);
    return b;
//    printf("mapIndex\n");
//    throws(detail::NotYetImplementedException());
}
// ************************************ zip
// ***************************************
template<typename T>
template<typename T2, typename ZipFunctor>
void msl::DS<T>::zipInPlace(DS <T2> &b, ZipFunctor &f) {
    printf("zipInPlace\n");
    throws(detail::NotYetImplementedException());
}

template<typename T>
template<typename T2, typename ZipFunctor>
void
msl::DS<T>::zip(DS <T2> &b, DS <T2> &c,
                ZipFunctor &f) { // should have result type DS<R>; debug
    printf("Zip\n");
    throws(detail::NotYetImplementedException());
}

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DS<T>::zipIndexInPlace(DS <T2> &b, ZipIndexFunctor &f) {
    printf("zipIndexInPlace\n");
    throws(detail::NotYetImplementedException());
}
template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DS<T>::crossZipIndexInPlace(DS <T2> &b, ZipIndexFunctor &f) {
    printf("crossZipIndexInPlace\n");
    throws(detail::NotYetImplementedException());
}
template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DS<T>::zipIndex(DS <T2> &b, DS <T2> &c, ZipIndexFunctor &f) {
    printf("zipIndex\n");
    throws(detail::NotYetImplementedException());
}

template<typename T>
template<typename T2, typename T3, typename ZipFunctor>
void msl::DS<T>::zipInPlace3(DS <T2> &b, DS <T3> &c, ZipFunctor &f) {
    printf("zipInPlace3\n");
    throws(detail::NotYetImplementedException());
}


// *********** fold *********************************************
#ifdef __CUDACC__
template<typename T>
template<typename FoldFunctor>
T msl::DS<T>::fold(FoldFunctor &f, bool final_fold_on_cpu) {
    if (!cpuMemoryInSync) {
        updateHost();
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
    updateDevice();

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
    // TODO: when 0.0% cpu fraction this adds an extra element.
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
    msl::syncStreams();

    return final_result;
}
#else

template<typename T>
template<typename FoldFunctor>
T msl::DS<T>::fold(FoldFunctor &f, bool final_fold_on_cpu) {
    if (!cpuMemoryInSync) {
        updateHost();
    }
    T localresult = 0;

#pragma omp parallel for shared(localPartition) reduction(+: localresult)
    for (int i = 0; i < nLocal; i++) {
        localresult = f(localresult, localPartition[i]);
    }

    T *local_results = new T[np];
    T tmp = localresult;
    // gather all local results
    msl::allgather(&tmp, local_results, 1);

    // calculate global result from local results
    T global_result = local_results[0];
#pragma omp parallel for shared(local_results) reduction(+: global_result)
    for (int i = 1; i < np; i++) {
        global_result = f(global_result, local_results[i]);
    }

    // T* globalResults = new T[np];
    return global_result;
}
#endif

template<typename T>
template<typename MapStencilFunctor, typename NeutralValueFunctor>
void msl::DS<T>::mapStencilInPlace(MapStencilFunctor &f, NeutralValueFunctor &neutral_value_functor) {
    printf("mapStencilInPlace\n");
    throws(detail::NotYetImplementedException());
}

template<typename T>
template<typename MapStencilFunctor, typename NeutralValueFunctor>
void msl::DS<T>::mapStencil(DS<T> &result, MapStencilFunctor &f, NeutralValueFunctor &neutral_value_functor) {
    printf("mapStencilIn\n");
    throws(detail::NotYetImplementedException());
}