/*
 * CDM.cpp
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
#include <iostream>
#include "muesli.h"
#include "da.h"

template<typename T>
msl::CDM<T>::CDM():                  // distributed array (resides on GPUs until deleted!)
    n(0),                        // number of elements of distributed matrix
    ncol(0),                     // number of columns of distributed matrix
    nrow(0),                     // number of rows of distributed matrix
    np(0),                       // number of (MPI-) nodes (= Muesli::num_local_procs)
    id(0),                       // id of local node among all nodes (= Muesli::proc_id)
    cpuMemoryInSync(false),      // is GPU memory in sync with CPU?
    globalMemoryInSync(false),   // is GPU memory in sync with CPU?
    firstIndex(0),               // first global index of the CDM in the local partition
    firstRow(0),                 // first golbal row index of the CDM on the local partition
    plans(0),                    // GPU execution plans
// new: for combined usage of CPU and GPUs on every MPI-node
    ng(0),                       // number of GPUs per node (= Muesli::num_gpus)
    indexGPU(0)                  // number of elements on CPU = n - ng*nGPU
{}

// constructor creates a non-initialized CDM
template<typename T>
msl::CDM<T>::CDM(int row, int col)
    : ncol(col), nrow(row), n(col*row){
  init();
#ifdef __CUDACC__
  localPartition = new T[n];
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, n*sizeof(T)));
  initGPUs();
#else
#endif
  cpuMemoryInSync = true;
}

// constructor creates a CDM, initialized with v
template<typename T>
msl::CDM<T>::CDM(int row, int col, const T& v)
    : ncol(col), nrow(row), n(col*row){
  init();
#ifdef __CUDACC__
  localPartition = new T[n];
  // TODO die CPU Elemente brauchen wir nicht unbedingt.
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, n*sizeof(T)));
  initGPUs();
#else
#endif
#pragma omp parallel for
  for (int i=0; i< n; i++) localPartition[i] = v;
  upload();
}

// auxiliary method init()
template<typename T>
void msl::CDM<T>::init() {
  if (Muesli::proc_entrance == UNDEFINED) {
    throws(detail::MissingInitializationException());}
  id = Muesli::proc_id;
  np = Muesli::num_total_procs;
  ng = Muesli::num_gpus;
  n = ncol * nrow;
  firstIndex = 0;
}


// auxiliary method initGPUs
template<typename T>
void msl::CDM<T>::initGPUs() {
#ifdef __CUDACC__
  plans = new GPUExecutionPlan<T>[ng];
  int gpuBase = 0;
  for (int i = 0; i<ng; i++) {
    cudaSetDevice(i);
    plans[i].size = n;
    plans[i].nLocal = plans[i].size;
    plans[i].bytes = plans[i].size * sizeof(T);
    plans[i].first = 0;
    plans[i].h_Data = 0;
    CUDA_CHECK_RETURN(cudaMalloc(&plans[i].d_Data, plans[i].bytes));
  }
#endif
}

// destructor removes a CDM
template<typename T>
msl::CDM<T>::~CDM() {
  printf("TODO: Destroy Datastructure\n");
/*#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaFreeHost(localPartition));
  for (int i = 0; i < ng; i++) {
    if (plans[i].d_Data != 0) {
      cudaSetDevice(i);
      CUDA_CHECK_RETURN(cudaFree(plans[i].d_Data));
    }
  }
  delete[] plans;
#endif
  delete[] localPartition;
*/
}

// ***************************** auxiliary methods ******************************

template<typename T>
T msl::CDM<T>::get(int index) const {
  // TODO
  throws(detail::NotYetImplementedException());

  //msl::MSL_Broadcast(idSource, &message, 1);
  return index;
}

template<typename T>
int msl::CDM<T>::getSize() const {
  return n;
}

template<typename T>
int msl::CDM<T>::getCols() const {
  return ncol;
}

template<typename T>
int msl::CDM<T>::getRows() const {
  return nrow;
}

template<typename T>
int msl::CDM<T>::getFirstIndex() const {
  return firstIndex;
}

template<typename T>
void msl::CDM<T>::setCpuMemoryInSync(bool b) {
  cpuMemoryInSync = b;
}

template<typename T>
void msl::CDM<T>::set(int globalIndex, const T& v) {
  localPartition[globalIndex] = v;
}

template<typename T>
GPUExecutionPlan<T>* msl::CDM<T>::getExecPlans(){
  //std::vector<GPUExecutionPlan<T> > ret(plans, plans + Muesli::num_gpus);
  return plans;
}

template<typename T>
void msl::CDM<T>::upload() {
  std::vector<T*> dev_pointers;
#ifdef __CUDACC__
  if (!cpuMemoryInSync) {
    for (int i = 0; i < ng; i++) {
      cudaSetDevice(i);
      // upload data
      CUDA_CHECK_RETURN(
          cudaMemcpyAsync(plans[i].d_Data, plans[i].h_Data, plans[i].bytes,
                          cudaMemcpyHostToDevice, Muesli::streams[i]));
    }
    cpuMemoryInSync = true;
  }
#endif
  return;
}

template<typename T>
template<typename gatherfunctor>
void msl::CDM<T>::download(gatherfunctor gf) {
  // TODO merge with gather functor
  throws(detail::NotYetImplementedException());
#ifdef __CUDACC__
  if (!cpuMemoryInSync) {
    for (int i = 0; i< ng; i++) {
      cudaSetDevice(i);

      // download data from device
      CUDA_CHECK_RETURN(
          cudaMemcpyAsync(plans[i].h_Data, plans[i].d_Data, plans[i].bytes,
                          cudaMemcpyDeviceToHost, Muesli::streams[i]));
    }
    // wait until download is finished
    for (int i = 0; i < ng; i++) {
      CUDA_CHECK_RETURN(cudaStreamSynchronize(Muesli::streams[i]));
    }
    cpuMemoryInSync = true;
  }
#endif
}

template<typename T>
void msl::CDM<T>::show(const std::string& descr) {
  T* b = new T[n];
  std::ostringstream s;
  if (descr.size() > 0)
    s << descr << ": " << std::endl;
  if (!cpuMemoryInSync) {
    download();
  }
  msl::allgather(localPartition, b, n);

  if (msl::isRootProcess()) {
    s << "[";
    for (int i = 0; i < n - 1; i++) {
      s << b[i];
      ((i+1) % ncol == 0) ? s << "\n " : s << " ";;
    }
    s << b[n - 1] << "]" << std::endl;
    s << std::endl;
  }

  delete b;

  if (msl::isRootProcess()) printf("%s", s.str().c_str());
}

// SKELETONS / COMMUNICATION / BROADCAST PARTITION

template<typename T>
void msl::CDM<T>::broadcast() {
  // TODO send the data from p 1 to all nodes
  throws(detail::NotYetImplementedException());
  // Updates the GPU Memory
  upload();
}

template<typename T>
void msl::CDM<T>::broadcast(msl::DM<T> dm) {
  // TODO only for one nodes and then broadcast
#pragma omp parallel for
  for (int i=0; i< n; i++) localPartition[i] = dm[i];
  throws(detail::NotYetImplementedException());
  broadcast();
}

// SKELETONS / COMMUNICATION / GATHER

template<typename T>
template<typename gatherfunctor>
void msl::CDM<T>::gather(gatherfunctor gf) {
  printf("gather\n");
  throws(detail::NotYetImplementedException());
  msl::DM<T> result(nrow, ncol, 1);
  // TODO sum up all matrixes due to functor
  return result;
}


template<typename T>
void msl::CDM<T>::freeDevice() {
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

// *********** fold *********************************************
template <typename T>
template <typename FoldFunctor>
T msl::CDM<T>::fold(FoldFunctor& f, bool final_fold_on_cpu){
  throws(detail::NotYetImplementedException());

  if (!cpuMemoryInSync) {
    download();
  }
  std::vector<int> blocks(Muesli::num_gpus);
  std::vector<int> threads(Muesli::num_gpus);
  T* gpu_results = new T[Muesli::num_gpus];
  int maxThreads = 1024;   // preliminary
  int maxBlocks = 1024;    // preliminary
  for (int i = 0; i < Muesli::num_gpus; i++) {
    threads[i] = maxThreads;
    gpu_results[i] = 0;
  }
  T* local_results = new T[np];
  T** d_odata = new T*[Muesli::num_gpus];
  upload();

  //
  // Step 1: local fold
  //

  // prearrangement: calculate threads, blocks, etc.; allocate device memory
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    threads[i] = (plans[i].size < maxThreads) ? detail::nextPow2((plans[i].size+1)/2) : maxThreads;
    blocks[i] = plans[i].size / threads[i];
    if (blocks[i] > maxBlocks) {
      blocks[i] = maxBlocks;
    }
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_odata[i], blocks[i] * sizeof(T)));
  }

  // fold on gpus: step 1
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    detail::reduce<T, FoldFunctor>(plans[i].size, plans[i].d_Data, d_odata[i], threads[i], blocks[i], f, Muesli::streams[i], i);
  }

  // fold local elements on CPU (overlap with GPU computations)
  // TODO: openmp has parallel reduce operators.
  T cpu_result = localPartition[0];
  for (int k = 1; k<n; k++) {
    cpu_result = f(cpu_result, localPartition[k]);
  }

  msl::syncStreams();

  // fold on gpus: step 2
  for (int i = 0; i < Muesli::num_gpus; i++) {
    if (blocks[i] > 1) {
      cudaSetDevice(i);
      int threads = (detail::nextPow2(blocks[i]) == blocks[i]) ? blocks[i] : detail::nextPow2(blocks[i])/2;
      detail::reduce<T, FoldFunctor>(blocks[i], d_odata[i], d_odata[i], threads, 1, f, Muesli::streams[i], i);
    }
  }
  msl::syncStreams();

  // copy final sum from device to host
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&gpu_results[i],
                                      d_odata[i],
                                      sizeof(T),
                                      cudaMemcpyDeviceToHost,
                                      Muesli::streams[i]));
  }
  //printf("%d. Cpu Result %d\n"
  // "1.1 Gpu Result %d\n"
  //  "1.2 GpuResult %d \n", id, cpu_result, gpu_results[0], gpu_results[1]);
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
    T local_result; T* d_gpu_results;
    if (Muesli::num_gpus > 1) { // if there is more than 1 GPU
      cudaSetDevice(0);         // calculate local result on device 0

      // upload data
      CUDA_CHECK_RETURN(cudaMalloc((void**)&d_gpu_results, Muesli::num_gpus * sizeof(T)));
      CUDA_CHECK_RETURN(cudaMemcpyAsync(d_gpu_results,
                                        gpu_results,
                                        Muesli::num_gpus * sizeof(T),
                                        cudaMemcpyHostToDevice,
                                        Muesli::streams[0]));
      CUDA_CHECK_RETURN(cudaStreamSynchronize(Muesli::streams[0]));

      // final (local) fold
      detail::reduce<T, FoldFunctor>(Muesli::num_gpus, d_gpu_results, d_gpu_results, Muesli::num_gpus, 1, f, Muesli::streams[0], 0);
      CUDA_CHECK_RETURN(cudaStreamSynchronize(Muesli::streams[0]));

      // copy result from device to host
      CUDA_CHECK_RETURN(cudaMemcpyAsync(&local_result,
                                        d_gpu_results,
                                        sizeof(T),
                                        cudaMemcpyDeviceToHost,
                                        Muesli::streams[0]));
      CUDA_CHECK_RETURN(cudaStreamSynchronize(Muesli::streams[0]));
      CUDA_CHECK_RETURN(cudaFree(d_gpu_results));
    } else {
      local_result = gpu_results[0];
    }

    if (np > 1) {
      // gather all local results
      msl::allgather(&local_result, local_results, 1);

      // calculate global result from local results
      // upload data
      CUDA_CHECK_RETURN(cudaMalloc((void**)&d_gpu_results, np * sizeof(T)));
      CUDA_CHECK_RETURN(cudaMemcpyAsync(d_gpu_results,
                                        local_results,
                                        np * sizeof(T),
                                        cudaMemcpyHostToDevice,
                                        Muesli::streams[0]));

      // final fold
      detail::reduce<T, FoldFunctor>(np, d_gpu_results, d_gpu_results, np, 1, f, Muesli::streams[0], 0);
      CUDA_CHECK_RETURN(cudaStreamSynchronize(Muesli::streams[0]));

      // copy final result from device to host
      CUDA_CHECK_RETURN(cudaMemcpyAsync(&final_result,
                                        d_gpu_results,
                                        sizeof(T),
                                        cudaMemcpyDeviceToHost,
                                        Muesli::streams[0]));
      CUDA_CHECK_RETURN(cudaStreamSynchronize(Muesli::streams[0]));
      CUDA_CHECK_RETURN(cudaFree(d_gpu_results));
    } else {
      final_result = local_result;
    }
  }

  // Cleanup
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(Muesli::streams[i]));
    CUDA_CHECK_RETURN(cudaFree(d_odata[i]));
  }
  delete[] gpu_results;
  delete[] d_odata;
  delete[] local_results;

  return final_result;
}



