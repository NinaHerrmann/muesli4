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
#include <iostream>
#include "muesli.h"
#include "da.h"

template<typename T>
msl::DM<T>::DM():                  // distributed array (resides on GPUs until deleted!)
      n(0),                        // number of elements of distributed matrix
      ncol(0),                     // number of columns of distributed matrix
      nrow(0),                     // number of rows of distributed matrix
      nLocal(0),                   // number of local elements on a node    
      np(0),                       // number of (MPI-) nodes (= Muesli::num_local_procs)
      id(0),                       // id of local node among all nodes (= Muesli::proc_id)
      localPartition(0),           // local partition of the DM
      cpuMemoryInSync(false),      // is GPU memory in sync with CPU?
      firstIndex(0),               // first global index of the DM in the local partition
      firstRow(0),                 // first golbal row index of the DM on the local partition
      plans(0),                    // GPU execution plans
      dist(Distribution::DIST),    // distribution of DM: DIST (distributed) or COPY (for now: always DIST)
      gpuCopyDistributed(0),       // is GPU copy distributed? (for now: always "false")
// new: for combined usage of CPU and GPUs on every MPI-node
      ng(0),                       // number of GPUs per node (= Muesli::num_gpus)
      nGPU(0),                     // number of elements per GPU (all the same!)
      nCPU(0),                     // number of elements on CPU = nLocal - ng*nGPU
      indexGPU(0)                  // number of elements on CPU = nLocal - ng*nGPU
{}

// constructor creates a non-initialized DM
template<typename T>
msl::DM<T>::DM(int row, int col)
    : ncol(col), nrow(row), n(col*row){
  init();
#ifdef __CUDACC__
  localPartition = new T[nLocal];
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, nLocal*sizeof(T)));
  initGPUs();
#else 
#endif
  cpuMemoryInSync = true;
}

// constructor creates a DM, initialized with v
template<typename T>
msl::DM<T>::DM(int row, int col, const T& v)
    : ncol(col), nrow(row), n(col*row){
  init();
#ifdef __CUDACC__
  localPartition = new T[nLocal];
  // TODO die CPU Elemente brauchen wir nicht unbedingt.
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, nLocal*sizeof(T)));
  initGPUs();
#else
#endif
  #pragma omp parallel for
  for (int i=0; i< nLocal; i++) localPartition[i] = v;
  upload();
}

// auxiliary method init() 
template<typename T>
void msl::DM<T>::init() {
  if (Muesli::proc_entrance == UNDEFINED) {
    throws(detail::MissingInitializationException());}
  id = Muesli::proc_id;
  np = Muesli::num_total_procs;
  ng = Muesli::num_gpus;
  n = ncol * nrow;
  nLocal = n / np; 
  nGPU = ng > 0 ? nLocal * (1.0 - Muesli::cpu_fraction) / ng : 0;
  nCPU = nLocal - nGPU * ng;
  firstIndex =  id * nLocal;
  printf("loc prozesses %d , First index %d, firstRowGpu %d\n", Muesli::num_local_procs, firstIndex, indexGPU);
  printf("Building datastructure with %d nodes %d cpus and %d gpus\n", msl::Muesli::num_total_procs,
         msl::Muesli::num_local_procs, msl::Muesli::num_gpus);
}


// auxiliary method initGPUs
template<typename T>
void msl::DM<T>::initGPUs() {
#ifdef __CUDACC__
  plans = new GPUExecutionPlan<T>[ng];
  int gpuBase = nCPU;
  for (int i = 0; i<ng; i++) {
    cudaSetDevice(i);
    plans[i].size = nGPU;
    plans[i].nLocal = plans[i].size;
    plans[i].bytes = plans[i].size * sizeof(T);
    plans[i].first = gpuBase + firstIndex;
    plans[i].h_Data = localPartition + gpuBase;
    CUDA_CHECK_RETURN(cudaMalloc(&plans[i].d_Data, plans[i].bytes));
    gpuBase += plans[i].size;
  }
#endif
}

// destructor removes a DM
template<typename T>
msl::DM<T>::~DM() {
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
T* msl::DM<T>::getLocalPartition() {
  if (!cpuMemoryInSync)
    download();
  return localPartition;
}

template<typename T>
void msl::DM<T>::setLocalPartition(T* elements) {
  localPartition = elements;
  upload();
  return;
}

template<typename T>
T msl::DM<T>::get(int index) const {
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
      CUDA_CHECK_RETURN(
          cudaMemcpyAsync(&message,
              plans[device].d_Data+offset,
              sizeof(T),
              cudaMemcpyDeviceToHost,
              Muesli::streams[device]));
    } else {  // element is up to date in cpu memory
      message = localPartition[index-firstIndex];
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
int msl::DM<T>::getSize() const {
  return n;
}

template<typename T>
int msl::DM<T>::getLocalSize() const {
  return nLocal;
}

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
  if ((!cpuMemoryInSync) && (localIndex >= nCPU))
    download(); // easy, but inefficient, if random, non-consecutive access of elements
  return localPartition[localIndex];
}

// TODO:adjust to new structure
template<typename T>
void msl::DM<T>::setLocal(int localIndex, const T& v) {
  if (localIndex < nCPU) {
    localPartition[localIndex] = v;
  } else if (localIndex >= nLocal)
    throws(detail::NonLocalAccessException());
  else { // localIndex refers to a GPU 
     // approach 1: easy, but inefficient
     // if (!cpuMemoryInSync)
     //  download();
     // localPartition[localIndex] = v;
     // upload(); //easy, but inefficient
     // approach 2: more efficient, but also more complicated:
     // TODO adjust to new!
      int gpuId = (localIndex - nCPU) / nGPU;
      int idx = localIndex - nCPU - gpuId * nGPU; 
      //printf("setLocal: localIndex: %i, v:, %i, gpuId: %i, idx: %i, size: %i\n", // debug
        //     localIndex, v, gpuId, idx, sizeof(T));                              // debug
      cudaSetDevice(gpuId);
      CUDA_CHECK_RETURN(
          cudaMemcpy(&(plans[gpuId].d_Data[idx]), &v, sizeof(T), cudaMemcpyHostToDevice));
  }
}

template<typename T>
void msl::DM<T>::set(int globalIndex, const T& v) {
  if ((globalIndex >= firstIndex) && (globalIndex < firstIndex + nLocal)) {
    setLocal(globalIndex - firstIndex, v);
  }
}

template<typename T>
GPUExecutionPlan<T>* msl::DM<T>::getExecPlans(){
  //std::vector<GPUExecutionPlan<T> > ret(plans, plans + Muesli::num_gpus);
  return plans;
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

template<typename T>
void msl::DM<T>::upload() {
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
void msl::DM<T>::download() {
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
int msl::DM<T>::getGpuId(int index) const {
  return(index - firstIndex - nCPU) / nGPU;
}

// method (only) useful for debbuging
template<typename T>
void msl::DM<T>::showLocal(const std::string& descr) {
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
void msl::DM<T>::show(const std::string& descr) {
  T* b = new T[n];
  std::ostringstream s;
  if (descr.size() > 0)
    s << descr << ": " << std::endl;
  if (!cpuMemoryInSync) {
    download();
  }
  msl::allgather(localPartition, b, nLocal);

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
void msl::DM<T>::broadcastPartition(int partitionIndex) {
  if (partitionIndex < 0 || partitionIndex >= np) {
    throws(detail::IllegalPartitionException());
  }
  if (!cpuMemoryInSync) 
    download();
  msl::MSL_Broadcast(partitionIndex, localPartition, nLocal);
  cpuMemoryInSync = false;
  upload();
}

// SKELETONS / COMMUNICATION / GATHER

template<typename T>
void msl::DM<T>::gather(msl::DM<T>& da) {
  printf("gather\n");
  throws(detail::NotYetImplementedException());
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
// inline void msl::DM<T>::permutePartition(int (*f)(int)) {
//  permutePartition(curry(f));
//}


template<typename T>
void msl::DM<T>::freeDevice() {
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

//*********************************** Maps ********************************

template <typename T>
template <typename MapFunctor>
void msl::DM<T>::mapInPlace(MapFunctor& f){
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, plans[i].d_Data, plans[i].size, f);  // in, out, #bytes, function
  }

  #pragma omp parallel for
  for (int k = 0; k < nCPU; k++){
    localPartition[k] = f(localPartition[k]);
  }
  msl::syncStreams();
  cpuMemoryInSync = false;
}

template <typename T>
template <typename MapIndexFunctor>
void msl::DM<T>::mapIndexInPlace(MapIndexFunctor& f){

//  int colGPU = (ncol * (1 - Muesli::cpu_fraction)) / ng;  // is not used, HK
  for (int i = 0; i < ng; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, plans[i].d_Data, plans[i].nLocal, plans[i].first, f, ncol);
  }

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

template <typename T>
template <typename F>
msl::DM<T> msl::DM<T>::map(F& f) { //preliminary simplification in order to avoid type error
  DM<T> result(n);                 // should be: DM<R>

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
    //          plans[i].d_Data, plans[i].d_Data, plans[i].size, f); // in, out, #bytes, function
            plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].size, f);
  }

  #pragma omp parallel for
  for (int k = 0; k < nCPU; k++){
    result.setLocal(k, f(localPartition[k]));
  }

  // check for errors during gpu computation
  msl::syncStreams();
  result.setCpuMemoryInSync(false);
  result.download();

  return result;
}

template <typename T>
template <typename MapIndexFunctor>
msl::DM<T> msl::DM<T>::mapIndex(MapIndexFunctor& f){
  DM<T> result(nrow,ncol);

  // map on GPUs
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
              plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].nLocal,
              plans[i].first, f, ncol);
  }

  #pragma omp parallel for
  for (int k = 0; k < nCPU; k++){
      int j = (k + firstIndex) / ncol;
      int i = (k + firstIndex) % ncol;
      result.setLocal(k, f(i, j, localPartition[k]));
  }

  // check for errors during gpu computation
  msl::syncStreams();
  result.setCpuMemoryInSync(false);
  result.download();

  return result;
}

template <typename T>
template <typename MapStencilFunctor>
void msl::DM<T>::mapStencilInPlace(MapStencilFunctor& f, T neutral_value)
{
  printf("mapStencilInPlace\n");
  throws(detail::NotYetImplementedException());
}

template <typename T>
template <typename R, typename MapStencilFunctor>
msl::DM<R> msl::DM<T>::mapStencil(MapStencilFunctor& f, T neutral_value)
{
  printf("mapStencil\n");
  throws(detail::NotYetImplementedException());
}

// ************************************ zip ***************************************
template <typename T>
template <typename T2, typename ZipFunctor>
void msl::DM<T>::zipInPlace(DM<T2>& b, ZipFunctor& f){
  // zip on GPU
  for (int i = 0; i < ng; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    auto bplans = b.getExecPlans();
    detail::zipKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, bplans[i].d_Data, plans[i].d_Data, plans[i].size, f);
  }

  T2* bPartition = b.getLocalPartition();
  #pragma omp parallel for
  for (int k = 0; k < nCPU; k++) {
    localPartition[k] = f(localPartition[k], bPartition[k]);
  }

  // check for errors during gpu computation
  msl::syncStreams();
  cpuMemoryInSync = false;
}

template <typename T>
template <typename T2, typename ZipFunctor>
msl::DM<T> msl::DM<T>::zip(DM<T2>& b, ZipFunctor& f){   // should have result type DA<R>; debug
  DM<T> result(nrow,ncol);
  // zip on GPUs
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::zipKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].size, f);
  }
  // zip on CPU cores
  T2* bPartition = b.getLocalPartition();
  #pragma omp parallel for
  for (int k = 0; k < nCPU; k++) {
    result.setLocal(k, f(localPartition[k], bPartition[k]));
  }
  // check for errors during gpu computation
  msl::syncStreams();
  result.setCpuMemoryInSync(false);
  return result;
}


template <typename T>
template <typename T2, typename ZipIndexFunctor>
void msl::DM<T>::zipIndexInPlace(DM<T2>& b, ZipIndexFunctor& f){
  // zip on GPUs
  for (int i = 0; i < ng; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::zipIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, plans[i].d_Data, plans[i].nLocal,
        plans[i].first, f, ncol);
  }

  T2* bPartition = b.getLocalPartition();
  #pragma omp parallel for
  for (int k = 0; k < nCPU; k++){
      int i = (k + firstIndex) / ncol;
      int j = (k + firstIndex) % ncol;
      localPartition[k] = f(i, j, localPartition[k], bPartition[k]);
  }
  // check for errors during gpu computation
  msl::syncStreams();
  cpuMemoryInSync = false;
}

template <typename T>
template <typename T2, typename ZipIndexFunctor>
msl::DM<T> msl::DM<T>::zipIndex(DM<T2>& b, ZipIndexFunctor& f){
  DM<T> result(nrow,ncol);
  // zip on GPUs
  for (int i = 0; i < ng; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::zipIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].nLocal,
        plans[i].first, f, ncol);
  }
  // zip on CPU cores
  T2* bPartition = b.getLocalPartition();
  #pragma omp parallel for
  for (int k = 0; k < nCPU; k++){
      int i = (k + firstIndex) / ncol;
      int j = (k + firstIndex) % ncol;
      result.setLocal(k, f(i, j, localPartition[k], bPartition[k]));
  }
  // check for errors during gpu computation
  msl::syncStreams();
  result.setCpuMemoryInSync(false);
  return result;
}

template <typename T>
template <typename T2, typename T3, typename ZipFunctor>
void msl::DM<T>::zipInPlace3(DM<T2>& b, DM<T3>& c, ZipFunctor& f){
  // zip on GPU
  for (int i = 0; i < ng; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    auto bplans = b.getExecPlans();
    auto cplans = c.getExecPlans();
    detail::zipKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
                plans[i].d_Data, bplans[i].d_Data, cplans[i].d_Data, plans[i].d_Data,
		plans[i].nLocal, f);
  }

  T2* bPartition = b.getLocalPartition();
  T3* cPartition = c.getLocalPartition();
  #pragma omp parallel for
  for (int k = 0; k < nCPU; k++) {
    localPartition[k] = f(localPartition[k], bPartition[k], cPartition[k]);
  }

  // check for errors during gpu computation
  msl::syncStreams();
  cpuMemoryInSync = false;
}

template <typename T>
template <typename T2, typename T3, typename T4, typename ZipFunctor>
void msl::DM<T>::zipInPlaceAAM(DA<T2>& b, DA<T3>& c, DM<T4>& d, ZipFunctor& f){
  // zip on GPU
  for (int i = 0; i < ng; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    auto bplans = b.getExecPlans();
    auto cplans = c.getExecPlans();
    auto dplans = d.getExecPlans();
    detail::zipKernelAAM<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, bplans[i].d_Data, cplans[i].d_Data, d.plans[i].d_Data, plans[i].d_Data,
		plans[i].nLocal, plans[i].first, bplans[i].first, f, ncol);
  }

  T2* bPartition = b.getLocalPartition(); 
  T3* cPartition = c.getLocalPartition();
  T4* dPartition = d.getLocalPartition();
  int bfirst = b.getFirstIndex();
  #pragma omp parallel for
  for (int k = 0; k < nCPU; k++) {
    int i = ((k + firstIndex) / ncol) - bfirst;
//    printf("k=%d, i=%d, firstIndex=%d, ncol=%d, localPartition[k]=%d, 
//            bPartition[i]=%d, cPartition[i]=%d, dPartition[k]=%d, bfirst=%d\n",
//            k, i, firstIndex, ncol, localPartition[k], bPartition[i], cPartition[i], dPartition[k], bfirst); // debug
    localPartition[k] = f(localPartition[k], bPartition[i], cPartition[i], dPartition[k]);
  }

  // check for errors during gpu computation
  msl::syncStreams();
  cpuMemoryInSync = false;
}

// *********** fold *********************************************
template <typename T>
template <typename FoldFunctor>
T msl::DM<T>::fold(FoldFunctor& f, bool final_fold_on_cpu){
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
  for (int k = 1; k<nCPU; k++) {
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



