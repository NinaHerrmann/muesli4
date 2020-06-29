/*
 * da.cu
 *
 *      Author: Herbert Kuchen <kuchen@uni-muenster.de>, S
 *              Steffen Ernsting <s.ernsting@uni-muenster.de>
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
msl::DA<T>::DA():                  // distributed array (resides on GPUs until deleted!)
      n(0),                        // number of elements of distributed array
      nLocal(0),                   // number of local elements on a node    
      np(0),                       // number of (MPI-) nodes (= Muesli::num_local_procs)
      id(0),                       // id of local node among all nodes (= Muesli::proc_id)
      localPartition(0),           // local partition of the DA
      cpuMemoryFlag(true),         // is GPU memory in sync with CPU?
      firstIndex(0),               // first global index of the DA on the local partition
      plans(0),                    // GPU execution plans
      dist(Distribution::DIST),    // distribution of DA: DIST (distributed) or COPY (for now: always DIST)
      gpuCopyDistributed(0),       // is GPU copy distributed? (for now: always "false")
// new: for combined usage of CPU and GPUs on every MPI-node
      ng(0),                       // number of GPUs per node (= Muesli::num_gpus)
      nGPU(0),                     // number of elements per GPU (all the same!)
      nCPU(0)                      // number of elements on CPU = nLocal - ng*nGPU
{}

// constructor creates a non-initialized DA
template<typename T>
msl::DA<T>::DA(int size)
    : n(size) {
  init();
#ifdef __CUDACC__
  initGPUs(); 
#else 
  localPartition = new T[nLocal];
#endif
  cpuMemoryFlag = true;
}

// constructor creates a DA, initialized with v
template<typename T>
msl::DA<T>::DA(int size, const T& v)
    : n(size) {
  init();
#ifdef __CUDACC__
  initGPUs(); 
#else 
  localPartition = new T[nLocal];
#endif
  #pragma omp parallel for
  for (int i=0; i< nLocal; i++) localPartition[i] = v; 
  upload();
}

// auxiliary method init() 
template<typename T>
void msl::DA<T>::init() {
  if (Muesli::proc_entrance == UNDEFINED) {
    throws(detail::MissingInitializationException());}
  id = Muesli::proc_id;
  np = Muesli::num_local_procs;
  ng = Muesli::num_gpus;
  nLocal = n / np;
  nGPU = ng > 0 ? nLocal * (1.0 - Muesli::cpu_fraction) / ng : 0;
  nCPU = nLocal - nGPU * ng;
  firstIndex = id * nLocal;
  // for debugging:
  printf("id: %i, n: %i, ng: %i, nLocal: %i, nGPU: %i, nCPU: %i, firstIndex: %i\n", id, n, ng, nLocal, nGPU, nCPU, firstIndex);
}

// auxiliary method initGPUs
template<typename T>
void msl::DA<T>::initGPUs() {
#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, nLocal*sizeof(T)));
  plans = new GPUExecutionPlan<T>[ng];
  int gpuBase = nCPU;
  for (int i = 0; i<ng; i++) {
    cudaSetDevice(i);
    plans[i].size = nGPU;
    plans[i].nLocal = plans[i].size; //verdächtig, HK
    plans[i].bytes = plans[i].size * sizeof(T);
    plans[i].first = gpuBase + firstIndex;
    plans[i].h_Data = localPartition + gpuBase;
    CUDA_CHECK_RETURN(cudaMalloc(&plans[i].d_Data,plans[i].bytes));
    gpuBase += plans[i].size;
  }
#endif
}

// destructor removes a DA
template<typename T>
msl::DA<T>::~DA() {
#ifdef __CUDACC__
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
}

// ***************************** auxiliary methods ******************************
template<typename T>
T* msl::DA<T>::getLocalPartition() {
  if (!cpuMemoryFlag)
    download();
  return localPartition;
}

template<typename T>
T msl::DA<T>::get(int index) const {
  int idSource;
  T message;

  // element with global index is locally stored
  if (isLocal(index)) {
#ifdef __CUDACC__
    // element might not be up to date in cpu memory
    if (!cpuMemoryFlag) {
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
int msl::DA<T>::getSize() const {
  return n;
}

template<typename T>
int msl::DA<T>::getLocalSize() const {
  return nLocal;
}

template<typename T>
int msl::DA<T>::getFirstIndex() const {
  return firstIndex;
}

template<typename T>
bool msl::DA<T>::isLocal(int index) const {
  return (index >= firstIndex) && (index < firstIndex + nLocal);
}

template<typename T>
T msl::DA<T>::getLocal(int localIndex) const {
  if (localIndex >= nLocal) 
    throws(detail::NonLocalAccessException());
  if ((!cpuMemoryFlag) && (localIndex >= nCPU))
    download(); // easy, but inefficient
  return localPartition[localIndex];
}

template<typename T>
void msl::DA<T>::setLocal(int localIndex, const T& v) {
  if (localIndex >= nLocal) 
    throws(detail::NonLocalAccessException());
  if (!cpuMemoryFlag) {
    download();
  }
  localPartition[localIndex] = v;
  if (localIndex >= nCPU)
    upload(); //easy, but inefficient
}

template<typename T>
void msl::DA<T>::set(int globalIndex, const T& v) {
  if ((globalIndex >= firstIndex) && (globalIndex < firstIndex + nLocal)) {
    setLocal(globalIndex - firstIndex, v);
  }
}

template<typename T>
GPUExecutionPlan<T>* msl::DA<T>::getExecPlans(){
  //std::vector<GPUExecutionPlan<T> > ret(plans, plans + Muesli::num_gpus);
  return plans;
}

template<typename T>
void msl::DA<T>::upload() {
  std::vector<T*> dev_pointers;

#ifdef __CUDACC__
  if (!cpuMemoryFlag) {
    for (int i = 0; i < ng; i++) {
      cudaSetDevice(i);
      // upload data
      CUDA_CHECK_RETURN(
        cudaMemcpyAsync(plans[i].d_Data, plans[i].h_Data, plans[i].bytes, 
                        cudaMemcpyHostToDevice, Muesli::streams[i]));
    }
    cpuMemoryFlag = true;
  } 
#endif
  return;
}

template<typename T>
void msl::DA<T>::download() {
#ifdef __CUDACC__
  if (!cpuMemoryFlag) {
    for (int i = 0; i< ng; i++) {
      cudaSetDevice(i);

      // download data from device
      CUDA_CHECK_RETURN(
          cudaMemcpyAsync(plans[i].h_Data, plans[i].d_Data, plans[i].bytes,
              cudaMemcpyDeviceToHost, Muesli::streams[i]));

      // free data on device (deleted, since DA now resides on GPUs)
      // cudaFree(plans[i].d_Data);
      // plans[i].d_Data = 0;
    }
    // wait until download is finished
    for (int i = 0; i < ng; i++) {
      CUDA_CHECK_RETURN(cudaStreamSynchronize(Muesli::streams[i]));
    }
    cpuMemoryFlag = true;
  }
#endif
}

template<typename T>
int msl::DA<T>::getGpuId(int index) const {
  return(index - firstIndex - nCPU) / ng;
}

// method (only) useful for debbuging
template<typename T>
void msl::DA<T>::showLocal(const std::string& descr) {
  if (!cpuMemoryFlag) {
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
void msl::DA<T>::show(const std::string& descr) {
  T* b = new T[n];
  std::ostringstream s;
  if (descr.size() > 0)
    s << descr << ": " << std::endl;
  if (!cpuMemoryFlag) {  
    download();
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

  delete b;

  if (msl::isRootProcess()) printf("%s", s.str().c_str());
}


// SKELETONS / COMMUNICATION / BROADCAST PARTITION

template<typename T>
void msl::DA<T>::broadcastPartition(int partitionIndex) {
  if (partitionIndex < 0 || partitionIndex >= np) {
    throws(detail::IllegalPartitionException());
  }
  if (!cpuMemoryFlag) 
    download();
  msl::MSL_Broadcast(partitionIndex, localPartition, nLocal);
  cpuMemoryFlag = false;
  upload();
}

// SKELETONS / COMMUNICATION / GATHER

template<typename T>
void msl::DA<T>::gather(msl::DA<T>& da) {
  printf("gather\n");
  throws(detail::NotYetImplementedException());
}

// SKELETONS / COMMUNICATION / PERMUTE PARTITION

template<typename T>
template<typename Functor>
inline void msl::DA<T>::permutePartition(Functor& f) {
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
    if (!cpuMemoryFlag) 
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
    cpuMemoryFlag = false;
    upload();
  }
}

// template<typename T>
// inline void msl::DA<T>::permutePartition(int (*f)(int)) {
//  permutePartition(curry(f));
//}


template<typename T>
void msl::DA<T>::freeDevice() {
#ifdef __CUDACC__
  if (!cpuMemoryFlag) {
    for (int i = 0; i < ng; i++) {
    //  if(plans[i].d_Data == 0) {
    //    continue;
    //  }
      cudaFree(plans[i].d_Data);
      plans[i].d_Data = 0;
    }
  cpuMemoryFlag = true;
  }
#endif
}

//*********************************** Maps ********************************

template <typename T>
template <typename MapFunctor>
void msl::DA<T>::mapInPlace(MapFunctor& f)
{
  mapInPlace(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapFunctor)>());
}

template <typename T>
template <typename MapFunctor>
void msl::DA<T>::mapInPlace(MapFunctor& f, Int2Type<true>){
  // set kernel configuration
  int tile_width = f.getTileWidth();
  int threads = Muesli::threads_per_block;
  if (tile_width != -1) {
    threads = tile_width;
  }

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].first);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(threads);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    printf("dimGrid: %i, dimBlock: %i, smem_bytes: %i, f(2): %i\n", dimGrid, dimBlock, smem_bytes, f(2));
    detail::mapKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
        plans[i].d_Data, plans[i].d_Data, plans[i].size, f);
  }
  
  f.init(nCPU, firstIndex);
  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    localPartition[i] = f(localPartition[i]);
  }
  // check for errors during gpu computation
  msl::syncStreams();
  cpuMemoryFlag = false;
}

template <typename T>
template <typename MapFunctor>
void msl::DA<T>::mapInPlace(MapFunctor& f, Int2Type<false>){
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, plans[i].d_Data, plans[i].size, f);  // in, out, #bytes, function
  }
  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    localPartition[i] = f(localPartition[i]);
  }
  msl::syncStreams();
  cpuMemoryFlag = false;
}

template <typename T>
template <typename MapIndexFunctor>
void msl::DA<T>::mapIndexInPlace(MapIndexFunctor& f)
{
  mapIndexInPlace(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapIndexFunctor)>());
}

template <typename T>
template <typename MapIndexFunctor>
void msl::DA<T>::mapIndexInPlace(MapIndexFunctor& f, Int2Type<true>)
{
  // upload data first (if necessary)
  // upload();

  // set kernel configuration
  int tile_width = f.getTileWidth();
  int threads = Muesli::threads_per_block;
  if (tile_width != -1) {
    threads = tile_width;
  }

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].first);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(threads);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapIndexKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
        plans[i].d_Data, plans[i].d_Data, plans[i].nLocal, plans[i].first, f, f.useLocalIndices());
  }

  f.init(nCPU, firstIndex);

  // calculate offsets for indices
  // int offset = f.useLocalIndices() ? 0 : firstIndex;
  int offset = firstIndex;

  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    localPartition[i] = f(i+offset, localPartition[i]);
  }
  // check for errors during gpu computation
  msl::syncStreams();
  cpuMemoryFlag = false;
}

template <typename T>
template <typename MapIndexFunctor>
void msl::DA<T>::mapIndexInPlace(MapIndexFunctor& f, Int2Type<false>)
{
  // upload data first (if necessary)
  // upload();

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, plans[i].d_Data, plans[i].nLocal, plans[i].first, f, false);
  }
  // calculate offsets for indices
  // int offset = f.useLocalIndices() ? 0 : firstIndex;
  int offset = firstIndex;

  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    localPartition[i] = f(i+offset, localPartition[i]);
  }
  // check for errors during gpu computation
  msl::syncStreams();
  cpuMemoryFlag = false;
}

template <typename T>
template <typename R, typename MapFunctor>
msl::DA<R> msl::DA<T>::map(MapFunctor& f)
{
  return map<R, MapFunctor>(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapFunctor)>());
}

template <typename T>
template <typename R, typename MapFunctor>
msl::DA<R> msl::DA<T>::map(MapFunctor& f, Int2Type<true>)
{
  DA<R> result(n, dist);

  // upload data first (if necessary)
  // upload();
  // result.upload(1); // alloc only

  // set kernel configuration
  int tile_width = f.getTileWidth();
  int threads = Muesli::threads_per_block;
  if (tile_width != -1) {
    threads = tile_width;
  }

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].first);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(threads);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
            plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].size, f);
  }

  f.init(nCPU, firstIndex);
  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    result.setLocal(i, f(localPartition[i]));
  }

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template <typename T>
template <typename R, typename MapFunctor>
msl::DA<R> msl::DA<T>::map(MapFunctor& f, Int2Type<false>)
{
  DA<R> result(n, dist);

  // upload data first (if necessary)
  // upload();
  // result.upload(1); // alloc only

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
            plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].size, f);
  }

  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    result.setLocal(i, f(localPartition[i]));
  }

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template <typename T>
template <typename R, typename MapIndexFunctor>
msl::DA<R> msl::DA<T>::mapIndex(MapIndexFunctor& f)
{
  return mapIndex<R, MapIndexFunctor>(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapIndexFunctor)>());
}

template <typename T>
template <typename R, typename MapIndexFunctor>
msl::DA<R> msl::DA<T>::mapIndex(MapIndexFunctor& f, Int2Type<true>)
{
  DA<R> result(n, dist);

  // upload data first (if necessary)
  // upload();
  // result.upload(1); // alloc only

  // set kernel configuration
  int tile_width = f.getTileWidth();
  int threads = Muesli::threads_per_block;
  if (tile_width != -1) {
    threads = tile_width;
  }

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].first);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(threads);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapIndexKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
              plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].nLocal,
              plans[i].first, f, f.useLocalIndices());
  }

  f.init(nCPU, firstIndex);
  // calculate offsets for indices
  int offset = f.useLocalIndices() ? 0 : firstIndex;

  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    result.setLocal(i, f(i+offset, localPartition[i]));
  }

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template <typename T>
template <typename R, typename MapIndexFunctor>
msl::DA<R> msl::DA<T>::mapIndex(MapIndexFunctor& f, Int2Type<false>)
{
  DA<R> result(n, dist);

  // upload data first (if necessary)
  upload();
  result.upload(1); // alloc only

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
              plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].nLocal,
              plans[i].first, f, false);
  }

  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    result.setLocal(i, f(i, localPartition[i]));
  }
  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template <typename T>
template <typename MapStencilFunctor>
void msl::DA<T>::mapStencilInPlace(MapStencilFunctor& f, T neutral_value)
{
  printf("mapStencilInPlace\n");
  throws(detail::NotYetImplementedException());
}

template <typename T>
template <typename R, typename MapStencilFunctor>
msl::DA<R> msl::DA<T>::mapStencil(MapStencilFunctor& f, T neutral_value)
{
  printf("mapStencil\n");
  throws(detail::NotYetImplementedException());
}

// ************************************ zip ***************************************
template <typename T>
template <typename T2, typename ZipFunctor>
void msl::DA<T>::zipInPlace(DA<T2>& b, ZipFunctor& f)
{
  zipInPlace(b, f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipFunctor)>());
}

template <typename T>
template <typename T2, typename ZipFunctor>
void msl::DA<T>::zipInPlace(DA<T2>& b, ZipFunctor& f, Int2Type<true>)
{
  // upload data first (if necessary)
  // upload();
  // b.upload();

  // set kernel configuration
  int tile_width = f.getTileWidth();
  int threads = Muesli::threads_per_block;
  if (tile_width != -1) {
    threads = tile_width;
  }

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].first);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(threads);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::zipKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, plans[i].d_Data, plans[i].size, f);
  }

  f.init(nCPU, firstIndex);
  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    localPartition[i] = f(localPartition[i], b.getLocal(i));
  }
  // check for errors during gpu computation
  msl::syncStreams();
  cpuMemoryFlag = false;
}

template <typename T>
template <typename T2, typename ZipFunctor>
void msl::DA<T>::zipInPlace(DA<T2>& b, ZipFunctor& f, Int2Type<false>)
{
  // upload data first (if necessary)
  // upload();
  // b.upload();

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    auto bplans = b.getExecPlans();
    detail::zipKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, bplans[i].d_Data, plans[i].d_Data, plans[i].size, f);
  }

  T* bPartition = b.getLocalPartition();
  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    localPartition[i] = f(localPartition[i], bPartition[i]); 
  }
  // check for errors during gpu computation
  msl::syncStreams();
  cpuMemoryFlag = false;
}

template <typename T>
template <typename T2, typename ZipIndexFunctor>
void msl::DA<T>::zipIndexInPlace(DA<T2>& b, ZipIndexFunctor& f)
{
  zipIndexInPlace(b, f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipIndexFunctor)>());
}

template <typename T>
template <typename T2, typename ZipIndexFunctor>
void msl::DA<T>::zipIndexInPlace(DA<T2>& b, ZipIndexFunctor& f, Int2Type<true>)
{
  // upload data first (if necessary)
  // upload();
  // b.upload();

  // set kernel configuration
  int tile_width = f.getTileWidth();
  int threads = Muesli::threads_per_block;
  if (tile_width != -1) {
    threads = tile_width;
  }

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].first);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(threads);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::zipIndexKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, plans[i].d_Data, plans[i].nLocal,
        plans[i].first, f, f.useLocalIndices());
  }

  f.init(nCPU, firstIndex);
  // calculate offsets for indices
  int offset = f.useLocalIndices() ? 0 : firstIndex;

  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    setLocal(i, f(i+offset, localPartition[i], b.getLocal(i)));
  }
  // check for errors during gpu computation
  msl::syncStreams();
}

template <typename T>
template <typename T2, typename ZipIndexFunctor>
void msl::DA<T>::zipIndexInPlace(DA<T2>& b, ZipIndexFunctor& f, Int2Type<false>)
{
  // upload data first (if necessary)
  // upload();
  // b.upload();

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::zipIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, plans[i].d_Data, plans[i].nLocal,
        plans[i].first, f, false);
  }

  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    setLocal(i, f(i, localPartition[i], b.getLocal(i)));
  }
  // check for errors during gpu computation
  msl::syncStreams();
}

template <typename T>
template <typename R, typename T2, typename ZipFunctor>
msl::DA<R> msl::DA<T>::zip(DA<T2>& b, ZipFunctor& f)
{
  return zip<R>(b, f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipFunctor)>());
}

template <typename T>
template <typename R, typename T2, typename ZipFunctor>
msl::DA<R> msl::DA<T>::zip(DA<T2>& b, ZipFunctor& f, Int2Type<true>)
{
  DA<R> result(n, dist);

  // upload data first (if necessary)
  // upload();
  // b.upload();
  // result.upload(1); // alloc only

  // set kernel configuration
  int tile_width = f.getTileWidth();
  int threads = Muesli::threads_per_block;
  if (tile_width != -1) {
    threads = tile_width;
  }

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].first);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(threads);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::zipKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].size, f);
  }

  f.init(nCPU, firstIndex);
  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    result.setLocal(i, f(localPartition[i], b.getLocal(i)));
  }

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template <typename T>
template <typename R, typename T2, typename ZipFunctor>
msl::DA<R> msl::DA<T>::zip(DA<T2>& b, ZipFunctor& f, Int2Type<false>)
{
  DA<R> result(n, dist);

  // upload data first (if necessary)
  // upload();
  // b.upload();
  // result.upload(1); // alloc only

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::zipKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].size, f);
  }

  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    result.setLocal(i, f(localPartition[i], b.getLocal(i)));
  }
  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template <typename T>
template <typename R, typename T2, typename ZipIndexFunctor>
msl::DA<R> msl::DA<T>::zipIndex(DA<T2>& b, ZipIndexFunctor& f)
{
  return zipIndex<R>(b, f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipIndexFunctor)>());
}

template <typename T>
template <typename R, typename T2, typename ZipIndexFunctor>
msl::DA<R> msl::DA<T>::zipIndex(DA<T2>& b, ZipIndexFunctor& f, Int2Type<true>)
{
  DA<R> result(n, dist);

  // upload data first (if necessary)
  // upload();
  // b.upload();
  // result.upload(1); // alloc only

  // set kernel configuration
  int tile_width = f.getTileWidth();
  int threads = Muesli::threads_per_block;
  if (tile_width != -1) {
    threads = tile_width;
  }

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].first);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(threads);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::zipIndexKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].nLocal,
        plans[i].first, f, f.useLocalIndices());
  }

  f.init(nCPU, firstIndex);
  // calculate offsets for indices
  int offset = f.useLocalIndices() ? 0 : firstIndex;

  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    result.setLocal(i, f(i+offset, localPartition[i], b.getLocal(i)));
  }
  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template <typename T>
template <typename R, typename T2, typename ZipIndexFunctor>
msl::DA<R> msl::DA<T>::zipIndex(DA<T2>& b, ZipIndexFunctor& f, Int2Type<false>)
{
  DA<R> result(n, dist);

  // upload data first (if necessary)
  // upload();
  // b.upload();
  // result.upload(1); // alloc only

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::zipIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].nLocal,
        plans[i].first, f, false);
  }

  #pragma omp parallel for
  for (int i = 0; i < nCPU; i++) {
    result.setLocal(i, f(i, localPartition[i], b.getLocal(i)));
  }
  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

// *********** fold *********************************************
template <typename T>
template <typename FoldFunctor>
T msl::DA<T>::fold(FoldFunctor& f, bool final_float_on_cpu)
{
  return fold(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, FoldFunctor)>(), final_float_on_cpu);
}

template <typename T>
template <typename FoldFunctor>
T msl::DA<T>::fold(FoldFunctor& f, Int2Type<true>, bool final_fold_on_cpu)
{
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

  // upload(); done before

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
    f.notify();
    cudaSetDevice(i);
    detail::reduce<T, FoldFunctor>(plans[i].size, plans[i].d_Data, d_odata[i], threads[i], blocks[i], f, Muesli::streams[i], i);
  }

  
  msl::syncStreams();

  // fold on gpus: step 2
  for (int i = 0; i < Muesli::num_gpus; i++) {
    if (blocks[i] > 1) {
      f.notify();
      cudaSetDevice(i);
      int threads = (detail::nextPow2(blocks[i]) == blocks[i]) ? blocks[i] : detail::nextPow2(blocks[i])/2;
      detail::reduce<T, FoldFunctor>(blocks[i], d_odata[i], d_odata[i], threads, 1, f, Muesli::streams[i], i);
    }
  }
  // fold local elements on CPU (overlap with GPU computations)
  T cpu_result = localPartition[0];
  for (int i = 1; i < nCPU; i++) 
    cpu_result = f(cpu_result, localPartition[i]);

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
      f.notify();
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

// ******************* fold ******************
template <typename T>
template <typename FoldFunctor>
T msl::DA<T>::fold(FoldFunctor& f, Int2Type<false>, bool final_fold_on_cpu)
{
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
  T cpu_result = localPartition[0];
  for (int i = 1; i<nCPU; i++) 
    cpu_result = f(cpu_result, localPartition[i]);

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


