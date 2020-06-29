/*
 * darray_common.cpp
 *
 *      Author: Steffen Ernsting <s.ernsting@uni-muenster.de>
 * 
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2014 Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *                Herbert Kuchen <kuchen@uni-muenster.de.
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

template<typename T>
msl::DArray<T>::DArray()
    : n(0),
      nLocal(0),
      np(0),
      id(0),
      localPartition(0),
      cpuMemoryFlag(1),
      firstIndex(0),
      plans(0),
      dist(Distribution::DIST),
      gpuCopyDistributed(0) {
}

template<typename T>
msl::DArray<T>::DArray(int size, Distribution d)
    : n(size),
      dist(d) {
  init();

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, nLocal*sizeof(T)));
  initGPU();
#else
  localPartition = new T[nLocal];
#endif

  cpuMemoryFlag = true;
}

template<typename T>
msl::DArray<T>::DArray(int size, const T& initial_value, Distribution d)
    : n(size),
      dist(d) {
  init();

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, nLocal*sizeof(T)));
  initGPU();
#else
  localPartition = new T[nLocal];
#endif

  fill(initial_value);

  cpuMemoryFlag = true;
}

template<typename T>
msl::DArray<T>::DArray(int size, T* const initial_array, Distribution d, bool root_init)
    : n(size),
      dist(d) {
  init();

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, nLocal*sizeof(T)));
  initGPU();
#else
  localPartition = new T[nLocal];
#endif

  if(root_init){
    fill_root_init(initial_array);
  }else{
    fill(initial_array);
  }

  cpuMemoryFlag = true;
}

template<typename T>
msl::DArray<T>::DArray(int size, T (*f)(int), Distribution d)
    : n(size),
      dist(d) {
  init();

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, nLocal*sizeof(T)));
  initGPU();
#else
  localPartition = new T[nLocal];
#endif

  fill(f);

  cpuMemoryFlag = true;
}

template<typename T>
template<typename F2>
msl::DArray<T>::DArray(int size, const F2& f, Distribution d)
    : n(size),
      dist(d) {
  init();

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, nLocal*sizeof(T)));
  initGPU();
#else
  localPartition = new T[nLocal];
#endif

  fill(f);

  cpuMemoryFlag = true;
}

template<typename T>
msl::DArray<T>::DArray(const DArray<T>& cs)
    : n(cs.n),
      dist(cs.dist) {
  init();

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, nLocal*sizeof(T)));
  plans = new GPUExecutionPlan<T> [Muesli::num_gpus];
  gpuCopyDistributed = cs.gpuCopyDistributed;

  int gpuBase = 0;
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    plans[i] = cs.plans[i];
    if (!gpuCopyDistributed) {
      plans[i].h_Data = localPartition + gpuBase;
      gpuBase += plans[i].size;
    } else {
      plans[i].h_Data = localPartition;
    }
    plans[i].d_Data = 0;
  }

  cpuMemoryFlag = cs.cpuMemoryFlag;

  // if data is not up to date in main memory, copy data on device
  if (!cs.cpuMemoryFlag) {
    for (int i = 0; i < Muesli::num_gpus; i++) {
      cudaSetDevice(i);
      int threads = detail::getMaxThreadsPerBlock(i);
      int blocks = plans[i].size / threads + 1;
      CUDA_CHECK_RETURN(cudaMalloc(&plans[i].d_Data, plans[i].bytes));
      detail::copyKernel<<<blocks, threads, 0, Muesli::streams[i]>>>(cs.plans[i].d_Data,
          plans[i].d_Data,
          plans[i].size);
    }
  } else {  // data is up to date in main memory
    std::copy(cs.localPartition, cs.localPartition + nLocal, localPartition);
  }

#else
  localPartition = new T[nLocal];
  std::copy(cs.localPartition, cs.localPartition + nLocal, localPartition);
#endif

}

template<typename T>
msl::DArray<T>::~DArray() {
#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaFreeHost(localPartition));
  for (int i = 0; i < Muesli::num_gpus; i++) {
    if (plans[i].d_Data != 0) {
      CUDA_CHECK_RETURN(cudaFree(plans[i].d_Data));
    }
  }
  delete[] plans;
#else
  delete[] localPartition;
#endif
}

template<typename T>
msl::DArray<T>& msl::DArray<T>::operator=(const DArray<T>& rhs) {
  if (this != &rhs) {
    n = rhs.n;
    dist = rhs.dist;
    init();

    bool create_new_local_partition = false;

    if (nLocal * sizeof(T) != rhs.nLocal * sizeof(T)) {
      create_new_local_partition = true;
    }

    T* new_localPartition;

#ifdef __CUDACC__
    if(create_new_local_partition) {
      CUDA_CHECK_RETURN(cudaMallocHost(&new_localPartition, nLocal*sizeof(T)));
    }

    bool old_gpu_copy_distributed = gpuCopyDistributed;

    gpuCopyDistributed = rhs.gpuCopyDistributed;
    GPUExecutionPlan<T>* new_plans = new GPUExecutionPlan<T> [Muesli::num_gpus];

    int gpuBase = 0;
    for (int i = 0; i < Muesli::num_gpus; i++) {

      cudaSetDevice(i);
      new_plans[i] = rhs.plans[i];

      if(create_new_local_partition) {
        if (!gpuCopyDistributed) {
          new_plans[i].h_Data = new_localPartition + gpuBase;
          gpuBase += new_plans[i].size;
        } else {
          new_plans[i].h_Data = new_localPartition;
        }
        new_plans[i].d_Data = 0;
      } else {
        if(old_gpu_copy_distributed == gpuCopyDistributed) {
          new_plans[i].h_Data = plans[i].h_Data;
          new_plans[i].d_Data = plans[i].d_Data;
        } else if(!old_gpu_copy_distributed) {
          new_plans[i].h_Data = plans[0].h_Data;
          new_plans[i].d_Data = plans[0].d_Data;
        } else {
          new_plans[i].h_Data = plans[i].h_Data + gpuBase;
          if(plans[i].d_Data != 0) {
            new_plans[i].d_Data = plans[i].d_Data + gpuBase;
          } else {
            new_plans[i].d_Data = 0;
          }
          gpuBase += new_plans[i].size;
        }
      }
    }

    if(create_new_local_partition) {
      freeDevice();  // remove old data from devices
    }

    cpuMemoryFlag = rhs.cpuMemoryFlag;

    // if data is not up to date in main memory, copy data on device
    if (!rhs.cpuMemoryFlag) {
      for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        int threads = detail::getMaxThreadsPerBlock(i);
        int blocks = new_plans[i].size / threads + 1;

        if(new_plans[i].d_Data == 0) {
          CUDA_CHECK_RETURN(cudaMalloc(&new_plans[i].d_Data, new_plans[i].bytes));
        }

        detail::copyKernel<<<blocks, threads, 0, Muesli::streams[i]>>>(rhs.plans[i].d_Data,
            new_plans[i].d_Data,
            new_plans[i].size);
      }
      for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        CUDA_CHECK_RETURN(cudaStreamSynchronize(Muesli::streams[i]));
      }
    } else {  // data is up to date in main memory
      if(create_new_local_partition) {
        std::copy(rhs.localPartition, rhs.localPartition + nLocal, new_localPartition);
      } else {
        std::copy(rhs.localPartition, rhs.localPartition + nLocal, localPartition);
      }
    }

    // free old memory
    delete[] plans;

    // assign new memory to object
    plans = new_plans;

    // free old memory
    if(create_new_local_partition) {
      CUDA_CHECK_RETURN(cudaFreeHost(localPartition));
    }

#else
    if (create_new_local_partition) {
      new_localPartition = new T[nLocal];
      std::copy(rhs.localPartition, rhs.localPartition + nLocal,
                new_localPartition);
      // free old memory
      delete[] localPartition;
    } else {
      std::copy(rhs.localPartition, rhs.localPartition + nLocal,
                localPartition);
    }
#endif

    // assign new memory to object
    if (create_new_local_partition) {
      localPartition = new_localPartition;
    }
  }
  return *this;
}

template<typename T>
void msl::DArray<T>::init() {
  if (Muesli::proc_entrance == UNDEFINED) {
    throws(detail::MissingInitializationException());
  }

  id = Muesli::proc_id;
  np = Muesli::num_local_procs;

  if (dist != Distribution::COPY) {
    if (n % np != 0) {
      throws(detail::PartitioningImpossibleException());
    }
  }

  if (dist == Distribution::COPY) {
    nLocal = n;
    firstIndex = 0;
  } else {
    // for simplicity: assuming np divides n
    nLocal = n / np;
    firstIndex = id * nLocal;
  }
}

template<typename T>
void msl::DArray<T>::initGPU() {
#ifdef __CUDACC__
  plans = new GPUExecutionPlan<T>[Muesli::num_gpus];
  gpuCopyDistributed = 0;

  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    plans[i].size = nLocal / Muesli::num_gpus;
  }
  for (int i = 0; i < nLocal % Muesli::num_gpus; i++) {
    plans[i].size++;
  }
  int gpuBase = 0;
  for (int i = 0; i < Muesli::num_gpus; i++) {
    plans[i].nLocal = plans[i].size;
    plans[i].bytes = plans[i].size * sizeof(T);
    plans[i].first = gpuBase + firstIndex;
    plans[i].h_Data = localPartition + gpuBase;
    plans[i].d_Data = 0;
    gpuBase += plans[i].size;
  }
#endif
}

template<typename T>
void msl::DArray<T>::fill(const T& value) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    localPartition[i] = value;
  }
}

template<typename T>
void msl::DArray<T>::fill(T* const values) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    localPartition[i] = values[i + firstIndex];
  }
}

template<typename T>
void msl::DArray<T>::fill(T (*f)(int)) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    localPartition[i] = f(i + firstIndex);
  }
}

template<typename T>
template<typename F>
void msl::DArray<T>::fill(const F& f) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    localPartition[i] = f(i + firstIndex);
  }
}

template<typename T>
void msl::DArray<T>::fill_root_init(T* const values) {
  scatter(values, localPartition, nLocal);
}

template<typename T>
T* msl::DArray<T>::getLocalPartition() const {
  return localPartition;
}

template<typename T>
T msl::DArray<T>::get(int index) const {
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
int msl::DArray<T>::getSize() const {
  return n;
}

template<typename T>
int msl::DArray<T>::getLocalSize() const {
  return nLocal;
}

template<typename T>
int msl::DArray<T>::getFirstIndex() const {
  return firstIndex;
}

template<typename T>
bool msl::DArray<T>::isLocal(int index) const {
  return (index >= firstIndex) && (index < firstIndex + nLocal);
}

template<typename T>
T msl::DArray<T>::getLocal(int localIndex) const {
//  if (index >= nLocal) {
//    throws(detail::NonLocalAccessException());
//  }
  return localPartition[localIndex];
}

template<typename T>
void msl::DArray<T>::setLocal(int localIndex, const T& v) {
//  if (localIndex >= nLocal) {
//    throws(detail::NonLocalAccessException());
//  }

  localPartition[localIndex] = v;
}

template<typename T>
void msl::DArray<T>::set(int globalIndex, const T& v) {
#ifdef __CUDACC__
  download();
#endif

  if ((globalIndex >= firstIndex) && (globalIndex < firstIndex + nLocal)) {
    localPartition[globalIndex - firstIndex] = v;
  }
}

// SKELETONS / COMMUNICATION / BROADCAST PARTITION

template<typename T>
void msl::DArray<T>::broadcastPartition(int partitionIndex) {
  if (partitionIndex < 0 || partitionIndex >= np) {
    throws(detail::IllegalPartitionException());
  }

  bool upload_after = 0;
  if (!cpuMemoryFlag) {
    download();
    upload_after = 1;
  }

  int rootId = partitionIndex;
  msl::MSL_Broadcast(rootId, localPartition, nLocal);

  if (upload_after)
    upload();
}

// SKELETONS / COMMUNICATION / GATHER

template<typename T>
void msl::DArray<T>::gather(T* b) {
  if (dist == Distribution::COPY) {
    download();

    // if array is copy distributed, all processes store same data
    std::copy(localPartition, localPartition + n, b);
  } else {
    download();

    msl::allgather(localPartition, b, nLocal);
  }
}

template<typename T>
void msl::DArray<T>::gather(msl::DArray<T>& da) {
  if (da.dist == Distribution::COPY) {
    // download current data if necessary
    download();
    if (dist == COPY) {
      // if data is uploaded, free gpu memory.
      // necessary because data will be updated in cpu memory and needs to be uploaded again.
      if (!da.cpuMemoryFlag) {
        da.freeDevice();
      }

      // if array is copy distributed, all processes store same data
      std::copy(localPartition, localPartition + n, da.getLocalPartition());
    } else {
      // if data is uploaded, free gpu memory.
      // necessary because data will be updated in cpu memory and needs to be uploaded again.
      if (!da.cpuMemoryFlag) {
        da.freeDevice();
      }

      msl::allgather(localPartition, da.getLocalPartition(), nLocal);
    }
  }
}

// SKELETONS / COMMUNICATION / PERMUTE PARTITION

template<typename T>
template<typename F>
inline void msl::DArray<T>::permutePartition(const Fct1<int, int, F>& f) {
  if (dist == Distribution::COPY) {
    // if array is copy distributed, permuting the partitions makes no sense since
    // all processes store same data
    return;
  }

  download();

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
    T* buffer = new T[nLocal];

#ifdef __CUDACC__
    for (i = 0; i < nLocal; i++) {
      buffer[i] = localPartition[i];
    }

    MPI_Status stat;
    MPI_Request req;

    MSL_ISend(receiver, buffer, req, nLocal, msl::MYTAG);
    MSL_Recv(sender, localPartition, stat, nLocal, msl::MYTAG);
    MPI_Wait(&req, &stat);

    delete[] buffer;
#else
    MPI_Status stat[2];
    MPI_Request req[2];

    MSL_IRecv(sender, buffer, req[0], nLocal, msl::MYTAG);
    MSL_ISend(receiver, localPartition, req[1], nLocal, msl::MYTAG);

    MPI_Waitall(2, req, stat);

    T* swap = localPartition;
    localPartition = buffer;
    delete[] swap;
#endif
  }
}

template<typename T>
inline void msl::DArray<T>::permutePartition(int (*f)(int)) {
  permutePartition(curry(f));
}

template<typename T>
void msl::DArray<T>::setGpuDistribution(Distribution dist) {
#ifdef __CUDACC__

  // switch distribution mode
  if (dist == Distribution::COPY && !gpuCopyDistributed) {
    // if data is uploaded, download data and free device memory first.
    // this is necessary because the size of the gpu partitions change
    // when switching the distribution mode
    if (!cpuMemoryFlag) {
      download();
    }

    for (int i = 0; i < Muesli::num_gpus; i++) {
      plans[i].size = nLocal;
      plans[i].nLocal = plans[i].size;
      plans[i].bytes = plans[i].size * sizeof(T);
      plans[i].first = firstIndex;
      plans[i].h_Data = localPartition;
      plans[i].d_Data = 0;
    }
  } else if (gpuCopyDistributed && !(dist == Distribution::COPY)) {
    // if data is uploaded, download data and free device memory first.
    // this is necessary because the size of the gpu partitions change
    // when switching the distribution mode
    if (!cpuMemoryFlag) {
      download();
    }

    for (int i = 0; i < Muesli::num_gpus; i++) {
      plans[i].size = nLocal / Muesli::num_gpus;
    }
    for (int i = 0; i < nLocal % Muesli::num_gpus; i++) {
      plans[i].size++;
    }
    int gpuBase = 0;
    for (int i = 0; i < Muesli::num_gpus; i++) {
      plans[i].nLocal = plans[i].size;
      plans[i].bytes = plans[i].size * sizeof(T);
      plans[i].first = gpuBase + firstIndex;
      plans[i].h_Data = localPartition + gpuBase;
      plans[i].d_Data = 0;
      gpuBase += plans[i].size;
    }
  }
  gpuCopyDistributed = dist == Distribution::COPY ? 1 : 0;
#endif
}

template<typename T>
msl::Distribution msl::DArray<T>::getGpuDistribution() {
#ifdef __CUDACC__
  return gpuCopyDistributed ? Distribution::COPY : Distribution::DIST;
#else
  return Distribution::DIST;
#endif
}

template<typename T>
std::vector<T*> msl::DArray<T>::upload(bool allocOnly) {
  std::vector<T*> dev_pointers;

#ifdef __CUDACC__
  if (cpuMemoryFlag) {
    for (int i = 0; i < Muesli::num_gpus; i++) {
      cudaSetDevice(i);
      // alloc memory
      if (plans[i].d_Data == 0) {
        CUDA_CHECK_RETURN(cudaMalloc(&plans[i].d_Data, plans[i].bytes));
      }
      // upload data
      if (!allocOnly) {
        CUDA_CHECK_RETURN(
            cudaMemcpyAsync(plans[i].d_Data, plans[i].h_Data, plans[i].bytes, cudaMemcpyHostToDevice, Muesli::streams[i]));
      }
      // store device pointer
      dev_pointers.push_back(plans[i].d_Data);
    }
    cpuMemoryFlag = false;
  } else {
    for (int i = 0; i < Muesli::num_gpus; i++) {
      dev_pointers.push_back(plans[i].d_Data);
    }
  }
#else
  dev_pointers.push_back(localPartition);
#endif

  return dev_pointers;
}

template<typename T>
void msl::DArray<T>::download() {
#ifdef __CUDACC__
  if (!cpuMemoryFlag) {
    for (int i = 0; i < Muesli::num_gpus; i++) {
      cudaSetDevice(i);

      // download data from device
      CUDA_CHECK_RETURN(
          cudaMemcpyAsync(plans[i].h_Data,
              plans[i].d_Data,
              plans[i].bytes,
              cudaMemcpyDeviceToHost,
              Muesli::streams[i]));

      // free data on device
      cudaFree(plans[i].d_Data);
      plans[i].d_Data = 0;
    }
    // wait until download is finished
    for (int i = 0; i < Muesli::num_gpus; i++) {
      CUDA_CHECK_RETURN(cudaStreamSynchronize(Muesli::streams[i]));
    }
    cpuMemoryFlag = true;
  }
#endif
}

template<typename T>
void msl::DArray<T>::freeDevice() {
#ifdef __CUDACC__
  if (!cpuMemoryFlag) {
    for (int i = 0; i < Muesli::num_gpus; i++) {
      if(plans[i].d_Data == 0) {
        continue;
      }

      cudaFree(plans[i].d_Data);
      plans[i].d_Data = 0;
    }
    cpuMemoryFlag = true;
  }
#endif
}

template<typename T>
int msl::DArray<T>::getGpuId(int index) const {
  int id = 0;
  for (int i = 1; i < Muesli::num_gpus; i++) {
    if (index >= plans[i].first) {
      id = i;
    }
  }
  return id;
}

template<typename T>
std::vector<GPUExecutionPlan<T> > msl::DArray<T>::getExecPlans() {
  upload();
  std::vector<GPUExecutionPlan<T> > ret(plans, plans + Muesli::num_gpus);
  return ret;
}

template<typename T>
GPUExecutionPlan<T> msl::DArray<T>::getExecPlan(int device) {
  upload();
  return plans[device];
}

template<typename T>
void msl::DArray<T>::setCopyDistribution() {
  if (np < 2) {
    dist = Distribution::COPY;
    return;
  }

  if (dist == Distribution::DIST) {
    download();

    T* tmp = new T[n];
    gather(tmp);

    dist = Distribution::COPY;
    init();

#ifdef __CUDACC__
    CUDA_CHECK_RETURN(cudaFreeHost(localPartition));
    CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, nLocal*sizeof(T)));
    initGPU();
#else
    delete[] localPartition;
    localPartition = new T[nLocal];
#endif

    fill(tmp);
    delete[] tmp;
  }
}

template<typename T>
void msl::DArray<T>::setDistribution() {
  if (np < 2) {
    dist = Distribution::DIST;
    return;
  }

  if (dist == Distribution::COPY) {
    download();

    T* tmp = new T[n];
    gather(tmp);

    dist = Distribution::DIST;
    init();

#ifdef __CUDACC__
    CUDA_CHECK_RETURN(cudaFreeHost(localPartition));
    CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, nLocal*sizeof(T)));
    initGPU();
#else
    delete[] localPartition;
    localPartition = new T[nLocal];
#endif

    fill(tmp);
    delete[] tmp;
  }
}

template<typename T>
void msl::DArray<T>::show(const std::string& descr) {
  T* b = new T[n];
  std::ostringstream s;
  if (descr.size() > 0)
    s << descr << ": " << std::endl;

  gather(b);

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

  if (msl::isRootProcess()) {
    printf("%s", s.str().c_str());
  }
}

template<typename T>
void msl::DArray<T>::printLocal() {
  download();

  for (int i = 0; i < Muesli::num_local_procs; i++) {
    if (Muesli::proc_id == i) {
      //print
      std::cout << Muesli::proc_id << ": [" << localPartition[0];
      for (int j = 1; j < nLocal; j++) {
        std::cout << " " << localPartition[j];
      }
      std::cout << "]" << std::endl;
    }
    MPI_Barrier (MPI_COMM_WORLD);
  }
}

