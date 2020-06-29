/*
 * dmatrix_common.cpp
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
msl::DMatrix<T>::DMatrix()
    : n(0),
      m(0),
      nLocal(0),
      mLocal(0),
      localsize(0),
      np(0),
      firstRow(0),
      firstCol(0),
      nextRow(0),
      nextCol(0),
      localPosition(0),
      localRowPosition(0),
      localColPosition(0),
      blocksInRow(0),
      blocksInCol(0),
      id(0),
      localPartition(0),
      cpuMemoryFlag(1),
      firstIndex(0),
      plans(0),
      dist(Distribution::DIST),
      gpuCopyDistributed(0) {
}

template<typename T>
msl::DMatrix<T>::DMatrix(int n0, int m0, int rows, int cols, Distribution d)
    : n(n0),
      m(m0),
      dist(d) {
  init(rows, cols);

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, localsize*sizeof(T)));
  initGPU();
#else
  localPartition = new T[localsize];
#endif

  cpuMemoryFlag = true;
}

template<typename T>
msl::DMatrix<T>::DMatrix(int n0, int m0, int rows, int cols,
                         const T& initial_value, Distribution d)
    : n(n0),
      m(m0),
      dist(d) {
  init(rows, cols);

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, localsize*sizeof(T)));
  initGPU();
#else
  localPartition = new T[localsize];
#endif

  fill(initial_value);

  cpuMemoryFlag = true;
}

template<typename T>
msl::DMatrix<T>::DMatrix(int n0, int m0, int rows, int cols,
                         T* const initial_matrix, Distribution d, bool root_init)
    : n(n0),
      m(m0),
      dist(d) {
  init(rows, cols);

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, localsize*sizeof(T)));
  initGPU();
#else
  localPartition = new T[localsize];
#endif

  if(root_init){
      fill_root_init(initial_matrix);
    }else{
      fill(initial_matrix);
    }

  cpuMemoryFlag = true;
}

template<typename T>
msl::DMatrix<T>::DMatrix(int n0, int m0, int rows, int cols, T (*f)(int, int),
                         Distribution d)
    : n(n0),
      m(m0),
      dist(d) {
  init(rows, cols);

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, localsize*sizeof(T)));
  initGPU();
#else
  localPartition = new T[localsize];
#endif

  fill(f);

  cpuMemoryFlag = true;
}

template<typename T>
template<typename F2>
msl::DMatrix<T>::DMatrix(int n0, int m0, int rows, int cols, const F2& f,
                         Distribution d)
    : n(n0),
      m(m0),
      dist(d) {
  init(rows, cols);

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, localsize*sizeof(T)));
  initGPU();
#else
  localPartition = new T[localsize];
#endif

  fill(f);

  cpuMemoryFlag = true;
}

template<typename T>
msl::DMatrix<T>::DMatrix(int n0, int m0)
    : n(n0),
      m(m0),
      dist(Distribution::COPY) {
  init(1, 1);

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, localsize*sizeof(T)));
  initGPU();
#else
  localPartition = new T[localsize];
#endif

  cpuMemoryFlag = true;
}

template<typename T>
msl::DMatrix<T>::DMatrix(int n0, int m0, const T& initial_value)
    : n(n0),
      m(m0),
      dist(Distribution::COPY) {
  init(1, 1);

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, localsize*sizeof(T)));
  initGPU();
#else
  localPartition = new T[localsize];
#endif

  fill(initial_value);

  cpuMemoryFlag = true;
}

template<typename T>
msl::DMatrix<T>::DMatrix(int n0, int m0, T* const initial_matrix)
    : n(n0),
      m(m0),
      dist(Distribution::COPY) {
  init(1, 1);

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, localsize*sizeof(T)));
  initGPU();
#else
  localPartition = new T[localsize];
#endif

  fill(initial_matrix);

  cpuMemoryFlag = true;
}

template<typename T>
msl::DMatrix<T>::DMatrix(int n0, int m0, T (*f)(int, int))
    : n(n0),
      m(m0),
      dist(Distribution::COPY) {
  init(1, 1);

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, localsize*sizeof(T)));
  initGPU();
#else
  localPartition = new T[localsize];
#endif

  fill(f);

  cpuMemoryFlag = true;
}

template<typename T>
template<typename F2>
msl::DMatrix<T>::DMatrix(int n0, int m0, const F2& f)
    : n(n0),
      m(m0),
      dist(Distribution::COPY) {
  init(1, 1);

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, localsize*sizeof(T)));
  initGPU();
#else
  localPartition = new T[localsize];
#endif

  fill(f);

  cpuMemoryFlag = true;
}

template<typename T>
msl::DMatrix<T>::DMatrix(const DMatrix<T>& cs)
    : n(cs.n),
      m(cs.m),
      dist(cs.dist) {
  init(cs.blocksInCol, cs.blocksInRow);

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, localsize*sizeof(T)));

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
    for (int i = 0; i < Muesli::num_gpus; i++) {
      cudaSetDevice(i);
      CUDA_CHECK_RETURN(cudaStreamSynchronize(Muesli::streams[i]));
    }
  } else {  // data is up to date in main memory
    std::copy(cs.localPartition, cs.localPartition + localsize, localPartition);
  }

#else
  localPartition = new T[localsize];
  std::copy(cs.localPartition, cs.localPartition + localsize, localPartition);
#endif

}

template<typename T>
msl::DMatrix<T>::~DMatrix() {
#ifdef __CUDACC__
  cudaFreeHost(localPartition);
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
msl::DMatrix<T>& msl::DMatrix<T>::operator=(const DMatrix<T>& rhs) {
  if (this != &rhs) {
    n = rhs.n;
    m = rhs.m;
    dist = rhs.dist;
    init(rhs.blocksInCol, rhs.blocksInRow);

    bool create_new_local_partition = false;

    if (nLocal != rhs.nLocal || mLocal != rhs.mLocal) {
      create_new_local_partition = true;
    }

    T* new_localPartition;

#ifdef __CUDACC__
    if(create_new_local_partition) {
      CUDA_CHECK_RETURN(cudaMallocHost(&new_localPartition, localsize*sizeof(T)));
    }

    gpuCopyDistributed = rhs.gpuCopyDistributed;
    GPUExecutionPlan<T>* new_plans = new GPUExecutionPlan<T>[Muesli::num_gpus];

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
        new_plans[i].h_Data = plans[i].h_Data;
        new_plans[i].d_Data = plans[i].d_Data;
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
        std::copy(rhs.localPartition, rhs.localPartition + rhs.localsize, new_localPartition);
      } else {
        std::copy(rhs.localPartition, rhs.localPartition + rhs.localsize, localPartition);
      }
    }

    // free old memory
    delete[] plans;

    // assign new memory
    plans = new_plans;

    // free old memory
    if(create_new_local_partition) {
      CUDA_CHECK_RETURN(cudaFreeHost(localPartition));
    }
#else
    if (create_new_local_partition) {
      new_localPartition = new T[localsize];
      std::copy(rhs.localPartition, rhs.localPartition + rhs.localsize,
                new_localPartition);
      // free old memory
      delete[] localPartition;
    } else {
      std::copy(rhs.localPartition, rhs.localPartition + rhs.localsize,
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
void msl::DMatrix<T>::init(int rows, int cols) {
  if (Muesli::proc_entrance == UNDEFINED) {
    throws(detail::MissingInitializationException());
  }

  blocksInRow = cols;
  blocksInCol = rows;

  id = Muesli::proc_id;
  np = Muesli::num_local_procs;

  if (dist != Distribution::COPY) {
    if (rows * cols != np || n % rows != 0 || m % cols != 0) {
      throws(detail::PartitioningImpossibleException());
    }
  }

  if (dist == Distribution::COPY) {
    nLocal = n;
    mLocal = m;
    localColPosition = 0;
    localRowPosition = 0;
  } else {
    // for simplicity: assuming rows divides n
    nLocal = n / rows;
    // for simplicity: assuming cols divides m
    mLocal = m / cols;
    // blocks assigned row by row
    localColPosition = Muesli::proc_id % cols;
    localRowPosition = Muesli::proc_id / cols;
  }
  localsize = nLocal * mLocal;
  localPosition = localRowPosition * cols + localColPosition;
  firstRow = localRowPosition * nLocal;
  firstCol = localColPosition * mLocal;
  nextRow = firstRow + nLocal;
  nextCol = firstCol + mLocal;
  firstIndex = firstRow * mLocal + firstCol;
}

template<typename T>
void msl::DMatrix<T>::initGPU() {
#ifdef __CUDACC__
  if (plans != 0) {
    delete[] plans;
  }
  plans = new GPUExecutionPlan<T> [Muesli::num_gpus];
  gpuCopyDistributed = 0;
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    plans[i].nLocal = nLocal / Muesli::num_gpus;
    plans[i].mLocal = mLocal;
  }
  for (int i = 0; i < nLocal % Muesli::num_gpus; i++) {
    plans[i].nLocal++;
  }
  int gpuBase = 0;
  int rowBase = 0;
  for (int i = 0; i < Muesli::num_gpus; i++) {
    plans[i].size = plans[i].nLocal * plans[i].mLocal;
    plans[i].bytes = plans[i].size * sizeof(T);
    plans[i].first = gpuBase + firstIndex;
    plans[i].firstRow = firstRow + rowBase;
    plans[i].firstCol = firstCol;
    plans[i].h_Data = localPartition + gpuBase;
    plans[i].d_Data = 0;
    gpuBase += plans[i].size;
    rowBase += plans[i].nLocal;
  }
#endif
}

template<typename T>
void msl::DMatrix<T>::fill(const T& value) {
#pragma omp parallel for
  for (int i = 0; i < localsize; i++) {
    localPartition[i] = value;
  }
}

template<typename T>
void msl::DMatrix<T>::fill(T* const values) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    for (int j = 0; j < mLocal; j++) {
      localPartition[i * mLocal + j] =
          values[(i + firstRow) * m + j + firstCol];
    }
  }
}

template<typename T>
void msl::DMatrix<T>::fill(T** const values) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    for (int j = 0; j < mLocal; j++) {
      localPartition[i * mLocal + j] = values[i + firstRow][j + firstCol];
    }
  }
}

template<typename T>
void msl::DMatrix<T>::fill(T (*f)(int, int)) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    for (int j = 0; j < mLocal; j++) {
      localPartition[i * mLocal + j] = f(i + firstRow, j + firstCol);
    }
  }
}

template<typename T>
template<typename F2>
void msl::DMatrix<T>::fill(const F2& f) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    for (int j = 0; j < mLocal; j++) {
      localPartition[i * mLocal + j] = f(i + firstRow, j + firstCol);
    }
  }
}

template<typename T>
void msl::DMatrix<T>::fill_root_init(T* const values) {
  scatter(values, localPartition, nLocal * mLocal);
}


template<typename T>
T* msl::DMatrix<T>::getLocalPartition() const {
  return localPartition;
}

template<typename T>
T msl::DMatrix<T>::get(size_t row, size_t col) const {
  int idSource;
  T message;

  // element with global index is locally stored
  if (isLocal(row, col)) {
#ifdef __CUDACC__
    // element might not be up to date in cpu memory
    if (!cpuMemoryFlag) {
      // find GPU that stores the desired element
      int device = getGpuId(row, col);
      cudaSetDevice(device);
      // download element
      int offset = (row-plans[device].firstRow) * plans[device].mLocal + col-plans[device].firstCol;
      CUDA_CHECK_RETURN(
          cudaMemcpyAsync(&message,
              plans[device].d_Data+offset,
              sizeof(T),
              cudaMemcpyDeviceToHost,
              Muesli::streams[device]));
    } else {  // element is up to date in cpu memory
      message = localPartition[(row-firstRow)*mLocal + col-firstCol];
    }
#else
    message = localPartition[(row - firstRow) * mLocal + col - firstCol];
#endif
    idSource = Muesli::proc_id;
  }
  // Element with global index is not locally stored
  else {
    // Calculate id of the process that locally stores the element
    int x = row / nLocal;
    int y = col / mLocal;
    idSource = blocksInRow * x + y;
  }

  msl::MSL_Broadcast(idSource, &message, 1);
  return message;
}

template<typename T>
int msl::DMatrix<T>::getFirstCol() const {
  return firstCol;
}

template<typename T>
int msl::DMatrix<T>::getFirstRow() const {
  return firstRow;
}

template<typename T>
int msl::DMatrix<T>::getLocalCols() const {
  return mLocal;
}

template<typename T>
int msl::DMatrix<T>::getLocalRows() const {
  return nLocal;
}

template<typename T>
int msl::DMatrix<T>::getRows() const {
  return n;
}

template<typename T>
int msl::DMatrix<T>::getCols() const {
  return m;
}

template<typename T>
int msl::DMatrix<T>::getBlocksInCol() const {
  return blocksInCol;
}

template<typename T>
int msl::DMatrix<T>::getBlocksInRow() const {
  return blocksInRow;
}

template<typename T>
bool msl::DMatrix<T>::isLocal(int i, int j) const {
  return (i >= firstRow) && (i < nextRow) && (j >= firstCol) && (j < nextCol);
}

template<typename T>
T msl::DMatrix<T>::getLocal(int i, int j) const {
//  if ((i >= nLocal) || (j >= mLocal)) {
//    throws(detail::NonLocalAccessException());
//  }
  return localPartition[i * mLocal + j];
}

template<typename T>
void msl::DMatrix<T>::setLocal(int i, int j, const T& v) {
//  if ((i >= nLocal) || (j >= mLocal)) {
//    throws(detail::NonLocalAccessException());
//  }

  localPartition[i * mLocal + j] = v;
}

template<typename T>
void msl::DMatrix<T>::set(int i, int j, const T& v) {
#ifdef __CUDACC__
  download();
#endif

  if ((i >= firstRow) && (i < nextRow) && (j >= firstCol) && (j < nextCol)) {
    localPartition[(i - firstRow) * mLocal + (j - firstCol)] = v;
  }
}

template<typename T>
int msl::DMatrix<T>::getLocalSize() const {
  return localsize;
}

// SKELETONS / COMMUNICATION / BROADCAST PARTITION

template<typename T>
void msl::DMatrix<T>::broadcastPartition(int blockRow, int blockCol) {
  if (blockRow < 0 || blockRow >= blocksInCol || blockCol < 0
      || blockCol >= blocksInRow) {
    throws(detail::IllegalPartitionException());
  }

  bool upload_after = 0;
  if (!cpuMemoryFlag) {
    download();
    upload_after = 1;
  }

  int rootId = blockRow * blocksInRow + blockCol;
  msl::MSL_Broadcast(rootId, localPartition, localsize);

  if (upload_after)
    upload();
}

// SKELETONS / COMMUNICATION / GATHER

template<typename T>
void msl::DMatrix<T>::gather(T** b) {
  download();
  if (dist == Distribution::COPY) {
    // if matrix is copy distributed, all processes store same data
    for (int i = 0; i < n; i++) {
      std::copy(localPartition + (i * m), localPartition + ((i + 1) * m), b[i]);
    }
  } else {
    // gather matrix
    T* buffer = new T[n * m];
    msl::allgather(localPartition, buffer, localsize);

    // the (two dimensional) local partitions of this matrix are gathered (one by one)
    // in a one dimensional array. that means, when writing this array into the two
    // dimensional result array b, we have to calculate the correct indices.
    // Example: 4x4 matrix with 2x2 local partitions (4 processes)
    // [1,  2,  3,  4]     gather
    // [4,  5,  6,  7]   ---------> [1, 2, 4, 5, 3, 4, 6, 7, 7, 8, 10, 11, 9, 10, 12, 13]
    // [7,  8,  9,  10]  write back
    // [10, 11, 12, 13]  <---------
    // this index calculation below can probably be simplified.
    if (blocksInRow == 1) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
          b[i][j] = buffer[(i % nLocal) * mLocal + (i / nLocal) * localsize
              + (j / mLocal) * localsize + j % mLocal];
        }
      }
    } else {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
          b[i][j] = buffer[(i % nLocal) * mLocal
              + (i / nLocal) * localsize * blocksInCol
              + (j / mLocal) * localsize + j % mLocal];
        }
      }
    }
    delete[] buffer;
  }
}

template<typename T>
void msl::DMatrix<T>::gather(DMatrix<T>& dm) {
  if (dm.dist == Distribution::COPY) {
    // download current data if necessary
    download();
    if (dist == Distribution::COPY) {
      // if data is uploaded, free gpu memory.
      // necessary because data will be updated in cpu memory and needs to be uploaded again.
      if (!dm.cpuMemoryFlag) {
        dm.freeDevice();
      }
      // if matrix is copy distributed, all processes store same data
      std::copy(localPartition, localPartition + n, dm.getLocalPartition());
    } else {
      // if data is uploaded, free gpu memory.
      // necessary because data will be updated in cpu memory and needs to be uploaded again.
      if (!dm.cpuMemoryFlag) {
        dm.freeDevice();
      }

      T* buffer = new T[n * m];
      msl::allgather(localPartition, buffer, localsize);

      // the (two dimensional) local partitions of this matrix are gathered (one ny one)
      // in a one dimensional array. that means, when writing this array into the two
      // dimensional result matrix, we have to calculate the correct indices.
      // [1,  2,  3,  4]     gather
      // [4,  5,  6,  7]   ---------> [1, 2, 4, 5, 3, 4, 6, 7, 7, 8, 10, 11, 9, 10, 12, 13]
      // [7,  8,  9,  10]  write back
      // [10, 11, 12, 13]  <---------
      // this index calculation below can probably be simplified.
      if (blocksInRow == 1) {
        for (int i = 0; i < dm.getLocalRows(); i++) {
          for (int j = 0; j < dm.getLocalCols(); j++) {
            dm.setLocal(
                i,
                j,
                buffer[(i % nLocal) * mLocal + (i / nLocal) * localsize
                    + (j / mLocal) * localsize + j % mLocal]);
          }
        }
      } else {
        for (int i = 0; i < dm.getLocalRows(); i++) {
          for (int j = 0; j < dm.getLocalCols(); j++) {
            dm.setLocal(
                i,
                j,
                buffer[(i % nLocal) * mLocal
                    + (i / nLocal) * localsize * blocksInCol
                    + (j / mLocal) * localsize + j % mLocal]);
          }
        }
      }
    }
  }
}

// SKELETONS / COMMUNICATION / PERMUTE PARTITION

template<typename T>
template<typename F1, typename F2>
void msl::DMatrix<T>::permutePartition(const Fct2<int, int, int, F1>& newRow,
                                       const Fct2<int, int, int, F2>& newCol) {
  if (dist == Distribution::COPY) {
    // if matrix is copy distributed, permuting the partitions makes no sense since
    // all processes store same data
    return;
  }

  download();

  int i, j, receiver;
  receiver = newRow(localRowPosition, localColPosition) * blocksInRow
      + newCol(localRowPosition, localColPosition);

  if (receiver < 0 || receiver >= Muesli::num_local_procs) {
    throws(detail::IllegalPartitionException());
  }

  int sender = UNDEFINED;

  // determine sender
  for (i = 0; i < blocksInCol; i++) {
    for (j = 0; j < blocksInRow; j++) {
      if (newRow(i, j) * blocksInRow + newCol(i, j) == Muesli::proc_id) {
        if (sender == UNDEFINED) {
          sender = i * blocksInRow + j;
        }  // newRow and newCol don't produce a permutation
        else {
          throws(detail::IllegalPermuteException());
        }
      }
    }
  }

  // newRow and newCol don't produce a permutation
  if (sender == UNDEFINED) {
    throws(detail::IllegalPermuteException());
  }

  if (receiver != Muesli::proc_id) {
    T* buffer = new T[localsize];

    // purely asynchronous communication crashes when compile with nvcc.
#ifdef __CUDACC__
    for (i = 0; i < localsize; i++) {
      buffer[i] = localPartition[i];
    }

    MPI_Status stat;
    MPI_Request req;

    MSL_ISend(receiver, buffer, req, localsize, msl::MYTAG);
    MSL_Recv(sender, localPartition, stat, localsize, msl::MYTAG);
    MPI_Wait(&req, &stat);

    delete[] buffer;
#else
    MPI_Status stat[2];
    MPI_Request req[2];

    MSL_IRecv(sender, buffer, req[0], localsize, msl::MYTAG);
    MSL_ISend(receiver, localPartition, req[1], localsize, msl::MYTAG);

    MPI_Waitall(2, req, stat);

    T* swap = localPartition;
    localPartition = buffer;
    delete[] swap;
#endif
  }
}

template<typename T>
template<typename F1, typename F2>
void msl::DMatrix<T>::permutePartition(F1& newRow, F2& newCol) {
  if (dist == Distribution::COPY) {
    // if matrix is copy distributed, permuting the partitions makes no sense since
    // all processes store same data
    return;
  }

  download();

  int i, j, receiver;
  receiver = newRow(localRowPosition) * blocksInRow + newCol(localColPosition);

  if (receiver < 0 || receiver >= Muesli::num_local_procs) {
    throws(detail::IllegalPartitionException());
  }

  int sender = UNDEFINED;

  // determine sender
  for (i = 0; i < blocksInCol; i++) {
    for (j = 0; j < blocksInRow; j++) {
      if (newRow(i) * blocksInRow + newCol(j) == Muesli::proc_id) {
        if (sender == UNDEFINED) {
          sender = i * blocksInRow + j;
        }  // newRow and newCol don't produce a permutation
        else {
          throws(detail::IllegalPermuteException());
        }
      }
    }
  }

  // newRow and newCol don't produce a permutation
  if (sender == UNDEFINED) {
    throws(detail::IllegalPermuteException());
  }

  if (receiver != Muesli::proc_id) {
    T* buffer = new T[localsize];

    // purely asynchronous communication crashes when compile with nvcc.
#ifdef __CUDACC__
    for (i = 0; i < localsize; i++) {
      buffer[i] = localPartition[i];
    }

    MPI_Status stat;
    MPI_Request req;

    MSL_ISend(receiver, buffer, req, localsize, msl::MYTAG);
    MSL_Recv(sender, localPartition, stat, localsize, msl::MYTAG);
    MPI_Wait(&req, &stat);

    delete[] buffer;
#else
    MPI_Status stat[2];
    MPI_Request req[2];

    MSL_IRecv(sender, buffer, req[0], localsize, msl::MYTAG);
    MSL_ISend(receiver, localPartition, req[1], localsize, msl::MYTAG);

    MPI_Waitall(2, req, stat);

    T* swap = localPartition;
    localPartition = buffer;
    delete[] swap;
#endif
  }
}

template<typename T>
void msl::DMatrix<T>::permutePartition(int (*f)(int, int), int (*g)(int, int)) {
  permutePartition(curry(f), curry(g));
}

template<typename T>
template<typename F>
void msl::DMatrix<T>::permutePartition(int (*f)(int, int),
                                       const Fct2<int, int, int, F>& g) {
  permutePartition(curry(f), g);
}

template<typename T>
template<typename F>
void msl::DMatrix<T>::permutePartition(const Fct2<int, int, int, F>& f,
                                       int (*g)(int, int)) {
  permutePartition(f, curry(g));
}

// SKELETONS / COMMUNICATION / ROTATE

// SKELETONS / COMMUNICATION / ROTATE / ROTATE COLUMNS

template<typename T>
template<typename F>
void msl::DMatrix<T>::shiftCols(F& f) {
  auto identity = [] (int a) {return a;};
  permutePartition(identity, f);
}

template<typename T>
template<typename F>
void msl::DMatrix<T>::shiftRows(F& f) {
  auto identity = [] (int a) {return a;};
  permutePartition(f, identity);
}

template<typename T>
template<typename F>
void msl::DMatrix<T>::rotateCols(const Fct1<int, int, F>& f) {
  permutePartition(
      curry((int (*)(const Fct1<int, int, F>&, int, int, int))auxRotateCols<F>)(f)(blocksInCol),
  curry((int (*)(int, int))proj2_2<int, int>));
}

template<typename T>
void msl::DMatrix<T>::rotateCols(int (*f)(int)) {
  rotateCols(curry(f));
}

template<typename T>
void msl::DMatrix<T>::rotateCols(int rows) {
  rotateCols(curry((int (*)(int, int))proj1_2<int, int>)(rows));
}

// SKELETONS / COMMUNICATION / ROTATE / ROTATE ROWS

template<typename T>
template<typename F>
void msl::DMatrix<T>::rotateRows(const Fct1<int, int, F>& f) {
  permutePartition(curry((int (*)(int, int)) proj1_2<int, int>), curry((int (*)(const Fct1<int, int, F>&,
              int,
              int,
              int)) auxRotateRows<F>)(f)(blocksInRow));
}

template<typename T>
void msl::DMatrix<T>::rotateRows(int (*f)(int)) {
  rotateRows(curry(f));
}

template<typename T>
void msl::DMatrix<T>::rotateRows(int cols) {
  rotateRows(curry((int (*)(int, int))proj1_2<int, int>)(cols));
}

template<typename T>
inline void msl::DMatrix<T>::transposeLocalPartition() {
  // Square matrix (nLocal == mLocal)
  if (nLocal == mLocal) {
    T* tmp;
#ifdef __CUDACC__
    CUDA_CHECK_RETURN(cudaMallocHost(&tmp, localsize*sizeof(T)));
#else
    tmp = new T[localsize];
#endif

#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
      for (int j = 0; j < mLocal; j++) {
        tmp[i * mLocal + j] = localPartition[j * mLocal + i];
      }
    }

#ifdef __CUDACC__
    cudaFreeHost(localPartition);
#else
    delete[] localPartition;
#endif

    localPartition = tmp;
  } else {
    // not yet implemented.
    // need to adjust GPU execution plans after transposing
  }
}

template<typename T>
void msl::DMatrix<T>::setGpuDistribution(Distribution dist) {
#ifdef __CUDACC__

  // switch distribution mode
  if (dist == Distribution::COPY && !gpuCopyDistributed) {
    // if data is uploaded, download data and free device memory first.
    // this is necessary because the size of the gpu partitions change
    // when switching the distribution mode.
    if (!cpuMemoryFlag) {
      download();
    }

    for (int i = 0; i < Muesli::num_gpus; i++) {
      plans[i].nLocal = nLocal;
      plans[i].mLocal = mLocal;
      plans[i].size = nLocal * mLocal;  // localsize
      plans[i].bytes = plans[i].size * sizeof(T);
      plans[i].first = firstIndex;
      plans[i].firstRow = firstRow;
      plans[i].firstCol = firstCol;
      plans[i].h_Data = localPartition;
      plans[i].d_Data = 0;
    }
  } else if (gpuCopyDistributed && !(dist == Distribution::COPY)) {
    // if data is uploaded, download data and free device memory first.
    // this is necessary because the size of the gpu partitions change
    // when switching the distribution mode.
    if (!cpuMemoryFlag) {
      download();
    }

    for (int i = 0; i < Muesli::num_gpus; i++) {
      plans[i].nLocal = nLocal / Muesli::num_gpus;
      plans[i].mLocal = mLocal;
    }
    for (int i = 0; i < nLocal % Muesli::num_gpus; i++) {
      plans[i].nLocal++;
    }
    int gpuBase = 0;
    int rowBase = 0;
    for (int i = 0; i < Muesli::num_gpus; i++) {
      plans[i].size = plans[i].nLocal * plans[i].mLocal;
      plans[i].bytes = plans[i].size * sizeof(T);
      plans[i].first = gpuBase + firstIndex;
      plans[i].firstRow = firstRow + rowBase;
      plans[i].firstCol = firstCol;
      plans[i].h_Data = localPartition + gpuBase;
      plans[i].d_Data = 0;
      gpuBase += plans[i].size;
      rowBase += plans[i].nLocal;
    }
  }
  gpuCopyDistributed = dist == Distribution::COPY ? true : false;
#endif
}

template<typename T>
msl::Distribution msl::DMatrix<T>::getGpuDistribution() {
#ifdef __CUDACC__
  return gpuCopyDistributed ? Distribution::COPY : Distribution::DIST;
#else
  return Distribution::DIST;
#endif
}

template<typename T>
std::vector<T*> msl::DMatrix<T>::upload(bool allocOnly) {
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
    // wait until upload is finished
    for (int i = 0; i < Muesli::num_gpus; i++) {
      CUDA_CHECK_RETURN(cudaStreamSynchronize(Muesli::streams[i]));
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
void msl::DMatrix<T>::download() {
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
void msl::DMatrix<T>::freeDevice() {
#ifdef __CUDACC__
  if (!cpuMemoryFlag) {
    for (int i = 0; i < Muesli::num_gpus; i++) {
      cudaFree(plans[i].d_Data);
      plans[i].d_Data = 0;
    }
    cpuMemoryFlag = true;
  }
#endif
}

template<typename T>
int msl::DMatrix<T>::getGpuId(int row, int col) const {
  int id = 0;
  int local_index = row * mLocal + col;
  for (int i = 1; i < Muesli::num_gpus; i++) {
    if (local_index >= plans[i].first) {
      id = i;
    }
  }
  return id;
}

template<typename T>
std::vector<GPUExecutionPlan<T> > msl::DMatrix<T>::getExecPlans() {
  upload();
  std::vector<GPUExecutionPlan<T> > ret(plans, plans + Muesli::num_gpus);
  return ret;
}

template<typename T>
GPUExecutionPlan<T> msl::DMatrix<T>::getExecPlan(int device) {
  upload();
  return plans[device];
}

template<typename T>
void msl::DMatrix<T>::setCopyDistribution() {
  if (np < 2) {
    dist = Distribution::COPY;
    return;
  }

  if (dist == Distribution::DIST) {
    download();

    T** tmp = new T*[n];
    for (int i = 0; i < n; i++) {
      tmp[i] = new T[m];
    }
    gather(tmp);

    dist = Distribution::COPY;
    init(1, 1);

#ifdef __CUDACC__
    CUDA_CHECK_RETURN(cudaFreeHost(localPartition));
    CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, localsize*sizeof(T)));
    initGPU();
#else
    delete[] localPartition;
    localPartition = new T[localsize];
#endif

    fill(tmp);
    for (int i = 0; i < n; i++) {
      delete[] tmp[i];
    }
    delete[] tmp;
  }
}

template<typename T>
void msl::DMatrix<T>::setDistribution(int rows, int cols) {
  if (np < 2) {
    dist = Distribution::DIST;
    return;
  }

  download();

  T** tmp = new T*[n];
  for (int i = 0; i < n; i++) {
    tmp[i] = new T[m];
  }
  gather(tmp);

  dist = Distribution::DIST;
  init(rows, cols);

#ifdef __CUDACC__
  CUDA_CHECK_RETURN(cudaFreeHost(localPartition));
  CUDA_CHECK_RETURN(cudaMallocHost(&localPartition, localsize*sizeof(T)));
  initGPU();
#else
  delete[] localPartition;
  localPartition = new T[localsize];
#endif

  fill(tmp);

  for (int i = 0; i < n; i++) {
    delete[] tmp[i];
  }
  delete[] tmp;
}

template<typename T>
void msl::DMatrix<T>::show(const std::string& descr) {
  std::ostringstream s;
  if (descr.size() > 0)
    s << descr << ": " << std::endl;

  T** b = new T*[n];
  for (int i = 0; i < n; i++) {
    b[i] = new T[m];
  }

  gather(b);

  if (msl::isRootProcess()) {
    for (int i = 0; i < n; i++) {
      s << "[";

      for (int j = 0; j < m; j++) {
        s << b[i][j];

        if (j < m - 1) {
          s << " ";
        }
      }

      s << "]" << std::endl;
    }
    s << std::endl;
  }

  for (int i = 0; i < n; i++) {
    delete[] b[i];
  }

  delete[] b;

  if (msl::isRootProcess()) {
    printf("%s", s.str().c_str());
  }
}

template<typename T>
void msl::DMatrix<T>::printLocal() {
  download();

  for (int i = 0; i < Muesli::num_local_procs; i++) {
    if (Muesli::proc_id == i) {
      //print
      std::cout << Muesli::proc_id << ": [" << localPartition[0];
      for (int j = 1; j < localsize; j++) {
        std::cout << " " << localPartition[j];
      }
      std::cout << "]" << std::endl;
    }
    MPI_Barrier (MPI_COMM_WORLD);
  }
}

template<class T>
std::vector<double> msl::DMatrix<T>::getStencilTimes() {
  double* sbuf = new double[3];
  sbuf[0] = upload_time;
  sbuf[1] = padding_time;
  sbuf[2] = kernel_time;
  double* rbuf = new double[msl::Muesli::num_total_procs * 3];

  msl::allgather(sbuf, rbuf, 3);

  double avg_upload = 0.0, avg_padding = 0.0, avg_kernel = 0.0;
  for (int i = 0; i < msl::Muesli::num_total_procs * 3; i += 3) {
    avg_upload += rbuf[i];
    avg_padding += rbuf[i + 1];
    avg_kernel += rbuf[i + 2];
  }

  std::vector<double> ret;
  ret.push_back(avg_upload / Muesli::num_total_procs);
  ret.push_back(avg_padding / Muesli::num_total_procs);
  ret.push_back(avg_kernel / Muesli::num_total_procs);
  delete[] sbuf;
  delete[] rbuf;
  return ret;
}

