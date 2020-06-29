/*
 * dmatrix.cu
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
template<typename MapFunctor>
void msl::DMatrix<T>::mapInPlace(MapFunctor& f) {
  mapInPlace(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapFunctor)>());
}

template<typename T>
template<typename MapFunctor>
void msl::DMatrix<T>::mapInPlace(MapFunctor& f, Int2Type<true>) {
  // upload data first (if necessary)
  upload();

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].mLocal, plans[i].firstRow,
           plans[i].firstCol);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
    detail::mapKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
        plans[i].d_Data, plans[i].d_Data, plans[i].size, f);
  }

  // check for errors during gpu computation
  msl::syncStreams();
}

template<typename T>
template<typename MapFunctor>
void msl::DMatrix<T>::mapInPlace(MapFunctor& f, Int2Type<false>) {
  // upload data first (if necessary)
  upload();

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
    detail::mapKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, plans[i].d_Data, plans[i].size, f);
  }

  // check for errors during gpu computation
  msl::syncStreams();
}

template<typename T>
template<typename MapIndexFunctor>
void msl::DMatrix<T>::mapIndexInPlace(MapIndexFunctor& f) {
  mapIndexInPlace(
      f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapIndexFunctor)>());
}

template<typename T>
template<typename MapIndexFunctor>
void msl::DMatrix<T>::mapIndexInPlace(MapIndexFunctor& f, Int2Type<true>) {
  // upload data first (if necessary)
  upload();

  // set kernel configuration
  int tile_width = f.getTileWidth();
  int tpbX = Muesli::tpb_x;
  int tpbY = Muesli::tpb_y;
  if (tile_width != -1) {
    tpbX = tpbY = tile_width;
  }

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].mLocal, plans[i].firstRow,
           plans[i].firstCol);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(tpbX, tpbY);
    dim3 dimGrid((plans[i].mLocal + dimBlock.x - 1) / dimBlock.x,
                 (plans[i].nLocal + dimBlock.y - 1) / dimBlock.y);
    detail::mapIndexKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
        plans[i].d_Data, plans[i].d_Data, plans[i], f, f.useLocalIndices());
  }

  // check for errors during gpu computation
  msl::syncStreams();
}

template<typename T>
template<typename MapIndexFunctor>
void msl::DMatrix<T>::mapIndexInPlace(MapIndexFunctor& f, Int2Type<false>) {
  // upload data first (if necessary)
  upload();

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::tpb_x, Muesli::tpb_y);
    dim3 dimGrid((plans[i].mLocal + dimBlock.x - 1) / dimBlock.x,
                 (plans[i].nLocal + dimBlock.y - 1) / dimBlock.y);
    detail::mapIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, plans[i].d_Data, plans[i], f, false);
  }

  // check for errors during gpu computation
  msl::syncStreams();
}

template<typename T>
template<typename R, typename MapFunctor>
msl::DMatrix<R> msl::DMatrix<T>::map(MapFunctor& f) {
  return map<R>(f,
                Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapFunctor)>());
}

template<typename T>
template<typename R, typename MapFunctor>
msl::DMatrix<R> msl::DMatrix<T>::map(MapFunctor& f, Int2Type<true>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);

  // upload data first (if necessary)
  upload();
  result.upload(1);  // alloc only

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].mLocal, plans[i].firstRow,
           plans[i].firstCol);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
    detail::mapKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
        plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].size, f);
  }

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template<typename T>
template<typename R, typename MapFunctor>
msl::DMatrix<R> msl::DMatrix<T>::map(MapFunctor& f, Int2Type<false>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);

  // upload data first (if necessary)
  upload();
  result.upload(1);  // alloc only

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
    detail::mapKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].size, f);
  }

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template<typename T>
template<typename R, typename MapIndexFunctor>
msl::DMatrix<R> msl::DMatrix<T>::mapIndex(MapIndexFunctor& f) {
  return mapIndex<R>(
      f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapIndexFunctor)>());
}

template<typename T>
template<typename R, typename MapIndexFunctor>
msl::DMatrix<R> msl::DMatrix<T>::mapIndex(MapIndexFunctor& f, Int2Type<true>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);

  // upload data first (if necessary)
  upload();
  result.upload(1);  // alloc only

  // set kernel configuration
  int tile_width = f.getTileWidth();
  int tpbX = Muesli::tpb_x;
  int tpbY = Muesli::tpb_y;
  if (tile_width != -1) {
    tpbX = tpbY = tile_width;
  }

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].mLocal, plans[i].firstRow,
           plans[i].firstCol);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(tpbX, tpbY);
    dim3 dimGrid((plans[i].mLocal + dimBlock.x - 1) / dimBlock.x,
                 (plans[i].nLocal + dimBlock.y - 1) / dimBlock.y);
    detail::mapIndexKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
        plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i], f,
        f.useLocalIndices());
  }

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template<typename T>
template<typename R, typename MapIndexFunctor>
msl::DMatrix<R> msl::DMatrix<T>::mapIndex(MapIndexFunctor& f, Int2Type<false>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);

  // upload data first (if necessary)
  upload();
  result.upload(1);  // alloc only

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::tpb_x, Muesli::tpb_y);
    dim3 dimGrid((plans[i].mLocal + dimBlock.x - 1) / dimBlock.x,
                 (plans[i].nLocal + dimBlock.y - 1) / dimBlock.y);
    detail::mapIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i], f, false);
  }

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template<typename T>
template<typename MapStencilFunctor>
void msl::DMatrix<T>::mapStencilInPlace(MapStencilFunctor& f, T neutral_value) {
  double t = MPI_Wtime();
  // Check for row distribution.
  if (blocksInRow > 1) {
    std::cout
        << "Matrix must not be block distributed with more than 1 horizontal block for mapStencil!\n";
    return;
  }

  // Obtain stencil size.
  int stencil_size = f.getStencilSize();
  int size = (nLocal + 2 * stencil_size) * mLocal;
  // Prepare padded local partition. We need additional 2*stencil_size rows.
  T* padded_local_matrix;
  cudaMallocHost(&padded_local_matrix, size * sizeof(T));
  padding_time += (MPI_Wtime() - t);
  t = MPI_Wtime();
  // Update data in main memory if necessary.
  download();
  upload_time += (MPI_Wtime() - t);
  t = MPI_Wtime();

  // Gather border regions.
  MPI_Status stat;
  MPI_Request req;
  int padding_size = stencil_size * mLocal;
  // Top down (send last stencil_size rows to successor):
  // Non-blocking send.
  if (Muesli::proc_id < Muesli::num_local_procs - 1) {
    T* buffer = localPartition + (nLocal - stencil_size) * mLocal;
    MSL_ISend(Muesli::proc_id + 1, buffer, req, padding_size, msl::MYTAG);
  }

  // Copy localPartition to padded_local_matrix
  std::copy(localPartition, localPartition + localsize,
            padded_local_matrix + (stencil_size * mLocal));

  // Blocking receive.
  if (Muesli::proc_id > 0) {
    MSL_Recv(Muesli::proc_id - 1, padded_local_matrix, stat, padding_size,
             msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id < Muesli::num_local_procs - 1) {
    MPI_Wait(&req, &stat);
  }

  // Bottom up (send first stencil_rows to predecessor):
  // Non-blocking send.
  if (Muesli::proc_id > 0) {
    MSL_ISend(Muesli::proc_id - 1, localPartition, req, padding_size,
              msl::MYTAG);
  }

  // Blocking receive.
  if (Muesli::proc_id < Muesli::num_local_procs - 1) {
    T* buffer = padded_local_matrix + (nLocal + stencil_size) * mLocal;
    MSL_Recv(Muesli::proc_id + 1, buffer, stat, padding_size, msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id > 0) {
    MPI_Wait(&req, &stat);
  }

  // Process 0 and process n-1 need to fill upper (lower) border regions with neutral value.
  if (Muesli::proc_id == 0) {
    for (int i = 0; i < stencil_size * mLocal; i++) {
      padded_local_matrix[i] = neutral_value;
    }
  }
  if (Muesli::proc_id == Muesli::num_local_procs - 1) {
    for (int i = (nLocal + stencil_size) * mLocal;
        i < (nLocal + 2 * stencil_size) * mLocal; i++) {
      padded_local_matrix[i] = neutral_value;
    }
  }

  int tile_width = f.getTileWidth();
  padding_time += (MPI_Wtime() - t);
  t = MPI_Wtime();

  // Upload buffers.
  std::vector<T*> d_padded_local_matrix(Muesli::num_gpus);
  // Create padded local matrix.
  msl::PLMatrix<T> plm(n, m, nLocal, mLocal, stencil_size, tile_width,
                       neutral_value);
  for (int i = 0; i < msl::Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN(
        cudaMalloc((void**) &d_padded_local_matrix[i], sizeof(T) * size));
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(d_padded_local_matrix[i], padded_local_matrix,
                        sizeof(T) * size, cudaMemcpyHostToDevice,
                        Muesli::streams[i]));
    plm.addDevicePtr(d_padded_local_matrix[i]);
  }

  // Upload data (this).
  upload(1);

  // Upload padded local partitions.
  std::vector<PLMatrix<T>*> d_plm(Muesli::num_gpus);
  for (int i = 0; i < msl::Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    plm.setFirstRowGPU(plans[i].firstRow);
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_plm[i], sizeof(PLMatrix<T> )));
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(d_plm[i], &plm, sizeof(PLMatrix<T> ),
                        cudaMemcpyHostToDevice, Muesli::streams[i]));
    plm.update();
  }

  upload_time += (MPI_Wtime() - t);
  t = MPI_Wtime();

  // Map stencil
  int smem_size = (tile_width + 2 * stencil_size)
      * (tile_width + 2 * stencil_size) * sizeof(T);
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].mLocal, plans[i].firstRow,
           plans[i].firstCol);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(tile_width, tile_width);
    dim3 dimGrid((plans[i].mLocal + dimBlock.x - 1) / dimBlock.x,
                 (plans[i].nLocal + dimBlock.y - 1) / dimBlock.y);
    detail::mapStencilKernel<<<dimGrid, dimBlock, smem_size, Muesli::streams[i]>>>(
        plans[i].d_Data, plans[i].d_Data, plans[i], d_plm[i], f, tile_width);
  }

  // Check for errors during gpu computation.
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[i]));CUDA_CHECK_RETURN
    (cudaFree(d_plm[i]));
    CUDA_CHECK_RETURN(cudaFree(d_padded_local_matrix[i]));
  }

  kernel_time += (MPI_Wtime() - t);

  // Clean up.
  cudaFreeHost(padded_local_matrix);
}

template<typename T>
template<typename R, typename MapStencilFunctor>
msl::DMatrix<R> msl::DMatrix<T>::mapStencil(MapStencilFunctor& f,
                                            T neutral_value) {
  double t = MPI_Wtime();
  // Check for row distribution.
  if (blocksInRow > 1) {
    std::cout
        << "Matrix must not be block distributed with more than 1 horizontal block for mapStencil!\n";
    fail_exit();
  }

  // Obtain stencil size.
  int stencil_size = f.getStencilSize();
  int size = (nLocal + 2 * stencil_size) * mLocal;
  // Prepare padded local partition. We need additional 2*stencil_size rows.
  T* padded_local_matrix;
  cudaMallocHost(&padded_local_matrix, size * sizeof(T));
  padding_time += (MPI_Wtime() - t);
  t = MPI_Wtime();
  // Update data in main memory if necessary.
  download();
  upload_time += (MPI_Wtime() - t);

  t = MPI_Wtime();
  // Gather border regions.
  MPI_Status stat;
  MPI_Request req;
  int padding_size = stencil_size * mLocal;
  // Top down (send last stencil_size rows to successor):
  // Non-blocking send.
  if (Muesli::proc_id < Muesli::num_local_procs - 1) {
    T* buffer = localPartition + (nLocal - stencil_size) * mLocal;
    MSL_ISend(Muesli::proc_id + 1, buffer, req, padding_size, msl::MYTAG);
  }

  // Copy localPartition to padded_local_matrix
  std::copy(localPartition, localPartition + localsize,
            padded_local_matrix + (stencil_size * mLocal));

  // Blocking receive.
  if (Muesli::proc_id > 0) {
    MSL_Recv(Muesli::proc_id - 1, padded_local_matrix, stat, padding_size,
             msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id < Muesli::num_local_procs - 1) {
    MPI_Wait(&req, &stat);
  }

  // Bottom up (send first stencil_rows to predecessor):
  // Non-blocking send.
  if (Muesli::proc_id > 0) {
    MSL_ISend(Muesli::proc_id - 1, localPartition, req, padding_size,
              msl::MYTAG);
  }

  // Blocking receive.
  if (Muesli::proc_id < Muesli::num_local_procs - 1) {
    T* buffer = padded_local_matrix + (nLocal + stencil_size) * mLocal;
    MSL_Recv(Muesli::proc_id + 1, buffer, stat, padding_size, msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id > 0) {
    MPI_Wait(&req, &stat);
  }

  // Process 0 and process n-1 need to fill upper (lower) border regions with neutral value.
  if (Muesli::proc_id == 0) {
    for (int i = 0; i < stencil_size * mLocal; i++) {
      padded_local_matrix[i] = neutral_value;
    }
  }
  if (Muesli::proc_id == Muesli::num_local_procs - 1) {
    for (int i = (nLocal + stencil_size) * mLocal;
        i < (nLocal + 2 * stencil_size) * mLocal; i++) {
      padded_local_matrix[i] = neutral_value;
    }
  }

  int tile_width = f.getTileWidth();
  padding_time += (MPI_Wtime() - t);
  t = MPI_Wtime();

  // Upload buffers.
  std::vector<T*> d_padded_local_matrix(Muesli::num_gpus);
  // Create padded local matrix.
  msl::PLMatrix<T> plm(n, m, nLocal, mLocal, stencil_size, tile_width,
                       neutral_value);
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN(
        cudaMalloc((void**) &d_padded_local_matrix[i], sizeof(T) * size));
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(d_padded_local_matrix[i], padded_local_matrix,
                        sizeof(T) * size, cudaMemcpyHostToDevice,
                        Muesli::streams[i]));
    plm.addDevicePtr(d_padded_local_matrix[i]);
  }

  // Upload data (this).
  upload(1);

  // Upload padded local partitions.
  std::vector<PLMatrix<T>*> d_plm(Muesli::num_gpus);
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    plm.setFirstRowGPU(plans[i].firstRow);
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_plm[i], sizeof(PLMatrix<T> )));
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(d_plm[i], &plm, sizeof(PLMatrix<T> ),
                        cudaMemcpyHostToDevice, Muesli::streams[i]));
    plm.update();
  }

  // Map stencil
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);
  result.upload(1);  // alloc only
  upload_time += (MPI_Wtime() - t);
  t = MPI_Wtime();

  int smem_size = (tile_width + 2 * stencil_size)
      * (tile_width + 2 * stencil_size) * sizeof(T);
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].mLocal, plans[i].firstRow,
           plans[i].firstCol);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(tile_width, tile_width);
    dim3 dimGrid((plans[i].mLocal + dimBlock.x - 1) / dimBlock.x,
                 (plans[i].nLocal + dimBlock.y - 1) / dimBlock.y);
    detail::mapStencilKernel<<<dimGrid, dimBlock, smem_size, Muesli::streams[i]>>>(
        plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i], d_plm[i], f,
        tile_width);
  }

  // Check for errors during gpu computation.
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[i]));CUDA_CHECK_RETURN
    (cudaFree(d_plm[i]));
    CUDA_CHECK_RETURN(cudaFree(d_padded_local_matrix[i]));
  }

  kernel_time += (MPI_Wtime() - t);
  // Clean up.
  cudaFreeHost(padded_local_matrix);

  return result;
}

template<typename T>
template<typename T2, typename ZipFunctor>
void msl::DMatrix<T>::zipInPlace(DMatrix<T2>& b, ZipFunctor& f) {
  zipInPlace(b, f,
             Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipFunctor)>());
}

template<typename T>
template<typename T2, typename ZipFunctor>
void msl::DMatrix<T>::zipInPlace(DMatrix<T2>& b, ZipFunctor& f,
                                 Int2Type<true>) {
  // upload data first (if necessary)
  upload();
  b.upload();

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].mLocal, plans[i].firstRow,
           plans[i].firstCol);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
    detail::zipKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, plans[i].d_Data,
        plans[i].size, f);
  }

  // check for errors during gpu computation
  msl::syncStreams();
}

template<typename T>
template<typename T2, typename ZipFunctor>
void msl::DMatrix<T>::zipInPlace(DMatrix<T2>& b, ZipFunctor& f,
                                 Int2Type<false>) {
  // upload data first (if necessary)
  upload();
  b.upload();

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
    detail::zipKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, plans[i].d_Data,
        plans[i].size, f);
  }

  // check for errors during gpu computation
  msl::syncStreams();
}

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DMatrix<T>::zipIndexInPlace(DMatrix<T2>& b, ZipIndexFunctor& f) {
  zipIndexInPlace(
      b, f,
      Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipIndexFunctor)>());
}

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DMatrix<T>::zipIndexInPlace(DMatrix<T2>& b, ZipIndexFunctor& f,
                                      Int2Type<true>) {
  // upload data first (if necessary)
  upload();
  b.upload();

  // set kernel configuration
  int tile_width = f.getTileWidth();
  int tpbX = Muesli::tpb_x;
  int tpbY = Muesli::tpb_y;
  if (tile_width != -1) {
    tpbX = tpbY = tile_width;
  }

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].mLocal, plans[i].firstRow,
           plans[i].firstCol);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(tpbX, tpbY);
    dim3 dimGrid((plans[i].mLocal + dimBlock.x - 1) / dimBlock.x,
                 (plans[i].nLocal + dimBlock.y - 1) / dimBlock.y);
    detail::zipIndexKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, plans[i].d_Data, plans[i],
        f, f.useLocalIndices());
  }

  // check for errors during gpu computation
  msl::syncStreams();
}

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DMatrix<T>::zipIndexInPlace(DMatrix<T2>& b, ZipIndexFunctor& f,
                                      Int2Type<false>) {
  // upload data first (if necessary)
  upload();
  b.upload();

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::tpb_x, Muesli::tpb_y);
    dim3 dimGrid((plans[i].mLocal + dimBlock.x - 1) / dimBlock.x,
                 (plans[i].nLocal + dimBlock.y - 1) / dimBlock.y);
    detail::zipIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, plans[i].d_Data, plans[i],
        f, false);
  }

  // check for errors during gpu computation
  msl::syncStreams();
}

template<typename T>
template<typename R, typename T2, typename ZipFunctor>
msl::DMatrix<R> msl::DMatrix<T>::zip(DMatrix<T2>& b, ZipFunctor& f) {
  return zip<R>(b, f,
                Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipFunctor)>());
}

template<typename T>
template<typename R, typename T2, typename ZipFunctor>
msl::DMatrix<R> msl::DMatrix<T>::zip(DMatrix<T2>& b, ZipFunctor& f,
                                     Int2Type<true>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);

  // upload data first (if necessary)
  upload();
  b.upload();
  result.upload(1);  // alloc only

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].mLocal, plans[i].firstRow,
           plans[i].firstCol);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
    detail::zipKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data,
        result.getExecPlans()[i].d_Data, plans[i].size, f);
  }

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template<typename T>
template<typename R, typename T2, typename ZipFunctor>
msl::DMatrix<R> msl::DMatrix<T>::zip(DMatrix<T2>& b, ZipFunctor& f,
                                     Int2Type<false>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);

  // upload data first (if necessary)
  upload();
  b.upload();
  result.upload(1);  // alloc only

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size + dimBlock.x) / dimBlock.x);
    detail::zipKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data,
        result.getExecPlans()[i].d_Data, plans[i].size, f);
  }

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template<typename T>
template<typename R, typename T2, typename ZipIndexFunctor>
msl::DMatrix<R> msl::DMatrix<T>::zipIndex(DMatrix<T2>& b, ZipIndexFunctor& f) {
  return zipIndex<R>(
      b, f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipIndexFunctor)>());
}

template<typename T>
template<typename R, typename T2, typename ZipIndexFunctor>
msl::DMatrix<R> msl::DMatrix<T>::zipIndex(DMatrix<T2>& b, ZipIndexFunctor& f,
                                          Int2Type<true>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);

  // upload data first (if necessary)
  upload();
  b.upload();
  result.upload(1);  // alloc only

  // set kernel configuration
  int tile_width = f.getTileWidth();
  int tpbX = Muesli::tpb_x;
  int tpbY = Muesli::tpb_y;
  if (tile_width != -1) {
    tpbX = tpbY = tile_width;
  }

  // calculate shared memory size in bytes
  int smem_bytes = f.getSmemSize();

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].mLocal, plans[i].firstRow,
           plans[i].firstCol);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(tpbX, tpbY);
    dim3 dimGrid((plans[i].mLocal + dimBlock.x - 1) / dimBlock.x,
                 (plans[i].nLocal + dimBlock.y - 1) / dimBlock.y);
    detail::zipIndexKernel<<<dimGrid, dimBlock, smem_bytes, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data,
        result.getExecPlans()[i].d_Data, plans[i], f, f.useLocalIndices());
  }

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template<typename T>
template<typename R, typename T2, typename ZipIndexFunctor>
msl::DMatrix<R> msl::DMatrix<T>::zipIndex(DMatrix<T2>& b, ZipIndexFunctor& f,
                                          Int2Type<false>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);

  // upload data first (if necessary)
  upload();
  b.upload();
  result.upload(1);  // alloc only

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::tpb_x, Muesli::tpb_y);
    dim3 dimGrid((plans[i].mLocal + dimBlock.x - 1) / dimBlock.x,
                 (plans[i].nLocal + dimBlock.y - 1) / dimBlock.y);
    detail::zipIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data,
        result.getExecPlans()[i].d_Data, plans[i], f, false);
  }

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template<typename T>
template<typename FoldFunctor>
T msl::DMatrix<T>::fold(FoldFunctor& f, bool final_fold_on_cpu) {
  return fold(f,
              Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, FoldFunctor)>(),
              final_fold_on_cpu);
}

template<typename T>
template<typename FoldFunctor>
T msl::DMatrix<T>::fold(FoldFunctor& f, Int2Type<true>,
                        bool final_fold_on_cpu) {
  std::vector<int> blocks(Muesli::num_gpus);
  std::vector<int> threads(Muesli::num_gpus);
  T* gpu_results = new T[Muesli::num_gpus];
  int maxThreads = 1024;
  int maxBlocks = 1024;
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

  // calculate threads, blocks, etc.; allocate device memory
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    threads[i] =
        (plans[i].size < maxThreads) ?
            detail::nextPow2((plans[i].size + 1) / 2) : maxThreads;
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
    detail::reduce<T, FoldFunctor>(plans[i].size, plans[i].d_Data, d_odata[i],
                                   threads[i], blocks[i], f, Muesli::streams[i],
                                   i);
  }
  msl::syncStreams();

  // fold on gpus: step 2
  for (int i = 0; i < Muesli::num_gpus; i++) {
    if (blocks[i] > 1) {
      f.notify();
      cudaSetDevice(i);
      int threads =
          (detail::nextPow2(blocks[i]) == blocks[i]) ?
              blocks[i] : detail::nextPow2(blocks[i]) / 2;
      detail::reduce<T, FoldFunctor>(blocks[i], d_odata[i], d_odata[i], threads,
                                     1, f, Muesli::streams[i], i);
    }
  }
  msl::syncStreams();

  // copy final sum from device to host
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(&gpu_results[i], d_odata[i], sizeof(T),
                        cudaMemcpyDeviceToHost, Muesli::streams[i]));
  }
  msl::syncStreams();

  T final_result, result;
  if (final_fold_on_cpu) {
    // calculate local result for all GPUs
    T tmp = gpu_results[0];
    for (int i = 1; i < Muesli::num_gpus; i++) {
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
    T* d_gpu_results;
    if (Muesli::num_gpus > 1) {  // if there are more than 1 GPU
      f.notify();
      cudaSetDevice(0);         // calculate local result on device 0

      // upload data
      CUDA_CHECK_RETURN(
          cudaMalloc((void**) &d_gpu_results, Muesli::num_gpus * sizeof(T)));
      CUDA_CHECK_RETURN(
          cudaMemcpyAsync(d_gpu_results, gpu_results,
                          Muesli::num_gpus * sizeof(T), cudaMemcpyHostToDevice,
                          Muesli::streams[0]));
      CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[0]));

      // final (local) fold
detail      ::reduce<T, FoldFunctor>(Muesli::num_gpus, d_gpu_results, d_gpu_results,
                               Muesli::num_gpus, 1, f, Muesli::streams[0], 0);
      CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[0]));

      // copy result from device to host
CUDA_CHECK_RETURN      (cudaMemcpyAsync(&local_result, d_gpu_results, sizeof(T),
                       cudaMemcpyDeviceToHost, Muesli::streams[0]));
      CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[0]));CUDA_CHECK_RETURN
      (cudaFree(d_gpu_results));
    } else {
      local_result = gpu_results[0];
    }

    if (np > 1) {
      // gather all local results
      msl::allgather(&local_result, local_results, 1);

      // calculate global result from local results
      // upload data
      cudaMalloc((void**) &d_gpu_results, np * sizeof(T));
      CUDA_CHECK_RETURN(
          cudaMemcpyAsync(d_gpu_results, local_results, np * sizeof(T),
                          cudaMemcpyHostToDevice, Muesli::streams[0]));

      // final fold
      detail::reduce<T, FoldFunctor>(np, d_gpu_results, d_gpu_results, np, 1, f,
                                     Muesli::streams[0], 0);
      CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[0]));

      // copy final result from device to host
CUDA_CHECK_RETURN      (cudaMemcpyAsync(&final_result, d_gpu_results, sizeof(T),
                       cudaMemcpyDeviceToHost, Muesli::streams[0]));
      CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[0]));CUDA_CHECK_RETURN
      (cudaFree(d_gpu_results));
    } else {
      final_result = local_result;
    }
  }

  // Cleanup
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[i]));CUDA_CHECK_RETURN
    (cudaFree(d_odata[i]));
  }
  delete[] gpu_results;
  delete[] d_odata;
  delete[] local_results;

  return final_result;
}

template<typename T>
template<typename FoldFunctor>
T msl::DMatrix<T>::fold(FoldFunctor& f, Int2Type<false>,
                        bool final_fold_on_cpu) {
  std::vector<int> blocks(Muesli::num_gpus);
  std::vector<int> threads(Muesli::num_gpus);
  T* gpu_results = new T[Muesli::num_gpus];
  int maxThreads = 1024;
  int maxBlocks = 1024;
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

  // calculate threads, blocks, etc.; allocate device memory
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    threads[i] =
        (plans[i].size < maxThreads) ?
            detail::nextPow2((plans[i].size + 1) / 2) : maxThreads;
    blocks[i] = plans[i].size / threads[i];
    if (blocks[i] > maxBlocks) {
      blocks[i] = maxBlocks;
    }
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_odata[i], blocks[i] * sizeof(T)));
  }

  // fold on gpus: step 1
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    detail::reduce<T, FoldFunctor>(plans[i].size, plans[i].d_Data, d_odata[i],
                                   threads[i], blocks[i], f, Muesli::streams[i],
                                   i);
  }
  msl::syncStreams();

  // fold on gpus: step 2
  for (int i = 0; i < Muesli::num_gpus; i++) {
    if (blocks[i] > 1) {
      cudaSetDevice(i);
      int threads =
          (detail::nextPow2(blocks[i]) == blocks[i]) ?
              blocks[i] : detail::nextPow2(blocks[i]) / 2;
      detail::reduce<T, FoldFunctor>(blocks[i], d_odata[i], d_odata[i], threads,
                                     1, f, Muesli::streams[i], i);
    }
  }
  msl::syncStreams();

  // copy final sum from device to host
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(&gpu_results[i], d_odata[i], sizeof(T),
                        cudaMemcpyDeviceToHost, Muesli::streams[i]));
  }
  msl::syncStreams();

  T final_result, result;
  final_fold_on_cpu = 0;  // lambda functions are only compiled as __device__ functions
  if (final_fold_on_cpu) {
    // calculate local result for all GPUs
    T tmp = gpu_results[0];
    for (int i = 1; i < Muesli::num_gpus; i++) {
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
    T* d_gpu_results;
    if (Muesli::num_gpus > 1) {  // if there are more than 1 GPU
      cudaSetDevice(0);         // calculate local result on device 0
      // upload data
      CUDA_CHECK_RETURN(
          cudaMalloc((void**) &d_gpu_results, Muesli::num_gpus * sizeof(T)));
      CUDA_CHECK_RETURN(
          cudaMemcpyAsync(d_gpu_results, gpu_results,
                          Muesli::num_gpus * sizeof(T), cudaMemcpyHostToDevice,
                          Muesli::streams[0]));
      CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[0]));

      // final (local) fold
detail      ::reduce<T, FoldFunctor>(Muesli::num_gpus, d_gpu_results, d_gpu_results,
                               Muesli::num_gpus, 1, f, Muesli::streams[0], 0);
      CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[0]));

      // copy result from device to host
CUDA_CHECK_RETURN      (cudaMemcpyAsync(&local_result, d_gpu_results, sizeof(T),
                       cudaMemcpyDeviceToHost, Muesli::streams[0]));
      CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[0]));CUDA_CHECK_RETURN
      (cudaFree(d_gpu_results));
    } else {
      local_result = gpu_results[0];
    }

    if (np > 1) {
      // gather all local results
      msl::allgather(&local_result, local_results, 1);

      // calculate global result from local results
      // upload data
      cudaMalloc((void**) &d_gpu_results, np * sizeof(T));
      CUDA_CHECK_RETURN(
          cudaMemcpyAsync(d_gpu_results, local_results, np * sizeof(T),
                          cudaMemcpyHostToDevice, Muesli::streams[0]));

      // final fold
      detail::reduce<T, FoldFunctor>(np, d_gpu_results, d_gpu_results, np, 1, f,
                                     Muesli::streams[0], 0);
      CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[0]));

      // copy final result from device to host
CUDA_CHECK_RETURN      (cudaMemcpyAsync(&final_result, d_gpu_results, sizeof(T),
                       cudaMemcpyDeviceToHost, Muesli::streams[0]));
      CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[0]));CUDA_CHECK_RETURN
      (cudaFree(d_gpu_results));
    } else {
      final_result = local_result;
    }
  }

  // Cleanup
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[i]));CUDA_CHECK_RETURN
    (cudaFree(d_odata[i]));
  }
  delete[] gpu_results;
  delete[] d_odata;
  delete[] local_results;

  return final_result;
}

template<typename T>
template<typename FoldFunctor>
msl::DArray<T> msl::DMatrix<T>::foldRows(FoldFunctor& f) {
  return foldRows(
      f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, FoldFunctor)>());
}

template<typename T>
template<typename FoldFunctor>
msl::DArray<T> msl::DMatrix<T>::foldRows(FoldFunctor& f, Int2Type<true>) {

  int num_gpus = Muesli::num_gpus;

  // local partition is row distributed on GPUs
  // more gpus than rows lead to errors
  if (nLocal < Muesli::num_gpus) {
    num_gpus = nLocal;
  }

  std::vector<int> blocks(num_gpus);
  std::vector<int> threads(num_gpus);
  T** gpu_results = new T*[num_gpus];
  int maxThreads = 1024;
  int maxBlocks = 65535;
  for (int i = 0; i < num_gpus; i++) {
    threads[i] = maxThreads;
    gpu_results[i] = new T[plans[i].nLocal];
  }
  T* local_results = new T[nLocal];
  T* global_results = new T[np * nLocal];
  T** d_odata = new T*[num_gpus];

  upload();

  //
  // Step 1: local fold
  //

  // calculate threads, blocks, etc.; allocate device memory
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    threads[i] =
        (plans[i].mLocal < maxThreads) ?
            detail::nextPow2((plans[i].mLocal + 1) / 2) : maxThreads;
    blocks[i] = plans[i].nLocal;
    if (blocks[i] > maxBlocks) {
      blocks[i] = maxBlocks;  // possibly throw exception, since this case is not handled yet, but should actualy never occur
    }
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_odata[i], blocks[i] * sizeof(T)));
  }

  // fold on gpus: step 1
  for (int i = 0; i < num_gpus; i++) {
    f.notify();
    cudaSetDevice(i);
    detail::foldRows<T, FoldFunctor>(plans[i].mLocal, plans[i].d_Data,
                                     d_odata[i], threads[i], blocks[i], f,
                                     Muesli::streams[i], i);
  }
  msl::syncStreams();

  // copy result arrays from device to host
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(gpu_results[i], d_odata[i], blocks[i] * sizeof(T),
                        cudaMemcpyDeviceToHost, Muesli::streams[i]));
  }
  msl::syncStreams();

  // final fold on CPU
  // calculate local result for all GPUs
  for (int i = 0; i < num_gpus; ++i) {
    for (int j = 0; j < plans[i].nLocal; ++j) {
      local_results[i * plans[i].nLocal + j] = gpu_results[i][j];
    }
  }

  // gather all local results
  msl::allgather(local_results, global_results, nLocal);

  // calculate global result from local results, TODO: could be parallelized with openMP
  int localRowsPerColBlock = blocksInRow * nLocal;
  for (int i = 0; i < blocksInCol; ++i) {
    for (int j = 0; j < localRowsPerColBlock; ++j) {
      int index = i * localRowsPerColBlock + j;
      int result_index = i * nLocal + (j % nLocal);
      if (j < nLocal) {
        global_results[result_index] = global_results[index];
      } else {
        global_results[result_index] = f(global_results[result_index],
                                         global_results[index]);
      }
    }
  }

  // Cleanup
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[i]));CUDA_CHECK_RETURN
    (cudaFree(d_odata[i]));
  }

  msl::DArray<T> result_array(n, global_results, Distribution::DIST);  // just takes the first n folded results from globalResults array

  for (int i = 0; i < num_gpus; ++i) {
    delete[] gpu_results[i];
  }
  delete[] gpu_results;
  delete[] d_odata;
  delete[] local_results;
  delete[] global_results;

  return result_array;
}

template<typename T>
template<typename FoldFunctor>
msl::DArray<T> msl::DMatrix<T>::foldRows(FoldFunctor& f, Int2Type<false>) {
  int num_gpus = Muesli::num_gpus;

  // local partition is row distributed on GPUs
  // more gpus than rows lead to errors
  if (nLocal < Muesli::num_gpus) {
    num_gpus = nLocal;
  }

  std::vector<int> blocks(num_gpus);
  std::vector<int> threads(num_gpus);
  T** gpu_results = new T*[num_gpus];
  int maxThreads = 1024;
  int maxBlocks = 65535;
  for (int i = 0; i < num_gpus; i++) {
    threads[i] = maxThreads;
    gpu_results[i] = new T[plans[i].nLocal];
  }
  T* local_results = new T[nLocal];
  T* global_results = new T[np * nLocal];
  T** d_odata = new T*[num_gpus];

  upload();

  //
  // Step 1: local fold
  //

  // calculate threads, blocks, etc.; allocate device memory
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    threads[i] =
        (plans[i].mLocal < maxThreads) ?
            detail::nextPow2((plans[i].mLocal + 1) / 2) : maxThreads;
    blocks[i] = plans[i].nLocal;
    if (blocks[i] > maxBlocks) {
      blocks[i] = maxBlocks;  // possibly throw exception, since this case is not handled yet, but should actualy never occur
    }
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_odata[i], blocks[i] * sizeof(T)));
  }

  // fold on gpus: step 1
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    detail::foldRows<T, FoldFunctor>(plans[i].mLocal, plans[i].d_Data,
                                     d_odata[i], threads[i], blocks[i], f,
                                     Muesli::streams[i], i);
  }
  msl::syncStreams();

  // copy result arrays from device to host
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(gpu_results[i], d_odata[i], blocks[i] * sizeof(T),
                        cudaMemcpyDeviceToHost, Muesli::streams[i]));
  }
  msl::syncStreams();

  // final fold on CPU
  // calculate local result for all GPUs
  for (int i = 0; i < num_gpus; ++i) {
    for (int j = 0; j < plans[i].nLocal; ++j) {
      local_results[i * plans[i].nLocal + j] = gpu_results[i][j];
    }
  }

  // gather all local results
  msl::allgather(local_results, global_results, nLocal);

  // calculate global result from local results, TODO: could be parallelized with openMP
  int localRowsPerColBlock = blocksInRow * nLocal;
  for (int i = 0; i < blocksInCol; ++i) {
    for (int j = 0; j < localRowsPerColBlock; ++j) {
      int index = i * localRowsPerColBlock + j;
      int result_index = i * nLocal + (j % nLocal);
      if (j < nLocal) {
        global_results[result_index] = global_results[index];
      } else {
        global_results[result_index] = f(global_results[result_index],
                                         global_results[index]);
      }
    }
  }

  // Cleanup
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[i]));CUDA_CHECK_RETURN
    (cudaFree(d_odata[i]));
  }

  msl::DArray<T> result_array(n, global_results, Distribution::DIST);  // just takes the first n folded results from globalResults array

  for (int i = 0; i < num_gpus; ++i) {
    delete[] gpu_results[i];
  }
  delete[] gpu_results;
  delete[] d_odata;
  delete[] local_results;
  delete[] global_results;

  return result_array;
}

template<typename T>
template<typename FoldFunctor>
msl::DArray<T> msl::DMatrix<T>::foldCols(FoldFunctor& f) {
  return foldCols(
      f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, FoldFunctor)>());
}

// The idea is to reduce one column per thread block, thus (#thread_blocks = mLocal )

template<typename T>
template<typename FoldFunctor>
msl::DArray<T> msl::DMatrix<T>::foldCols(FoldFunctor& f, Int2Type<true>) {
  int num_gpus = Muesli::num_gpus;

  // local partition is row distributed on GPUs
  // more gpus than rows lead to errors
  if (nLocal < Muesli::num_gpus) {
    num_gpus = nLocal;
  }

  std::vector<int> blocks(num_gpus);
  std::vector<int> threads(num_gpus);
  T** gpu_results = new T*[num_gpus];
  int maxThreads = 1024;
  int maxBlocks = 65535;
  for (int i = 0; i < num_gpus; i++) {
    threads[i] = maxThreads;
    gpu_results[i] = new T[plans[i].mLocal];
  }
  T* local_results = new T[mLocal];
  T* global_results = new T[np * mLocal];
  T** d_odata = new T*[num_gpus];

  upload();

  //
  // Step 1: local fold
  //

  // calculate threads, blocks, etc.; allocate device memory
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    threads[i] =
        (plans[i].nLocal < maxThreads) ?
            detail::nextPow2((plans[i].nLocal + 1) / 2) : maxThreads;
    blocks[i] = plans[i].mLocal;
    if (blocks[i] > maxBlocks) {
      blocks[i] = maxBlocks;  // possibly throw exception, since this case is not handled yet, but should actualy never occur
    }
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_odata[i], blocks[i] * sizeof(T)));
  }

  // fold on gpus: step 1
  for (int i = 0; i < num_gpus; i++) {
    f.notify();
    cudaSetDevice(i);
    detail::foldCols<T, FoldFunctor>(plans[i].size, plans[i].d_Data, d_odata[i],
                                     threads[i], blocks[i], f,
                                     Muesli::streams[i], i);
  }
  msl::syncStreams();

  // copy result arrays from device to host
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(gpu_results[i], d_odata[i], blocks[i] * sizeof(T),
                        cudaMemcpyDeviceToHost, Muesli::streams[i]));
  }
  msl::syncStreams();

  // final fold on CPU
  // calculate local result for all GPUs
  for (int i = 0; i < mLocal; ++i) {
    local_results[i] = gpu_results[0][i];
  }

  for (int i = 1; i < num_gpus; i++) {
    for (int j = 0; j < mLocal; ++j) {
      local_results[j] = f(local_results[j], gpu_results[i][j]);
    }
  }

  // gather all local results
  msl::allgather(local_results, global_results, mLocal);

  // calculate global result from local results
  int end_index = np * mLocal;
  for (int i = blocksInRow * mLocal; i < end_index; ++i) {
    int index = i % m;
    global_results[index] = f(global_results[index], global_results[i]);
  }

  // Cleanup
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[i]));CUDA_CHECK_RETURN
    (cudaFree(d_odata[i]));
  }

  msl::DArray<T> result_array(m, global_results, Distribution::DIST);  // just takes the first n folded results from globalResults array

  for (int i = 0; i < num_gpus; ++i) {
    delete[] gpu_results[i];
  }
  delete[] gpu_results;
  delete[] d_odata;
  delete[] local_results;
  delete[] global_results;

  return result_array;
}

template<typename T>
template<typename FoldFunctor>
msl::DArray<T> msl::DMatrix<T>::foldCols(FoldFunctor& f, Int2Type<false>) {
  int num_gpus = Muesli::num_gpus;

  // local partition is row distributed on GPUs
  // more gpus than rows lead to errors
  if (nLocal < Muesli::num_gpus) {
    num_gpus = nLocal;
  }

  std::vector<int> blocks(num_gpus);
  std::vector<int> threads(num_gpus);
  T** gpu_results = new T*[num_gpus];
  int maxThreads = 1024;
  int maxBlocks = 65535;
  for (int i = 0; i < num_gpus; i++) {
    threads[i] = maxThreads;
    gpu_results[i] = new T[plans[i].mLocal];
  }
  T* local_results = new T[mLocal];
  T* global_results = new T[np * mLocal];
  T** d_odata = new T*[num_gpus];

  upload();

  //
  // Step 1: local fold
  //

  // calculate threads, blocks, etc.; allocate device memory
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    threads[i] =
        (plans[i].nLocal < maxThreads) ?
            detail::nextPow2((plans[i].nLocal + 1) / 2) : maxThreads;
    blocks[i] = plans[i].mLocal;
    if (blocks[i] > maxBlocks) {
      blocks[i] = maxBlocks;  // possibly throw exception, since this case is not handled yet, but should actualy never occur
    }
    CUDA_CHECK_RETURN(cudaMalloc((void**) &d_odata[i], blocks[i] * sizeof(T)));
  }

  // fold on gpus: step 1
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    detail::foldCols<T, FoldFunctor>(plans[i].size, plans[i].d_Data, d_odata[i],
                                     threads[i], blocks[i], f,
                                     Muesli::streams[i], i);
  }
  msl::syncStreams();

  // copy result arrays from device to host
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(gpu_results[i], d_odata[i], blocks[i] * sizeof(T),
                        cudaMemcpyDeviceToHost, Muesli::streams[i]));
  }
  msl::syncStreams();

  // final fold on CPU
  // calculate local result for all GPUs
  for (int i = 0; i < mLocal; ++i) {
    local_results[i] = gpu_results[0][i];
  }

  for (int i = 1; i < num_gpus; i++) {
    for (int j = 0; j < mLocal; ++j) {
      local_results[j] = f(local_results[j], gpu_results[i][j]);
    }
  }

  // gather all local results
  msl::allgather(local_results, global_results, mLocal);

  // calculate global result from local results
  int end_index = np * mLocal;
  for (int i = blocksInRow * mLocal; i < end_index; ++i) {
    int index = i % m;
    global_results[index] = f(global_results[index], global_results[i]);
  }

  // Cleanup
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN (cudaStreamSynchronize(Muesli::streams[i]));CUDA_CHECK_RETURN
    (cudaFree(d_odata[i]));
  }

  msl::DArray<T> result_array(m, global_results, Distribution::DIST);  // just takes the first n folded results from globalResults array

  for (int i = 0; i < num_gpus; ++i) {
    delete[] gpu_results[i];
  }
  delete[] gpu_results;
  delete[] d_odata;
  delete[] local_results;
  delete[] global_results;

  return result_array;
}


