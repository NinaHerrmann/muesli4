/*
 * darray.cu
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

template <typename T>
template <typename MapFunctor>
void msl::DArray<T>::mapInPlace(MapFunctor& f)
{
  mapInPlace(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapFunctor)>());
}

template <typename T>
template <typename MapFunctor>
void msl::DArray<T>::mapInPlace(MapFunctor& f, Int2Type<true>)
{
  // upload data first (if necessary)
  upload();

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
        plans[i].d_Data, plans[i].d_Data, plans[i].size, f);
  }

  // check for errors during gpu computation
  msl::syncStreams();
}

template <typename T>
template <typename MapFunctor>
void msl::DArray<T>::mapInPlace(MapFunctor& f, Int2Type<false>)
{
  // upload data first (if necessary)
  upload();

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, plans[i].d_Data, plans[i].size, f);
  }

  msl::syncStreams();
}

template <typename T>
template <typename MapIndexFunctor>
void msl::DArray<T>::mapIndexInPlace(MapIndexFunctor& f)
{
  mapIndexInPlace(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapIndexFunctor)>());
}

template <typename T>
template <typename MapIndexFunctor>
void msl::DArray<T>::mapIndexInPlace(MapIndexFunctor& f, Int2Type<true>)
{
  // upload data first (if necessary)
  upload();

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

  // check for errors during gpu computation
  msl::syncStreams();
}

template <typename T>
template <typename MapIndexFunctor>
void msl::DArray<T>::mapIndexInPlace(MapIndexFunctor& f, Int2Type<false>)
{
  // upload data first (if necessary)
  upload();

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, plans[i].d_Data, plans[i].nLocal, plans[i].first, f, false);
  }

  // check for errors during gpu computation
  msl::syncStreams();
}

template <typename T>
template <typename R, typename MapFunctor>
msl::DArray<R> msl::DArray<T>::map(MapFunctor& f)
{
  return map<R, MapFunctor>(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapFunctor)>());
}

template <typename T>
template <typename R, typename MapFunctor>
msl::DArray<R> msl::DArray<T>::map(MapFunctor& f, Int2Type<true>)
{
  DArray<R> result(n, dist);

  // upload data first (if necessary)
  upload();
  result.upload(1); // alloc only

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

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template <typename T>
template <typename R, typename MapFunctor>
msl::DArray<R> msl::DArray<T>::map(MapFunctor& f, Int2Type<false>)
{
  DArray<R> result(n, dist);

  // upload data first (if necessary)
  upload();
  result.upload(1); // alloc only

  // map
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
            plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].size, f);
  }

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template <typename T>
template <typename R, typename MapIndexFunctor>
msl::DArray<R> msl::DArray<T>::mapIndex(MapIndexFunctor& f)
{
  return mapIndex<R, MapIndexFunctor>(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapIndexFunctor)>());
}

template <typename T>
template <typename R, typename MapIndexFunctor>
msl::DArray<R> msl::DArray<T>::mapIndex(MapIndexFunctor& f, Int2Type<true>)
{
  DArray<R> result(n, dist);

  // upload data first (if necessary)
  upload();
  result.upload(1); // alloc only

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

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template <typename T>
template <typename R, typename MapIndexFunctor>
msl::DArray<R> msl::DArray<T>::mapIndex(MapIndexFunctor& f, Int2Type<false>)
{
  DArray<R> result(n, dist);

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

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template <typename T>
template <typename MapStencilFunctor>
void msl::DArray<T>::mapStencilInPlace(MapStencilFunctor& f, T neutral_value)
{
  // Obtain stencil size.
  int stencil_size = f.getStencilSize();
  int size = nLocal + 2*stencil_size;
  // Prepare padded local matrix. We need additional 2*stencil_size entries.
  T* padded_local_array = new T[size];
  // Update data in main memory if necessary.
  download();

  // Gather border regions.
  MPI_Status stat;
  MPI_Request req;
  // Top down (send last stencil_size entries to successor):
  // Non-blocking send.
  if (Muesli::proc_id < Muesli::num_local_procs-1) {
    T* buffer = localPartition + nLocal - stencil_size;
    MSL_ISend(Muesli::proc_id+1, buffer, req, stencil_size, msl::MYTAG);
  }

  // Copy localPartition to padded_local_matrix
  std::copy(localPartition, localPartition+nLocal, padded_local_array+stencil_size);

  // Blocking receive.
  if (Muesli::proc_id > 0) {
    MSL_Recv(Muesli::proc_id-1, padded_local_array, stat, stencil_size, msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id < Muesli::num_local_procs-1) {
    MPI_Wait(&req, &stat);
  }

  // Bottom up (send first stencil_size entries to predecessor):
  // Non-blocking send.
  if (Muesli::proc_id > 0) {
    MSL_ISend(Muesli::proc_id-1, localPartition, req, stencil_size, msl::MYTAG);
  }

  // Blocking receive.
  if (Muesli::proc_id < Muesli::num_local_procs-1) {
    T* buffer = padded_local_array + nLocal + stencil_size;
    MSL_Recv(Muesli::proc_id+1, buffer, stat, stencil_size, msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id > 0) {
    MPI_Wait(&req, &stat);
  }

  // Process 0 and process n-1 need to fill upper (lower) border regions with neutral value
  if (Muesli::proc_id == 0) {
    for (int i = 0; i < stencil_size; i++) {
      padded_local_array[i] = neutral_value;
    }
  }
  if (Muesli::proc_id == Muesli::num_local_procs-1) {
    for (int i = nLocal+stencil_size; i < nLocal+2*stencil_size; i++) {
      padded_local_array[i] = neutral_value;
    }
  }

  int tile_width = f.getTileWidth();

  // Upload buffers.
  std::vector<T*> d_padded_local_array(Muesli::num_gpus);
  // Create padded local array.
  msl::PLArray<T> pla(n, nLocal, stencil_size, tile_width, neutral_value);
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_padded_local_array[i],sizeof(T)*size));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(d_padded_local_array[i],
                    padded_local_array,
                    sizeof(T)*size,
                    cudaMemcpyHostToDevice,
                    Muesli::streams[i]));
    pla.addDevicePtr(d_padded_local_array[i]);
  }

  // Upload data (this).
  upload();

  // Upload padded local partitions.
  std::vector<PLArray<T>*> d_pla(Muesli::num_gpus);
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    pla.setFirstIndexGPU(plans[i].first);
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_pla[i], sizeof(PLArray<T>)));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(d_pla[i], &pla, sizeof(PLArray<T>), cudaMemcpyHostToDevice, Muesli::streams[i]));
    pla.update();
  }

  // Map stencil
  int smem_size = (tile_width+2*stencil_size)*sizeof(T);
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].first);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(tile_width);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapStencilKernel<<<dimGrid, dimBlock, smem_size, Muesli::streams[i]>>>(
        plans[i].d_Data, plans[i].d_Data, plans[i], d_pla[i], f, tile_width);
  }

  // Check for errors during gpu computation.
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(Muesli::streams[i]));
    CUDA_CHECK_RETURN(cudaFree(d_pla[i]));
    CUDA_CHECK_RETURN(cudaFree(d_padded_local_array[i]));
  }

  // Clean up.
  delete[] padded_local_array;
}

template <typename T>
template <typename R, typename MapStencilFunctor>
msl::DArray<R> msl::DArray<T>::mapStencil(MapStencilFunctor& f, T neutral_value)
{
  // Obtain stencil size.
  int stencil_size = f.getStencilSize();
  int size = nLocal + 2*stencil_size;
  // Prepare padded local matrix. We need additional 2*stencil_size entries.
  T* padded_local_array = new T[nLocal + 2*size];
  // Update data in main memory if necessary.
  download();

  // Gather border regions.
  MPI_Status stat;
  MPI_Request req;
  // Top down (send last stencil_size entries to successor):
  // Non-blocking send.
  if (Muesli::proc_id < Muesli::num_local_procs-1) {
    T* buffer = localPartition + nLocal - stencil_size;
    MSL_ISend(Muesli::proc_id+1, buffer, req, stencil_size, msl::MYTAG);
  }

  // Copy localPartition to padded_local_matrix
  std::copy(localPartition, localPartition+nLocal, padded_local_array+stencil_size);

  // Blocking receive.
  if (Muesli::proc_id > 0) {
    MSL_Recv(Muesli::proc_id-1, padded_local_array, stat, stencil_size, msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id < Muesli::num_local_procs-1) {
    MPI_Wait(&req, &stat);
  }

  // Bottom up (send first stencil_size entries to predecessor):
  // Non-blocking send.
  if (Muesli::proc_id > 0) {
    MSL_ISend(Muesli::proc_id-1, localPartition, req, stencil_size, msl::MYTAG);
  }

  // Blocking receive.
  if (Muesli::proc_id < Muesli::num_local_procs-1) {
    T* buffer = padded_local_array + nLocal + stencil_size;
    MSL_Recv(Muesli::proc_id+1, buffer, stat, stencil_size, msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id > 0) {
    MPI_Wait(&req, &stat);
  }

  // Process 0 and process n-1 need to fill upper (lower) border regions with neutral value
  if (Muesli::proc_id == 0) {
    for (int i = 0; i < stencil_size; i++) {
      padded_local_array[i] = neutral_value;
    }
  }
  if (Muesli::proc_id == Muesli::num_local_procs-1) {
    for (int i = nLocal+stencil_size; i < nLocal+2*stencil_size; i++) {
      padded_local_array[i] = neutral_value;
    }
  }

  int tile_width = f.getTileWidth();

  // Upload buffers.
  std::vector<T*> d_padded_local_array(Muesli::num_gpus);
  // Create padded local array.
  msl::PLArray<T> pla(n, nLocal, stencil_size, tile_width, neutral_value);
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_padded_local_array[i],sizeof(T)*size));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(d_padded_local_array[i],
                    padded_local_array,
                    sizeof(T)*size,
                    cudaMemcpyHostToDevice,
                    Muesli::streams[i]));
    pla.addDevicePtr(d_padded_local_array[i]);
  }

  // Upload data (this).
  upload();

  // Upload padded local partitions.
  std::vector<PLArray<T>*> d_pla(Muesli::num_gpus);
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    pla.setFirstIndexGPU(plans[i].first);
    CUDA_CHECK_RETURN(cudaMalloc((void**)&d_pla[i], sizeof(PLArray<T>)));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(d_pla[i], &pla, sizeof(PLArray<T>), cudaMemcpyHostToDevice, Muesli::streams[i]));
    pla.update();
  }

  // Map stencil
  int smem_size = (tile_width+2*stencil_size)*sizeof(T);
  DArray<T> result(n, dist);
  result.upload(1); // alloc only
  for (int i = 0; i < Muesli::num_gpus; i++) {
    f.init(plans[i].nLocal, plans[i].first);
    f.notify();

    cudaSetDevice(i);
    dim3 dimBlock(tile_width);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::mapStencilKernel<<<dimGrid, dimBlock, smem_size, Muesli::streams[i]>>>(
        plans[i].d_Data, result.getExecPlans()[i].d_Data, plans[i], d_pla[i], f, tile_width);
  }

  // Check for errors during gpu computation.
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    CUDA_CHECK_RETURN(cudaStreamSynchronize(Muesli::streams[i]));
    CUDA_CHECK_RETURN(cudaFree(d_pla[i]));
    CUDA_CHECK_RETURN(cudaFree(d_padded_local_array[i]));
  }

  // Clean up.
  delete[] padded_local_array;

  return result;
}

template <typename T>
template <typename T2, typename ZipFunctor>
void msl::DArray<T>::zipInPlace(DArray<T2>& b, ZipFunctor& f)
{
  zipInPlace(b, f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipFunctor)>());
}

template <typename T>
template <typename T2, typename ZipFunctor>
void msl::DArray<T>::zipInPlace(DArray<T2>& b, ZipFunctor& f, Int2Type<true>)
{
  // upload data first (if necessary)
  upload();
  b.upload();

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

  // check for errors during gpu computation
  msl::syncStreams();
}

template <typename T>
template <typename T2, typename ZipFunctor>
void msl::DArray<T>::zipInPlace(DArray<T2>& b, ZipFunctor& f, Int2Type<false>)
{
  // upload data first (if necessary)
  upload();
  b.upload();

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::zipKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, plans[i].d_Data, plans[i].size, f);
  }

  // check for errors during gpu computation
  msl::syncStreams();
}

template <typename T>
template <typename T2, typename ZipIndexFunctor>
void msl::DArray<T>::zipIndexInPlace(DArray<T2>& b, ZipIndexFunctor& f)
{
  zipIndexInPlace(b, f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipIndexFunctor)>());
}

template <typename T>
template <typename T2, typename ZipIndexFunctor>
void msl::DArray<T>::zipIndexInPlace(DArray<T2>& b, ZipIndexFunctor& f, Int2Type<true>)
{
  // upload data first (if necessary)
  upload();
  b.upload();

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

  // check for errors during gpu computation
  msl::syncStreams();
}

template <typename T>
template <typename T2, typename ZipIndexFunctor>
void msl::DArray<T>::zipIndexInPlace(DArray<T2>& b, ZipIndexFunctor& f, Int2Type<false>)
{
  // upload data first (if necessary)
  upload();
  b.upload();

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::zipIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, plans[i].d_Data, plans[i].nLocal,
        plans[i].first, f, false);
  }

  // check for errors during gpu computation
  msl::syncStreams();
}

template <typename T>
template <typename R, typename T2, typename ZipFunctor>
msl::DArray<R> msl::DArray<T>::zip(DArray<T2>& b, ZipFunctor& f)
{
  return zip<R>(b, f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipFunctor)>());
}

template <typename T>
template <typename R, typename T2, typename ZipFunctor>
msl::DArray<R> msl::DArray<T>::zip(DArray<T2>& b, ZipFunctor& f, Int2Type<true>)
{
  DArray<R> result(n, dist);

  // upload data first (if necessary)
  upload();
  b.upload();
  result.upload(1); // alloc only

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

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template <typename T>
template <typename R, typename T2, typename ZipFunctor>
msl::DArray<R> msl::DArray<T>::zip(DArray<T2>& b, ZipFunctor& f, Int2Type<false>)
{
  DArray<R> result(n, dist);

  // upload data first (if necessary)
  upload();
  b.upload();
  result.upload(1); // alloc only

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::zipKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].size, f);
  }

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template <typename T>
template <typename R, typename T2, typename ZipIndexFunctor>
msl::DArray<R> msl::DArray<T>::zipIndex(DArray<T2>& b, ZipIndexFunctor& f)
{
  return zipIndex<R>(b, f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipIndexFunctor)>());
}

template <typename T>
template <typename R, typename T2, typename ZipIndexFunctor>
msl::DArray<R> msl::DArray<T>::zipIndex(DArray<T2>& b, ZipIndexFunctor& f, Int2Type<true>)
{
  DArray<R> result(n, dist);

  // upload data first (if necessary)
  upload();
  b.upload();
  result.upload(1); // alloc only

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

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template <typename T>
template <typename R, typename T2, typename ZipIndexFunctor>
msl::DArray<R> msl::DArray<T>::zipIndex(DArray<T2>& b, ZipIndexFunctor& f, Int2Type<false>)
{
  DArray<R> result(n, dist);

  // upload data first (if necessary)
  upload();
  b.upload();
  result.upload(1); // alloc only

  // zip
  for (int i = 0; i < Muesli::num_gpus; i++) {
    cudaSetDevice(i);
    dim3 dimBlock(Muesli::threads_per_block);
    dim3 dimGrid((plans[i].size+dimBlock.x)/dimBlock.x);
    detail::zipIndexKernel<<<dimGrid, dimBlock, 0, Muesli::streams[i]>>>(
        plans[i].d_Data, b.getExecPlans()[i].d_Data, result.getExecPlans()[i].d_Data, plans[i].nLocal,
        plans[i].first, f, false);
  }

  // check for errors during gpu computation
  msl::syncStreams();

  return result;
}

template <typename T>
template <typename FoldFunctor>
T msl::DArray<T>::fold(FoldFunctor& f, bool final_float_on_cpu)
{
  return fold(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, FoldFunctor)>(), final_float_on_cpu);
}

template <typename T>
template <typename FoldFunctor>
T msl::DArray<T>::fold(FoldFunctor& f, Int2Type<true>, bool final_fold_on_cpu)
{
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

template <typename T>
template <typename FoldFunctor>
T msl::DArray<T>::fold(FoldFunctor& f, Int2Type<false>, bool final_fold_on_cpu)
{
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
