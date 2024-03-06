/*
 * fold_kernels.cu
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

#include "map_kernels.cuh"

// This reduction kernel is based on Kernel6 presented in the Nvidia Reduction Whitepaper
// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
bool msl::detail::isPow2(size_t x) {
    return ((x & (x - 1)) == 0);
}

size_t msl::detail::nextPow2(size_t x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

template<typename T, typename F, size_t blockSize, bool nIsPow2>
__global__ void msl::detail::foldKernel(T *g_idata, T *g_odata, size_t n, F func) {
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockSize + threadIdx.x;
    size_t gridSize = blockSize * gridDim.x;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim). More blocks will result
    // in a larger gridSize and therefore fewer elements per thread.
    T result = g_idata[i];
    i += gridSize;

    while (i < n) {
        result = func(result, g_idata[i]);
        i += gridSize;
    }
    sdata[tid] = result;
    __syncthreads();

    // perform reduction in shared memory
    if ((blockSize >= 1024) && (tid < 512)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 512]);
    }
    __syncthreads();

    if ((blockSize >= 512) && (tid < 256)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 256]);
    }
    __syncthreads();

    if ((blockSize >= 256) && (tid < 128)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 128]);
    }
    __syncthreads();

    if ((blockSize >= 128) && (tid < 64)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 64]);
    }
    __syncthreads();

    if ((blockSize >= 64) && (tid < 32)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 32]);
    }
    __syncthreads();

    if ((blockSize >= 32) && (tid < 16)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 16]);
    }
    __syncthreads();

    if ((blockSize >= 16) && (tid < 8)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 8]);
    }
    __syncthreads();

    if ((blockSize >= 8) && (tid < 4)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 4]);
    }
    __syncthreads();

    if ((blockSize >= 4) && (tid < 2)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 2]);
    }
    __syncthreads();

    if ((blockSize >= 2) && (tid < 1)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 1]);
    }
    __syncthreads();

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

template<typename T, typename F, size_t blockSize, bool nIsPow2>
__global__ void msl::detail::foldColsKernel(T *g_idata, T *g_odata, size_t n,
                                            F func) {
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    size_t tid = threadIdx.x;
    size_t i = threadIdx.x * gridDim.x + blockIdx.x;
    size_t gridSize = blockSize * gridDim.x;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim). More blocks will result
    // in a larger gridSize and therefore fewer elements per thread.
    // R result = g_idata[i];
    T result = g_idata[i];
    i += gridSize;

    while (i < n) {
        result = func(result, g_idata[i]);
        i += gridSize;
    }
    sdata[tid] = result;
    __syncthreads();

    // perform reduction in shared memory
    if ((blockSize >= 1024) && (tid < 512)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 512]);
    }
    __syncthreads();

    if ((blockSize >= 512) && (tid < 256)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 256]);
    }
    __syncthreads();

    if ((blockSize >= 256) && (tid < 128)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 128]);
    }
    __syncthreads();

    if ((blockSize >= 128) && (tid < 64)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 64]);
    }
    __syncthreads();

    if ((blockSize >= 64) && (tid < 32)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 32]);
    }
    __syncthreads();

    if ((blockSize >= 32) && (tid < 16)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 16]);
    }
    __syncthreads();

    if ((blockSize >= 16) && (tid < 8)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 8]);
    }
    __syncthreads();

    if ((blockSize >= 8) && (tid < 4)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 4]);
    }
    __syncthreads();

    if ((blockSize >= 4) && (tid < 2)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 2]);
    }
    __syncthreads();

    if ((blockSize >= 2) && (tid < 1)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 1]);
    }
    __syncthreads();

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}


// size_t n is execPlan.mLocal
template<typename T, typename F, size_t blockSize, bool nIsPow2>
__global__ void msl::detail::foldRowsKernel(T *g_idata, T *g_odata, size_t n, F func) {

    T *sdata = SharedMemory<T>();
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * n + tid;

     size_t loop_end_index = blockIdx.x * n + n;

     // we reduce multiple elements per thread.  The number is determined by the
     // number of active thread blocks (via gridDim). More blocks will result
     // in a larger gridSize and therefore fewer elements per thread.

     T result = g_idata[i];

     i += blockSize;

     while (i < loop_end_index) {
         result = func(result, g_idata[i]);
         i += blockSize;
     }
    sdata[tid] = result;
    __syncthreads();

    // perform reduction in shared memory
    if ((blockSize >= 1024) && (tid < 512)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 512]);
    }
    __syncthreads();

    if ((blockSize >= 512) && (tid < 256)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 256]);
    }
    __syncthreads();

    if ((blockSize >= 256) && (tid < 128)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 128]);
    }
    __syncthreads();

    if ((blockSize >= 128) && (tid < 64)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 64]);
    }
    __syncthreads();

    if ((blockSize >= 64) && (tid < 32)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 32]);
    }
    __syncthreads();

    if ((blockSize >= 32) && (tid < 16)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 16]);
    }
    __syncthreads();

    if ((blockSize >= 16) && (tid < 8)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 8]);
    }
    __syncthreads();

    if ((blockSize >= 8) && (tid < 4)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 4]);
    }
    __syncthreads();

    if ((blockSize >= 4) && (tid < 2)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 2]);
    }
    __syncthreads();

    if ((blockSize >= 2) && (tid < 1)) {
        sdata[tid] = func(sdata[tid], sdata[tid + 1]);
    }
    __syncthreads();

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

template<typename T, typename F>
void msl::detail::reduce(unsigned int size, T *d_idata, T *d_odata, int threads,
                         int blocks, F &f, cudaStream_t &stream, int gpu) {
    cudaSetDevice(gpu);
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize =
            (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
    if (isPow2(size)) {
        switch (threads) {
            case 1024:
                foldKernel<T, F, 1024, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 512:
                foldKernel<T, F, 512, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 256:
                foldKernel<T, F, 256, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 128:
                foldKernel<T, F, 128, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 64:
                foldKernel<T, F, 64, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 32:
                foldKernel<T, F, 32, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 16:
                foldKernel<T, F, 16, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 8:
                foldKernel<T, F, 8, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 4:
                foldKernel<T, F, 4, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 2:
                foldKernel<T, F, 2, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 1:
                foldKernel<T, F, 1, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
        }
    } else {
        switch (threads) {
            case 1024:
                foldKernel<T, F, 1024, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 512:
                foldKernel<T, F, 512, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 256:
                foldKernel<T, F, 256, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 128:
                foldKernel<T, F, 128, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 64:
                foldKernel<T, F, 64, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 32:
                foldKernel<T, F, 32, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 16:
                foldKernel<T, F, 16, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 8:
                foldKernel<T, F, 8, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 4:
                foldKernel<T, F, 4, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 2:
                foldKernel<T, F, 2, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 1:
                foldKernel<T, F, 1, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
        }
    }
}

template<typename T, typename F>
void msl::detail::foldCols(unsigned int size, T *d_idata, T *d_odata, int threads,
                           int blocks, F &f, cudaStream_t &stream, int gpu) {
    cudaSetDevice(gpu);
    dim3 dimBlock(threads);
    dim3 dimGrid(blocks);
    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
    //detail::printGPU<<<1, 1>>>(d_idata, 32, 256);

    if (isPow2(size)) {
        switch (threads) {
            case 1024:
                foldColsKernel<T, F, 1024, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 512:
                foldColsKernel<T, F, 512, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 256:
                foldColsKernel<T, F, 256, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 128:
                foldColsKernel<T, F, 128, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 64:
                foldColsKernel<T, F, 64, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 32:
                foldColsKernel<T, F, 32, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 16:
                foldColsKernel<T, F, 16, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 8:
                foldColsKernel<T, F, 8, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 4:
                foldColsKernel<T, F, 4, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 2:
                foldColsKernel<T, F, 2, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 1:
                foldColsKernel<T, F, 1, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
        }
    } else {
        switch (threads) {
            case 1024:
                foldColsKernel<T, F, 1024, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 512:
                foldColsKernel<T, F, 512, false>  <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 256:
                foldColsKernel<T, F, 256, false>  <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 128:
                foldColsKernel<T, F, 128, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 64:
                foldColsKernel<T, F, 64, false>  <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 32:
                foldColsKernel<T, F, 32, false>  <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 16:
                foldColsKernel<T, F, 16, false>  <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 8:
                foldColsKernel<T, F, 8, false>  <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 4:
                foldColsKernel<T, F, 4, false>  <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 2:
                foldColsKernel<T, F, 2, false>  <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 1:
                foldColsKernel<T, F, 1, false>  <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
        }
    }
    //detail::printGPU<<<1, 1>>>(d_odata, size, 256);
}

template<typename T, typename F>
void msl::detail::foldRows(unsigned int size, T *d_idata, T *d_odata, int threads,
                           int blocks, F &f, cudaStream_t &stream, int gpu) {
    cudaSetDevice(gpu);
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
    if (isPow2(size)) {
        switch (threads) {
            case 1024:
                foldRowsKernel<T, F, 1024, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 512:
                foldRowsKernel<T, F, 512, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 256:
                foldRowsKernel<T, F, 256, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 128:
                foldRowsKernel<T, F, 128, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 64:
                foldRowsKernel<T, F, 64, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 32:
                foldRowsKernel<T, F, 32, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 16:
                foldRowsKernel<T, F, 16, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 8:
                foldRowsKernel<T, F, 8, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 4:
                foldRowsKernel<T, F, 4, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 2:
                foldRowsKernel<T, F, 2, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 1:
                foldRowsKernel<T, F, 1, true> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
        }
    } else {
        switch (threads) {
            case 1024:
                foldRowsKernel<T, F, 1024, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 512:
                foldRowsKernel<T, F, 512, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 256:
                foldRowsKernel<T, F, 256, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 128:
                foldRowsKernel<T, F, 128, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 64:
                foldRowsKernel<T, F, 64, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 32:
                foldRowsKernel<T, F, 32, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 16:
                foldRowsKernel<T, F, 16, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 8:
                foldRowsKernel<T, F, 8, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 4:
                foldRowsKernel<T, F, 4, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 2:
                foldRowsKernel<T, F, 2, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
            case 1:
                foldRowsKernel<T, F, 1, false> <<<dimGrid, dimBlock, smemSize, stream>>>(
                        d_idata, d_odata, size, f);
                break;
        }
    }

}
