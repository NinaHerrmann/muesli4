
/*
 * map_kernels.cpp
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

template <typename T, typename R, typename F>
__global__ void msl::detail::mapKernel(T *in, R *out, size_t size, F func) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < size) {
    out[x] = func(in[x]);
    //    printf("debug GPU: x: %i, in[x]: %i, out[x]: %i\n",x,in[x],out[x]);
  }
}

// new kernel for distributed matrices (DM)
template <typename T, typename R, typename F>
__global__ void msl::detail::mapIndexKernel(T *in, R *out, size_t size,
                                            size_t first, F func, int ncols) {
  size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = (k + first) / ncols;
  int j = (k + first) % ncols;
  if (k < size) {
    out[k] = func(i, j, in[k]);
  }
}

template <typename T, typename R, typename F>
__global__ void msl::detail::mapIndexKernel(T *in, R *out, size_t size,
                                            size_t first, F func,
                                            bool localIndices) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;

  size_t indexOffset = localIndices ? 0 : first;

  if (x < size) {
    out[x] = func(x + indexOffset, in[x]);
  }
}

template <typename T, typename R, typename F>
__global__ void msl::detail::mapIndexKernel(T *in, R *out,
                                            GPUExecutionPlan<T> plan, F func,
                                            bool localIndices) {
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;

  size_t rowOffset = localIndices ? 0 : plan.firstRow;
  size_t colOffset = localIndices ? 0 : plan.firstCol;

  if (y < plan.nLocal) {
    if (x < plan.mLocal) {
      out[y * plan.mLocal + x] =
          func(y + rowOffset, x + colOffset, in[y * plan.mLocal + x]);
    }
  }
}

template <typename T, typename R, typename F, typename NeutralValueFunctor>
__global__ void
msl::detail::mapStencilKernel(R *out, GPUExecutionPlan<T> plan,
                              PLMatrix<T> *input, F func,
                              int tile_width, int tile_height, NeutralValueFunctor nv) {

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  input->readToSharedMem(y + plan.firstRow, x, tile_width, tile_height,
                         plan.gpuRows, plan.gpuCols);
  if (y < plan.gpuRows) {
    if (x < plan.gpuCols) {

      if (!((y == 0 && x < plan.firstCol) ||
            (y == (plan.gpuRows - 1) && x > plan.lastCol))) {
        out[y * plan.gpuCols + x - plan.firstCol] =
            func(y + plan.firstRow, x, *input);
      }
    }
  }
}
template <typename T, typename R, typename F, typename NeutralValueFunctor>
__global__ void
msl::detail::mapStencilMMKernel(R *out, GPUExecutionPlan<T> plan,
                                T *inputdm, T *inputpadding, F func,
                              int tile_width, int tile_height, NeutralValueFunctor nv) {

    //int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int col = x % plan.gpuCols;
    int row = x / plan.gpuCols;
    //  input->readToSharedMem(y + plan.firstRow, x, tile_width, tile_height,
//                         plan.gpuRows, plan.gpuCols);

    extern __shared__ float smem[];
    // 512 per block
    // approximately 16.000 floats per SM Palma
    // printf("Thread y: %d, x: %d. GPU data size %d x %d\n", abs_ty, abs_tx,
    //        gpu_rows, gpu_columns);
   if (row < plan.gpuRows) {
        if (col < plan.gpuCols) {
            int localcol = threadIdx.x % plan.gpuCols;
            int localrow = threadIdx.x / plan.gpuCols;

            int index = localrow * plan.gpuCols + localcol;
            smem[index] = inputdm[index];
            if (threadIdx.x == 1 || threadIdx.x == 4){printf("col %d row %d localcol %d localrow %d %d, %d, %d, %d\n",col, row, localcol, localrow, blockIdx.x, blockDim.x, threadIdx.x, index);}

        }
    }
    __syncthreads();
    if (row < plan.gpuRows) {
        if (col < plan.gpuCols) {
            //printf("%d, %d , %d writeto %d -- \n", row,col,x, row * (plan.gpuCols) + col);
            out[row * plan.gpuCols + col] = func(row, col, inputdm, plan.gpuCols, plan.gpuRows, inputpadding);
        }
    }

}
template <typename T> __global__ void msl::detail::printFromGPU(T *A, int size) {
  for (int i = 0; i < size; i++) {
      printf("[%.1f];", A[i]);
  }
}