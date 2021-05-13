
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

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ T inputsm[];


    // 512 threads per block --> assuming all in one line we need 512*3+2 numbers in shared memory = 1538 nicht so gut für die wiederverwendung
    // einen "Block" abarbeiten z.B. 16*16 = 256 Elemente 64 surrounding weil block 
    // approximately 16.000 floats per SM Palma
    // 0  1   2  3  4   5  6  7
    // 8  9  10 11 12 13 14 15
    // 16 17 18 19 20 21 22 23
    // 24 25 26 27 28 29 30 31
    // 32 33 34 35 36 37 38 39
    // 40 41 42 43 44 45 46 47
    // 48 49 50 51 52 53 54 55
    // 56 57 58 59 60 61 62 63
    // 0  1   2  3  4   5  6  7  8  9
    // 10 11 12 13 14  15 16 17 18 19
    // ...
    int localcol = (x - (blockIdx.x * tile_height));
    int localrow = y - (blockIdx.y * tile_width);
    int localindex = localrow * tile_width + localcol;
    int realdataoffset = tile_width + 3 + (localrow *  2);
    int secondrealdataoffset = y * plan.gpuCols + x - plan.firstCol;
    // Copy all the "real data"

    inputsm[localindex + realdataoffset] = __ldg(&inputdm[secondrealdataoffset]);

    // Copy borders
    int modulo = localindex % tile_width;

    int inputoffset = modulo + blockIdx.x * tile_width;
    if (blockIdx.y != 0) {
        // In case we are not the first block vertical copy top from previous gpu
        inputsm[modulo+1] =__ldg(&inputdm[((blockIdx.y) * (plan.gpuCols)) * tile_height - plan.gpuCols +
                inputoffset]);
    } else {
        // In case we are in the end blockIdx.y == 0 we need to copy the top from padding_stencil
        inputsm[modulo+1] = __ldg(&inputpadding[inputoffset]);
    }

    int bottomoffset = (tile_width + 2)*(tile_height +1)+1 + modulo;
    // Top done ... continue with bottom
    if (blockIdx.y == ((plan.gpuRows / tile_height)-1)){
        inputsm[bottomoffset] = __ldg(&inputpadding[inputoffset + plan.gpuCols]);
    } else {
        // In case it is not the last tile we need to copy bottom from top of other tile
        inputsm[bottomoffset] = __ldg(&inputdm[(blockIdx.y + 1) * (plan.gpuCols) * tile_height + inputoffset]);
    }

    if (localindex < tile_height) {
        int inputoffset = (blockIdx.y) * (plan.gpuCols) * tile_height + (plan.gpuCols * localindex);
        // Same for left and right, if blockidX.x is != 0 we need to copy from left
        // If blockidx is not the last block we need to copy from right
        int rightoffset = tile_width + 2 + localindex * (tile_width+2);
        if (blockIdx.x != 0) {
            inputsm[rightoffset] = __ldg(&inputdm[inputoffset +
                                         (blockIdx.x) * tile_width - 1]);
        } else {
            // In case we are in the ad blockIdx.y == 0 we need to copy from padding_stencil
            inputsm[rightoffset] = 100;
        }
        int leftoffset= tile_width+1 + (tile_width + 2) + localindex * (tile_width+2);
        // In case we are the last tile we need to copy from the stencil otherwise we copy from the other gpu
        if (blockIdx.x == ((plan.gpuCols / tile_width)-1)){
             inputsm[leftoffset] = 100;
        } else {
            // In case it is not the last tile we need to copy right hand side from other tile
            inputsm[leftoffset] =__ldg(&inputdm[inputoffset + (blockIdx.x +1) * tile_width]);
        }
    }
    __syncthreads();
    if (localrow < tile_height) {
        if (localcol < tile_width) {
            out[y * plan.gpuCols + x - plan.firstCol] = func(localrow, localcol, inputsm, tile_width, tile_height);
        }
    }

}
template <typename T> __global__ void msl::detail::printFromGPU(T *A, int size) {
  for (int i = 0; i < size; i++) {
      printf("[%.1f];", A[i]);
  }
}
