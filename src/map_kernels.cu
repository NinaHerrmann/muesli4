
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
      // TODO for array only two arguments
    out[k] = func(i, j, in[k]);
  }
}

/*
template <typename T, typename R, typename F>
__global__ void msl::detail::mapIndexKernel(T *in, R *out, size_t size,
                                            size_t first, F func,
                                            bool localIndices) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;

  size_t indexOffset = localIndices ? 0 : first;

  if (x < size) {
    out[x] = func(x + indexOffset, in[x]);
  }
}*/

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

template <typename T, typename R, typename F>
__global__ void
msl::detail::mapStencilGlobalMem(R *out, GPUExecutionPlan<T> plan, PLMatrix<T> *dm, F func, int i) {

    size_t thread = threadIdx.x + blockIdx.x * blockDim.x;

    int y = thread /plan.gpuCols;
    int x = thread % plan.gpuCols;

    dm->readToGlobalMemory();

    if ((y) < plan.gpuRows) {
        if (x < plan.gpuCols) {
            out[thread] = func(y, x, dm, plan.gpuCols, plan.gpuRows);
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
template <typename T, typename R, typename F>
__global__ void
msl::detail::mapStencilMMKernel(R *out, GPUExecutionPlan<T> plan, PLMatrix<T> *pl,
                                F func, int tile_width, int reps) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // wie viele bloecke hatten wir schon? ((blockIdx.x * blockDim.x + threadIdx.x) / reps)
    int global_row = (reps*tile_width * (x / tile_width)) + (threadIdx.x%tile_width);
    // TODO
    /*if (x == 17 & y == 0) {
        printf("\nRows %d;%d + %d * %d\n", global_row, x, (x / tile_width) , reps*tile_width);
    }*/
    if (global_row < plan.gpuRows && y < plan.gpuCols) {
        pl->readToSM(global_row, y+plan.firstCol, reps);

        for (int j = 0; j < reps; j++) {
            if (global_row == 15 & y == 0 & j == 1) {
                pl->printSM(400);
                printf("\n fiiirst %d: %d:%d\n",j,global_row + plan.firstRow + (tile_width*j), y + plan.firstCol);}
            out[(global_row + (j * tile_width)) * plan.gpuCols + y] = func(global_row + plan.firstRow + (tile_width*j), y + plan.firstCol, pl, plan.gpuCols, plan.gpuRows);
        }
    }
}
template <typename T> __global__ void msl::detail::printFromGPU(T *A, int size, int breaker) {
  for (int i = 0; i < size; i++) {
      if (i%breaker==0){ printf("\n");}
      //printf("%d;", A[i] ? "true" : "false");
      printf("%d;", A[i]);
  }
}
template <typename T> __global__ void msl::detail::printStructFromGPU(T *A, int size, int breaker) {
  for (int i = 0; i < size; i++) {
      if (i%breaker==0){ printf("\n");}
      printf("%d %d;", A[i].starting, A[i].no_of_edges);
  }
}
template <typename T> __global__ void msl::detail::printsingleGPUElement(T *A, int index) {
      printf("%d:%d;\n", index, A[index]);
}
__global__ void msl::detail::teststh(int Size, int t) {
    if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
    printf(" write to %d t: %d Size: %d\n", Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t, t, Size);
}
