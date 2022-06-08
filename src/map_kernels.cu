
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
    }
}
template <typename T, typename R, typename F>
__global__ void msl::detail::mapKernel(T *in, R *out, size_t size, F func, GPUExecutionPlan<T> plan, int nrow, int ncol) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int localoverall = (z * (nrow*ncol)) + (y * ncol) + x;
  if (z < plan.gpuDepth) {
      if (y < plan.gpuRows) {
          if (x < plan.gpuCols) {

              out[localoverall] = func(in[localoverall]);
          }
      }
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

template <typename T, typename R, typename F>
__global__ void msl::detail::mapIndexKernel(T *in, R *out,
                                            GPUExecutionPlan<T> plan, F func,
                                            bool localIndices, int nrow, int ncol, bool dim3) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  int rowOffset = localIndices ? 0 : plan.firstRow;
  int colOffset = localIndices ? 0 : plan.firstCol;
  int depthOffset = localIndices ? 0 : plan.firstDepth;

  //int overall = ((z+depthOffset) * (nrow*ncol)) + (y * ncol) + x;
  int localoverall = (z * (nrow*ncol)) + (y * ncol) + x;
  if (z < plan.gpuDepth) {
      if (y < plan.gpuRows) {
          if (x < plan.gpuCols) {
              out[localoverall] = func(y + rowOffset, x + colOffset, z + depthOffset, in[localoverall]);
          }
      }
  }

}
template <typename T, typename R, typename F>
__global__ void msl::detail::mapPlaceKernel(T *in, R *out,
                                            GPUExecutionPlan<T> plan, F func,
                                            bool localIndices, int nrow, int ncol, bool dim3, bool inpl) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    //int overall = ((z+depthOffset) * (nrow*ncol)) + (y * ncol) + x;
    int localoverall = (z * (nrow*ncol)) + (y * ncol) + x;
    if (z < plan.gpuDepth) {
        if (y < plan.gpuRows) {
            if (x < plan.gpuCols) {
                out[localoverall] = func(in[localoverall]);
            }
        }
    }

}
/*
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
template <typename T, typename R, typename F>
__global__ void
msl::detail::mapStencilGlobalMem_rep(R *out, GPUExecutionPlan<T> plan, PLMatrix<T> *dm, F func, int i, int reps, int tile_width) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_row = (reps*tile_width * (x / tile_width)) + (threadIdx.x);
    __shared__ int *s;
    dm->readToGlobalMemory();
    for (int j = 0; j < reps; j++) {
        if (global_row + (j * tile_width) < plan.gpuRows && y < plan.gpuCols) {
            out[(global_row + (j * tile_width)) * plan.gpuCols + y] = func(global_row + plan.firstRow + (tile_width*j), y + plan.firstCol, dm, s);
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
msl::detail::mapStencilMMKernel(R *out,int gpuRows, int gpuCols, int firstCol, int firstRow, PLMatrix<T> *pl, T * current_data,
                                F func, int tile_width, int reps, int kw) {
    extern __shared__ int s[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int new_tile_width = tile_width + kw;
    int inside_elements = tile_width * tile_width;
    // wie viele bloecke hatten wir schon? ((blockIdx.x * blockDim.x + threadIdx.x) / reps)
    int global_row = (reps*tile_width * (x / tile_width)) + (threadIdx.x);
    int g_row = (reps*tile_width * (x / tile_width)) + (threadIdx.x%tile_width);
    int global_col = blockIdx.y * blockDim.y + threadIdx.y;
    const int newsize = new_tile_width * ((reps*tile_width)+kw);
    const int iterations = (newsize/(inside_elements)) + 1;

    for (int rr = 0; rr <= iterations; ++rr) {
        int local_index = (rr * (inside_elements)) + (threadIdx.x) * tile_width + ( threadIdx.y);
        int row = local_index / new_tile_width;
        int firstcol = global_col -  threadIdx.y;
        int g_col = firstcol + ((local_index) % new_tile_width);
        int readfrom = (((g_row-threadIdx.x) + row) * (gpuCols+kw)) + g_col;
        if (local_index < newsize) {
            s[local_index] = current_data[readfrom];
        }
    }
    __syncthreads();
    for (int j = 0; j < reps; j++) {
        if (global_row + (j * tile_width) < gpuRows && y < gpuCols) {
            out[(global_row + (j * tile_width)) * gpuCols + y] = func(global_row + firstRow + (tile_width*j), y + firstCol, pl, s);
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
template <typename T>
__global__ void msl::detail::fillsides(T *A, int paddingoffset, int gpuCols, int ss) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i<ss; i++){
        A[(paddingoffset + (x * gpuCols))+i] = 0;
        A[(paddingoffset + (x * gpuCols))-i+1] = 0;
    }
}
template <typename T>
__global__ void msl::detail::fillcore(T *destination, T *source, int paddingoffset, int gpuCols, int ss) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // x is the row (ungaro4k smaller one)
    destination[paddingoffset + ss + (x * (gpuCols+(2*ss)) + y)] = source[(x * gpuCols) + y];

}*/
__global__ void msl::detail::teststh(int Size, int t) {
    if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
    printf(" write to %d t: %d Size: %d\n", Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t, t, Size);
}
