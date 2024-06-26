#include "../include/detail/map_kernels.cuh"
#include "../include/plmatrix.h"

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

template<typename T, typename R, typename F>
__global__ void msl::detail::mapKernel(T *in, R *out, size_t size, F func) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < size) {
        out[x] = func(in[x]);
    }
}

template<typename T, typename R, typename F>
__global__ void
msl::detail::mapKernel3D(T *in, R *out, F func, int gpuRows, int gpuCols, int gpuDepth) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    // TODO incomplete rows?

    int localoverall = (z * (gpuRows * gpuCols)) + (y * gpuCols) + x;
    if (z < gpuDepth & y < gpuRows & x < gpuCols) {
        out[localoverall] = func(in[localoverall]);
    }
}


template<typename T, typename R, typename F>
__global__ void msl::detail::mapIndexKernelDM(T *in, R *out, size_t size,
                                              size_t first, F func, int nCols) {
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = (k + first) / nCols;
    int j = (k + first) % nCols;
    if (k < size-1) {
        //out[k] = in[k];
        out[k] = func(i, j, in[k]);
    }
}

template<typename T, typename R, typename F>
__global__ void msl::detail::mapIndexKernelDA(T *in, R *out, size_t size, size_t first, F func) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size) {
        out[x] = func(x + first, in[x]);
    }
}
template<typename T, typename T2, typename R, typename F>
__global__ void msl::detail::mapIndexDMKernelDA(T *in, T2* in2, R *out, size_t size, size_t first, int dmcols, F func) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size) {
        T2 * in2offset = &in2[x * dmcols];
        out[x] = func(x + first, in2offset, in[x]);
    }
}

template<typename T, typename R, typename F>
__global__ void msl::detail::mapIndexKernelDC(T *in, R *out, int gpuRows, int gpuCols, int offset, int elements, F func) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int elementlayer = gpuRows * gpuCols;
    int l = (x + offset) / (elementlayer);
    int remaining_index = (x + offset) % (elementlayer);
    int j = remaining_index / gpuCols;
    int i = remaining_index % gpuCols;
    // calculate
    if (x < elements) {
        out[x] = func(i, j, l, in[x]);
    }

}
template<typename T, typename F>
__global__ void msl::detail::mapInPlaceKernelDC(T *inout, int gpuRows, int gpuCols,
                                                int gpuDepth, F func) {
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    int localoverall = (z * (gpuRows * gpuCols)) + (y * gpuCols) + x;
    if (z < gpuDepth && y < gpuRows && x < gpuCols) {
        inout[localoverall] = func(inout[localoverall]);
    }
}


template<typename T, typename R, typename F>
__global__ void
msl::detail::mapStencilGlobalMem(R *out, GPUExecutionPlan<T> plan, PLMatrix <T> *pl, F func, int i) {

    size_t thread = threadIdx.x + blockIdx.x * blockDim.x;
    int y = thread / plan.gpuCols;
    int x = thread % plan.gpuCols;
    pl->readToGlobalMemory();
    if ((y) < plan.gpuRows) {
        if (x < plan.gpuCols) {
            out[thread] = func(y, x, pl, plan.gpuCols, plan.gpuRows);
        }
    }
}

template<typename T, typename R, typename F>
__global__ void
msl::detail::mapStencilGlobalMem_rep(R *out, GPUExecutionPlan<T> plan, PLMatrix <T> *pl, F func, int reps,
                                     int tile_width) {

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_row = (reps * tile_width * (x / tile_width)) + (threadIdx.x);
    pl->readToGlobalMemory();
    for (int j = 0; j < reps; j++) {
        if (global_row + (j * tile_width) < plan.gpuRows && y < plan.gpuCols) {
            out[(global_row + (j * tile_width)) * plan.gpuCols + y] = func(
                    global_row + plan.firstRow + (tile_width * j),
                    y + plan.firstCol, pl, plan.gpuCols, plan.gpuRows);
        }
    }
}

template<typename T, msl::NPLMMapStencilFunctor<T> f>
__global__ void
msl::detail::mapStencilKernelDMSM(T *out, NPLMatrix<T> in, unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }
    int2 coords = in.indexToCoordinate(in.dataStartIndex + i);
    in.readTosm(in.dataStartIndex + i, threadIdx.x, 1);
    __syncthreads();
    T v = f(in, coords.x, coords.y);
    out[i] = v;
}
template<typename T, typename R, typename F>
__global__ void
msl::detail::mapStencilMMKernel(R *out, int gpuRows, int gpuCols, int firstCol, int firstRow, PLMatrix <T> *pl,
                                T *current_data,
                                F func, int tile_width, int reps, int kw) {
    extern __shared__ int s[];

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    // wie viele bloecke hatten wir schon? ((blockIdx.x * blockDim.x + threadIdx.x) / reps)
    int global_row = (reps * tile_width * (x / tile_width)) + (threadIdx.x);
    /*int g_row = (reps*tile_width * (x / tile_width)) + (threadIdx.x%tile_width);
        int global_col = blockIdx.y * blockDim.y + threadIdx.y;
        int new_tile_width = tile_width + kw;

        int inside_elements = tile_width * tile_width;
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
        __syncthreads();*/
    for (int j = 0; j < reps; j++) {
        if (global_row + (j * tile_width) < gpuRows && y < gpuCols) {
            out[(global_row + (j * tile_width)) * gpuCols + y] = func(global_row + firstRow + (tile_width * j),
                                                                      y + firstCol, pl, 0, 0);
        }
    }
}

template<typename T>
__global__ void msl::detail::fillsides(T *A, int paddingoffset, int gpuCols, int ss, T neutral_value, int coloffset) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < coloffset) {
        for (int i = 0; i < ss; i++) {
            A[(paddingoffset + (x * gpuCols)) + i] = neutral_value;
            A[(paddingoffset + (x * gpuCols)) - i + 1] = neutral_value;
        }
    }
}

template<typename T>
__global__ void
msl::detail::fillcore(T *destination, T *source, int paddingoffset, int gpuCols, int ss, int rows, int cols) {
    // Does not fill the sides correctly.
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x * y < ss + rows * cols && y < (gpuCols + ss-1)) {
        destination[paddingoffset + ss + (x * (gpuCols + (2 * ss)) + y)] = source[(x * gpuCols) + y];
    }

}
template<typename T>
__global__ void
msl::detail::setColumn(T *wdata, T* rdata, int totalrows, int totalcolumns, int col) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < totalrows) {
        wdata[x*totalcolumns + col] = rdata[x];
    }
}
template<typename T>
__global__ void
msl::detail::printGPU(T *data, int size, int col) {
    for (int i = 0; i < size; i++) {
        if (i % col == 0 && i != 0) {printf("\n");}
        printf("%f;", data[i]);
    }
}
template<typename T>
__global__ void printNPL(T *data, int size, int col) {
    for (int i = 0; i < size; i++) {
        if (i % col == 0 && i != 0) {printf("\n");}
        printf("%.2f;\t", data[i]);
    }
}

__global__ void msl::detail::generateRandom(curandState* state, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

