/*
 * map_kernels.h
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

#pragma once

#include "exec_plan.h"
// #include "plmatrix.h"
// #include "da.h"
#include "dc.h"
// #include "plarray.h"

namespace msl {

namespace detail {

template <typename T, typename R, typename F>
__global__ void mapKernel(T *in, R *out, size_t size, F func);

template <typename T, typename R, typename F>
__global__ void mapKernel(T *in, R *out, size_t size, F func, GPUExecutionPlan<T> plan, int nrow, int ncol);


template <typename T, typename R, typename F>
__global__ void mapIndexKernel(T *in, R *out, size_t size, size_t first, F func,
                               bool localIndices);

// new kernel for DM, HK 06.11.2020
template <typename T, typename R, typename F>
__global__ void mapIndexKernel(T *in, R *out, size_t size, size_t first, F func,
                               int ncols);

template <typename T, typename R, typename F>
__global__ void mapIndexKernel(T *in, R *out, GPUExecutionPlan<T> plan, F func,
                               bool localIndices);


template <typename T, typename R, typename F>
__global__ void mapIndexKernel(T *in, R *out, GPUExecutionPlan<T> plan, F func,
                               bool localIndices, int nrow, int ncol, bool dim3);

template <typename T, typename R, typename F>
__global__ void mapPlaceKernel(T *in, R *out, GPUExecutionPlan<T> plan, F func,
                               bool localIndices, int nrow, int ncol, bool dim3, bool inpl);
template <typename T, typename R, typename F>
__global__ void mapIndexInPlaceKernel(T *in, R *out, GPUExecutionPlan<T> plan, F func,
                               bool localIndices);
/*
template <typename T, typename R, typename F, typename NeutralValueFunctor>
__global__ void mapStencilKernel(R *out, GPUExecutionPlan<T> plan,
                                 PLMatrix<T> *input,
                                 F func, int tile_width, int tile_height, NeutralValueFunctor nv);
template <typename T, typename R, typename F>
__global__ void mapStencilMMKernel(R *out,int gpuRows, int gpuCols, int firstCol, int firstRow,  PLMatrix<T> *dm, T * current_data,
                                 F func, int tile_width, int reps, int kw);
template <typename T, typename R, typename F>
__global__ void mapStencilGlobalMem(R *out, GPUExecutionPlan<T> plan, PLMatrix<T> *dm,
                                 F func, int i);
template <typename T, typename R, typename F>
__global__ void mapStencilGlobalMem_rep(R *out, GPUExecutionPlan<T> plan, PLMatrix<T> *dm,
                                        F func, int i, int rep, int tile_width);
template <typename T> __global__ void printFromGPU(T *A, int size, int breaker);
template <typename T> __global__ void printStructFromGPU(T *A, int size, int breaker);
template <typename T> __global__ void printsingleGPUElement(T *A, int index);
template <typename T> __global__ void fillsides(T *A, int paddingoffset, int gpuRows, int ss);
template <typename T> __global__ void fillcore(T *destination, T *source, int paddingoffset, int gpuCols, int ss);*/
__global__ void teststh(int Size, int t);


} // namespace detail
} // namespace msl

#include "../../src/map_kernels.cu"
