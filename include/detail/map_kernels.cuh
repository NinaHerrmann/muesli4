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
#include "plmatrix.h"
#include "simpleplmatrix.h"
// #include "plarray.h"

namespace msl {

namespace detail {

template <typename T, typename R, typename F>
__global__ void mapKernel(T *in, R *out, size_t size, F func);

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

template <typename T, typename R, typename F, typename NeutralValueFunctor>
__global__ void mapStencilKernel(R *out, GPUExecutionPlan<T> plan,
                                 PLMatrix<T> *input,
                                 F func, int tile_width, int tile_height, NeutralValueFunctor nv);
template <typename T, typename R, typename F, typename NeutralValueFunctor>
__global__ void mapSimpleStencilKernel(R *out, GPUExecutionPlan<T> plan, SimplePLMatrix<T> *input,
                                 F func, int tile_width, int tile_height, NeutralValueFunctor nv);
template <typename T> __global__ void printFromGPU(T *A);
// template <typename T, typename R, typename F>
//__global__ void mapStencilKernel(T* in,
//                                 R* out,
//                                 GPUExecutionPlan<T> plan,
//                                 PLArray<T>* input,
//                                 F func,
//                                 int tile_width);

} // namespace detail
} // namespace msl

#include "../../src/map_kernels.cu"
