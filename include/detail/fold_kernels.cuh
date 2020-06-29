/*
 * fold_kernels.h
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

#include "shared_mem.h"

namespace msl {

namespace detail {

bool isPow2(size_t x);

size_t nextPow2(size_t x);

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks,
                            int &threads);

template<typename T, typename F, size_t blockSize, bool nIsPow2>
__global__ void foldKernel(T *g_idata, T *g_odata, size_t n, F func);

template<typename T, typename F, size_t blockSize, bool nIsPow2>
__global__ void foldColsKernel(T *g_idata, T *g_odata, size_t n, F func);

template<typename T, typename F, size_t blockSize, bool nIsPow2>
__global__ void foldRowsKernel(T *g_idata, T *g_odata, size_t n, F func);

template<typename T, typename F>
void reduce(uint size, T* d_idata, T* d_odata, int threads, int blocks, F& f,
            cudaStream_t& stream, int gpu);

template<typename T, typename F>
void foldCols(uint size, T* d_idata, T* d_odata, int threads, int blocks, F& f,
              cudaStream_t& stream, int gpu);

template<typename T, typename F>
void foldRows(uint size, T* d_idata, T* d_odata, int threads, int blocks, F& f,
              cudaStream_t& stream, int gpu);

}
}

#include "../../src/fold_kernels.cu"

