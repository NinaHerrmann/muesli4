/*
 * zip_kernels.cu
 *
 *      Authors: Steffen Ernsting <s.ernsting@uni-muenster.de>
 *               Herbert Kuchen <kuchen@uni-muenster.de.
 *               Nina Herrmann <nina.herrmann@uni-muenster.de.
 * 
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2020 Steffen Ernsting <s.ernsting@uni-muenster.de>,
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
#include "../include/detail/zip_kernels.cuh"


template <typename T1, typename T2, typename R, typename FCT2>
__global__ void msl::detail::zipKernel(T1* in1, T2* in2, R* out, size_t n, FCT2 func){
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n) {
    out[x] = func(in1[x], in2[x]);
  }
}
template <typename T1, typename T2, typename R, typename FCT4>
__global__ void msl::detail::zip3DKernel(T1* in1, T2* in2, R* out, FCT4 func, int gpuDepth, int gpuRow, int gpuCol){
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int localoverall = (z * (gpuRow*gpuCol)) + (y * gpuCol) + x;
    if (z < gpuDepth && y < gpuRow && x < gpuCol) {
        out[localoverall] = func(in1[localoverall], in2[localoverall]);
    }
}
// new kernel for zip(InPlace)3, HK 19.11.2020
template <typename T1, typename T2, typename T3, typename R, typename FCT3>
__global__ void msl::detail::zipKernel(T1* in1, T2* in2, T3* in3, R* out, size_t n, FCT3 func){
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n) {
    out[x] = func(in1[x], in2[x], in3[x]);
  }
}


// new kernel for DM, HK 06.11.2020 -- TODO better to start matrix of threads?
template <typename T1, typename T2, typename R, typename FCT3>
__global__ void msl::detail::zipIndexKernelDM(T1* in1, T2* in2, R* out, size_t n, int first, FCT3 func, int nCols){
  size_t k = blockIdx.x * blockDim.x + threadIdx.x;

  if (k < n) {
      out[k] = func((k + first) / nCols, (k + first) % nCols, in1[k], in2[k]);
  }
}

template <typename T1, typename T2, typename T3, typename T4, typename R, typename FCT3>
__global__ void msl::detail::zipKernelAAM(T1* in1, T2* in2, T3* in3, T4*in4,
                                          R* out, size_t n, int first, int first2, FCT3 func, int nCols){
  size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = ((k + first) / nCols) - first2;
  if (k < n) {
    out[k] = func(in1[k], in2[i], in3[i], in4[k]);
  }
}

template <typename T1, typename T2, typename R, typename FCT3>
__global__ void msl::detail::zipIndexKernelDA(T1* in1, T2* in2, R* out, size_t n,
                                            int first, FCT3 func){
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n) {
      out[x] = func(x+first, in1[x], in2[x]);
  }
}


template <typename T1, typename T2, typename R, typename FCT4>
__global__ void msl::detail::zipIndexKernelDC(T1* in1, T2* in2, R* out, FCT4 func, int gpuRows, int gpuCols, int gpuDepth,
                                            int firstRow, int firstCol, int firstDepth){
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z * blockDim.z + threadIdx.z;


    // TODO compare map less params
    int localoverall = (z * (gpuRows*gpuCols)) + (y * gpuCols) + x;
    if (z < gpuDepth && y < gpuRows && x < gpuCols) {
        out[localoverall] = func(y + firstRow, x + firstCol, z + firstDepth,
                                        in1[localoverall], in2[localoverall]);
    }
}

template <typename T1, typename T2, typename R, typename FCT3>
__global__ void msl::detail::crossZipIndexKernel(T1* in1,T2* in2, R* out, size_t n,int first,FCT3 func,int nCols){
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        out[k] = func((k + first) / nCols,(k + first) % nCols, in1, in2);
    }
}

template <typename T1, typename T2, typename FCT3>
__global__ void msl::detail::crossZipInPlaceIndexKernel(T1* in1, T2* in2, size_t n, int first, FCT3 func, int nCols){
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;

    func((k + first) / nCols,(k + first) % nCols, in1, in2);

}




