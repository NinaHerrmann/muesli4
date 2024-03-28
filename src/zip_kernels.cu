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


template<typename T1, typename T2, typename R, typename FCT2>
__global__ void msl::detail::zipKernel(const T1 *in1, const T2 *in2, R *out, size_t n, FCT2 func) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) {
        out[x] = func(in1[x], in2[x]);
    }
}

template<typename T1, typename T2, typename T3, typename R, typename FCT3>
__global__ void msl::detail::zipIndexKernelDMMA(const T1 *in0, const T2 *in1, const T3 *in2, R *out, size_t n,
                                                int first, FCT3 func, int nCols) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) {
        int row = x / nCols;
        out[x] = func((x + first) / nCols, (x + first) % nCols, in0[x], in1[x], in2[row]);
    }
}
template<typename T1, typename T3, typename R, typename FCT3>
__global__ void msl::detail::zipIndexKernelA(const T1 *in0, const T3 *in2, R *out, size_t n,
                                                int first, FCT3 func, int nCols) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) {
        int row = x / nCols;
        out[x] = func((x + first) / nCols, (x + first) % nCols, in0[x], in2[row]);
    }
}
template<typename T1, typename T2, typename T3, typename T4, typename R, typename FCT3>
__global__ void msl::detail::zipIndexKernelAAA(const T1 *in0, const T2 *in1, const T3 *in2, const T4 *in3, R *out, size_t n,
                                                int first, FCT3 func, int nCols) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) {
        int row = x / nCols;
        out[x] = func((x + first) / nCols, (x + first) % nCols, in0[x], in1[row], in2[row], in3[row]);
    }
}
template<typename T, typename T2, typename T3, typename R, typename F>
__global__ void msl::detail::zipIndexDMKernelDA(T *in, T2* in2, T3* in3, R *out, size_t size, size_t first, int dmcols, F func) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size) {
        T2 * in2offset = &in2[x * dmcols];
        out[x] = func(x + first, in2offset, in3[x], in[x]);
    }
}

// new kernel for DM, HK 06.11.2020 -- TODO better to start matrix of threads?
template<typename T1, typename T2, typename R, typename FCT3>
__global__ void msl::detail::zipIndexKernelDM(T1 *in1, T2 *in2, R *out, size_t n, int first, FCT3 func, int nCols) {
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < n) {
        out[k] = func((k + first) / nCols, (k + first) % nCols, in1[k], in2[k]);
    }
}
// new kernel for DM, HK 06.11.2020 -- TODO better to start matrix of threads?
template<typename T1, typename T2, typename R, typename FCT3>
__global__ void msl::detail::zipIndexKernel3(T1 *in1, T2 *in2, T2 *in3, R *out, size_t n, int first, FCT3 func, int nCols) {
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < n) {
        out[k] = func((k + first) / nCols, (k + first) % nCols, in1[k], in2[k], in3[k]);
    }
}

template<typename T1, typename T2, typename T3, typename T4, typename R, typename FCT3>
__global__ void msl::detail::zipKernelAAM(T1 *in1, T2 *in2, T3 *in3, T4 *in4,
                                          R *out, size_t n, int first, int first2, FCT3 func, int nCols) {
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = ((k + first) / nCols) - first2;
    if (k < n) {
        out[k] = func(in1[k], in2[i], in3[i], in4[k]);
    }
}

/*
// new kernel for zipping a DM, two DAs and a DM, HK 20.11.2020
template <typename T1, typename T2, typename T3, typename T4, typename R, typename FCT3>
__global__ void msl::detail::zipKernelAAM(T1* in1, T2* in2, T3* in3, T4*in4,
                                          R* out, size_t n, int first, int first2, FCT3 func, int ncols){
  size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = ((k + first) / ncols) - first2;
  if (k < n) {
    out[k] = func(in1[k], in2[i], in3[i], in4[k]);
  }
}*/

template<typename T1, typename T2, typename R, typename FCT3>
__global__ void msl::detail::zipIndexKernelDA(T1 *in1, T2 *in2, R *out, size_t n,
                                              int first, FCT3 func) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) {
        out[x] = func(x + first, in1[x], in2[x]);
    }
}


template<typename T1, typename T2, typename R, typename FCT4>
__global__ void
msl::detail::zipIndexKernelDC(const T1 *in1, const T2 *in2, R *out, FCT4 func, int gpuRows, int gpuCols,
                              int offset, int elements) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int elementlayer = gpuRows * gpuCols;
    int l = (x + offset) / (elementlayer);
    int remaining_index = (x + offset) % (elementlayer);
    int j = remaining_index / gpuCols;
    int i = remaining_index % gpuCols;
    // calculate
    if (x < elements) {
        out[x] = func(i, j, l, in1[x], in2[x]);
    }

}

template<typename T1, typename T2, typename R, typename FCT3>
__global__ void msl::detail::crossZipIndexKernel(T1 *in1, T2 *in2, R *out, size_t n, int first, FCT3 func, int nCols) {
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        out[k] = func((k + first) / nCols, (k + first) % nCols, in1, in2);
    }
}

template<typename T1, typename T2, typename FCT3>
__global__ void msl::detail::crossZipInPlaceIndexKernel(T1 *in1, T2 *in2, size_t n, int first, FCT3 func, int nCols) {
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;

    func((k + first) / nCols, (k + first) % nCols, in1, in2);

}




