/*
 * zip_kernels.cu
 *
 *      Authors: Steffen Ernsting <s.ernsting@uni-muenster.de>
 *               Herbert Kuchen <kuchen@uni-muenster.de.
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

template <typename T1, typename T2, typename R, typename FCT2>
__global__ void msl::detail::zipKernel(T1* in1, T2* in2, R* out, size_t n, FCT2 func){
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n) {
    out[x] = func(in1[x], in2[x]);
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
__global__ void msl::detail::zipIndexKernel(T1* in1,T2* in2,R* out,size_t n,int first,FCT3 func,int ncols){
  size_t k = blockIdx.x * blockDim.x + threadIdx.x;

  //if (k < n) {
  out[k] = func((k + first) / ncols,(k + first) % ncols, in1[k], in2[k]);
 // }
}

// new kernel for DM, NH 06.11.2020
template <typename T1, typename T2, typename R, typename FCT3>
__global__ void msl::detail::crossZipIndexKernel(T1* in1,T2* in2,R* out,size_t n,int first,FCT3 func,int ncols){
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        out[k] = func((k + first) / ncols,(k + first) % ncols, in1, in2);
    }
}
// new kernel for DM, NH 06.11.2020
template <typename T1, typename T2, typename FCT3>
__global__ void msl::detail::crossZipInPlaceIndexKernel(T1* in1,T2* in2,size_t n,int first,FCT3 func,int ncols){
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;

    func((k + first) / ncols,(k + first) % ncols, in1, in2);

}
// new kernel for zipping a DM, two DAs and a DM, HK 20.11.2020
template <typename T1, typename T2, typename T3, typename T4, typename R, typename FCT3>
__global__ void msl::detail::zipKernelAAM(T1* in1, T2* in2, T3* in3, T4*in4,
                                          R* out, size_t n, int first, int first2, FCT3 func, int ncols){
  size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = ((k + first) / ncols) - first2;
  if (k < n) {
    out[k] = func(in1[k], in2[i], in3[i], in4[k]);
  }
}

template <typename T1, typename T2, typename R, typename FCT3>
__global__ void msl::detail::zipIndexKernel(T1* in1, T2* in2, R* out, size_t n,
                                            int first, FCT3 func, bool localIndices){
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = localIndices ? 0 : first;
  if (x < n) {
    out[x] = func(x + offset, in1[x], in2[x]);
  }
}

template <typename T1, typename T2, typename R, typename FCT4>
__global__ void msl::detail::zipIndexKernel(T1* in1, T2* in2, R* out, GPUExecutionPlan<T1> plan,
                                            FCT4 func,bool localIndices){
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;

  int rowOffset = localIndices ? 0 : plan.firstRow;
  int colOffset = localIndices ? 0 : plan.firstCol;

  if (y < plan.nLocal) {
    if (x < plan.mLocal) {
      out[y * plan.mLocal + x] = func(y + rowOffset,
    		     	 	 	 	 	  x + colOffset,
    		     	 	 	 	 	  in1[y * plan.mLocal + x],
    		     	 	 	 	 	  in2[y * plan.mLocal + x]);
    }
  }
}
template <typename T1, typename T2, typename R, typename FCT4>
__global__ void msl::detail::zipIndexKernel(T1* in1,T2* in2,R* out,GPUExecutionPlan<T1> plan,FCT4 func,
                                            bool localIndices, int nrow, int ncol, bool dim3){
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
                out[localoverall] = func(y + rowOffset,
                                         x + colOffset,
                                         z + depthOffset,
                                                in1[localoverall],
                                                in2[localoverall]);
            }
    }
  }
}






