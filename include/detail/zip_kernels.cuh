/*
 * zip_kernels.h
 *
 *      Authors: Steffen Ernsting <s.ernsting@uni-muenster.de>
 *               Herbert Kuchen <kuchen@uni-muenster.de>
 * 
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2020 Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *                Herbert Kuchen <kuchen@uni-muenster.de>.
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

namespace msl {

namespace detail {

template<typename T1, typename T2, typename R, typename FCT2>
__global__ void
zipKernel(T1* in1,
          T2* in2,
          R* out,
          size_t n,   
          FCT2 func);

template<typename T1, typename T2, typename R, typename FCT2>
__global__ void
zip3DKernel(T1* in1,
          T2* in2,
          R* out,
          GPUExecutionPlan<T1> plan,
          FCT2 func, int nrow, int ncol);
		  
template<typename T1, typename T2, typename T3, typename R, typename FCT3>
__global__ void
zipKernel(T1* in1,
	  T2* in2,
	  T3* in3,
	  R* out,
	  size_t n,
	  FCT3 func);

// new kernel for DM
template<typename T1, typename T2, typename R, typename FCT3>
__global__ void
zipIndexKernel(T1* in1,	  	   
               T2* in2,
      	       R* out,
	       size_t n,
  	       int first,
	       FCT3 func,
  	       int ncols);

// new kernel for DM
template<typename T1, typename T2, typename R, typename FCT3>
__global__ void
crossZipIndexKernel(T1* in1,
               T2* in2,
               R* out,
               size_t n,
               int first,
               FCT3 func,
               int ncols);
// new kernel for DM
template<typename T1, typename T2, typename FCT3>
__global__ void
crossZipInPlaceIndexKernel(T1* in1,
               T2* in2,
               size_t n,
               int first,
               FCT3 func,
               int ncols);
// new kernel for zipping a DM, two DAs and a DM		  	   	  	   
template <typename T1, typename T2, typename T3, typename T4, typename R, typename FCT3>
__global__ void 
zipKernelAAM(T1* in1, 
             T2* in2, 
             T3* in3, 
             T4* in4,
             R* out, 
             size_t n, 
             int first,
             int first2, 
             FCT3 func, 
             int ncols);

template<typename T1, typename T2, typename R, typename FCT3>
__global__ void
zipIndexKernel(T1* in1,
		  	   T2* in2,
		  	   R* out,
		  	   size_t n,
		  	   int first,
		  	   FCT3 func,
		  	   bool localIndices);

template<typename T1, typename T2, typename R, typename FCT4>
__global__ void
zipIndexKernel(T1* in1,
		       T2* in2,
		       R* out,
		       GPUExecutionPlan<T1> plan,
		       FCT4 func,
			   bool localIndices);

template<typename T1, typename T2, typename R, typename FCT4>
__global__ void
zipIndexKernel(T1* in1,
               T2* in2,
               R* out,
               GPUExecutionPlan<T1> plan,
               FCT4 func,
               bool localIndices, int nrow, int ncol, bool dim3);
}
}

#include "../../src/zip_kernels.cu"

