/*
 * plmatrix.h
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

#include "muesli.h"

#pragma once

namespace msl {

/**
 * \brief Class PLMatrix represents a padded local matrix (partition). It serves
 *        as input for the mapStencil skeleton and actually is a shallow copy that
 *        only stores the pointers to the data. The data itself is managed by the
 *        mapStencil skeleton. For the user, the only important part is the \em get
 *        function.
 *
 * @tparam T The element type.
 */
template <typename T>
class PLMatrix
{
public:
  /**
   * \brief Constructor: creates a PLMatrix.
   */
  PLMatrix(int N, int r, int c, int s, T nv, T* o, int NG, int ngpu)
	: n(N), rows(r), cols(c), stencil_size(s), neutral_value(nv), origin(o), ng(NG), nGPU(ngpu)
  {
#ifdef __CUDACC__
    current_data = new T[n];
    CUDA_CHECK_RETURN(cudaMallocHost(&current_data, n*sizeof(T)));
#else
#endif
    downloadorigin();
#ifdef __CUDACC__
    int bytes = sizeof(T) * nGPU;
    int nCPU = n - (ng * nGPU);
    gpupointers = (T**)malloc(ng * sizeof(T));
    for (int i = 0; i < ng; i++) {
      cudaSetDevice(i);
      CUDA_CHECK_RETURN(cudaMalloc((void **)&gpupointers[i], bytes));
      CUDA_CHECK_RETURN(
          cudaMemcpyAsync(gpupointers[i], current_data + (nCPU + (i * nGPU)), bytes,
                          cudaMemcpyHostToDevice, Muesli::streams[i]));
      }
#endif
  }

  /**
   * \brief Destructor.
   */
  ~PLMatrix()
  {
  }

  /**
   * \brief Updates the data on the CPU.
   */
  void updatecurrent()
  {
#ifdef __CUDACC__
    int bytes = sizeof(T) * nGPU;
    int nCPU = n - (ng * nGPU);
    for (int i = 0; i < ng; i++) {
      cudaSetDevice(i);
      CUDA_CHECK_RETURN(
          cudaMemcpyAsync(current_data + (nCPU + (i * nGPU)), gpupointers[i], bytes,
                          cudaMemcpyDeviceToHost, Muesli::streams[i]));
      }

    // wait until download is finished
    for (int i = 0; i < ng; i++) {
      CUDA_CHECK_RETURN(cudaStreamSynchronize(Muesli::streams[i]));
    }
#endif
  }
    /**
    * \brief Updates the data on the GPU.
    */
   void uploadcurrent() {
#ifdef __CUDACC__
    int bytes = sizeof(T) * nGPU;
    int nCPU = n - (ng * nGPU);
    for (int i = 0; i < ng; i++) {
      cudaSetDevice(i);
      CUDA_CHECK_RETURN(
          cudaMemcpyAsync(gpupointers[i], current_data + (nCPU + (i * nGPU)), bytes,
                          cudaMemcpyHostToDevice, Muesli::streams[i]));
      }

#endif
      return;
    }
  /**
   * \brief copies the data from origin to current.
   */
  void downloadorigin()
  {
    #pragma omp parallel for
    for (int i = 0; i < (cols * rows) -1; i++) {
      current_data[i] = origin[i];
    }
  }

    /**
     * \brief Get the respective device pointer.
     */
    T* getDevicePtr(int i)
    {
      return gpupointers[i];
    }
  /**
   * \brief Returns the element at the given global indices (\em row, \em col).
   *
   * @param row The global row index.
   * @param col The global col index.
   */
  MSL_USERFUNC
  T get(int row, int col) const
  {
    //updatecurrent();
    // CPU version: read from main memory.
	  // bounds check
    if ((col < 0) || (col >= cols) || (row < 0) || (row >= rows)) {
      // out of bounds -> return neutral value
      return neutral_value;
    } else { // in bounds -> return desired value
      return current_data[(row * cols) + (col % cols)];
    }
  }

private:
  int rows, cols, stencil_size, ng, nGPU, n;
  T neutral_value;
  T* current_data;
  T *origin;
  T** gpupointers;
};

}





