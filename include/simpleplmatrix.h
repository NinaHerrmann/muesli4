
/*
 * plmatrix.h
 *
 *      Author: Steffen Ernsting <s.ernsting@uni-muenster.de>, Endi Zhupani
 * <endizhupani@uni-muenster.de>
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

#pragma once

#include "argtype.h"
#include "detail/shared_mem.h"

namespace msl {

/**
 * \brief Class PLMatrix represents a padded local matrix (partition). It serves
 *        as input for the mapStencil skeleton and actually is a shallow copy
 * that only stores the pointers to the data. The data itself is managed by the
 *        mapStencil skeleton. For the user, the only important part is the \em
 * get function.
 *
 * @tparam T The element type.
 * @tparam NeutralValueFunctor Functor that will provide the neutral value. The
 * Out type of the function must be the same as T
 */
template <typename T>
class SimplePLMatrix : public ArgumentType {
public:
  /**
   * @brief Construct a new PLMatrix object
   *
   * @param n Global number of matrix rows
   * @param m Global number of matrix columns
   * @param r Local number of matrix rows. Without padding
   * @param c Local number of matrix columns. Without padding
   * @param ss Stencil size. Should be the stencil radius. The number of
   * elements inthe horizontal or vertical direction from the center of the
   * stencil to the edge
   * @param tw Width of the data tile without the padding
   * @param th Height of the data tile without the padding
   * @param gr Number of rows processed by the GPU
   * @param gc number of columns processed by the GPU
   * return neutral values for a specific position in the matrix
   */
  SimplePLMatrix(int n, int m, int r, int c, int ss, int tw, int th)
      : ArgumentType(), current_data(0), n(n), m(m), rows(r),
        cols(c), stencil_size(ss), firstRow(Muesli::proc_id * r),
        tile_width(tw), width(2 * stencil_size + tile_width) {}

  /**
   * \brief Default Constructor.
   */
  SimplePLMatrix() {}
  /**
   * \brief Destructor.
   */
  ~SimplePLMatrix() {}
    /**
     * \brief Adds another pointer to data residing in GPU or in CPU memory,
     *        respectively.
     */
    void addDevicePtr(T *d_ptr) {
        ptrs.push_back(d_ptr);
        if (it != ptrs.begin()) {
            it = ptrs.begin();
            current_data = *it;
        }
    }
    /**
     * \brief Updates the pointer to point to current data (that resides in one of
     *        the GPUs memory or in CPU memory, respectively.
     */
    void update() {
        if (++it == ptrs.end()) {
            it = ptrs.begin();
        }
        current_data = *it;
    }

  /**
   * \brief Returns the element at the given global indices (\em row, \em col).
   *
   * @param row The global row index.
   * @param col The global col index.
   */
  MSL_USERFUNC
    T get(int row, int col) const {
#ifdef __CUDA_ARCH__

      if ((col < 0) || (col >= m) || (row < 0)) {
          // out of bounds -> return neutral value
          return 100;
      } else {
          if (row >= n){
              return 0;
          } else {
              int r = blockIdx.y * blockDim.y + threadIdx.y;
              int c = blockIdx.x * blockDim.x + threadIdx.x;
              int onlycpucols = firstRowGPU ;

              return current_data[(row - firstRow + 1) * cols - (onlycpucols * cols) + col];
          }
      }
#else
    // CPU version: read from main memory.
    // bounds check
    if ((col < 0) || (col >= m) || (row < 0) || (row >= n)) {
      // out of bounds -> return neutral value
      return 100;
    } else { // in bounds -> return desired value

      return current_data[(row - firstRow +1 ) * cols + col];
    }
#endif
  }

    void show(const std::string &descr) {
        std::ostringstream s;
        s << descr;
        if (msl::isRootProcess()) {
            s << "[";
            for (int i = 0; i < (n*m) - 1; i++) {
                s << current_data[i];
                ((i + 1) % cols == 0) ? s << "\n " : s << " ";
                ;
            }
            s << current_data[n - 1] << "]" << std::endl;
            s << std::endl;
        }

        if (msl::isRootProcess())
            printf("%s", s.str().c_str());
    }
  /**
   * \brief Sets the first row index for the current device.
   *
   * @param fr The first row index.
   */
  void setFirstRowGPU(int fr) { firstRowGPU = fr; }

  /**
   * @brief Set the size of the Stencil.
   *
   * @param s
   */
  void setStencilSize(int s) { stencil_size = s; }

  /**
   * @brief Set Neutral Value functor
   *
   * @param nv
   */
  //template <typename NeutralValueFunctor>
  //void setNVV(NeutralValueFunctor nv) { neutral_value_functor = nv; }
void updateCpuCurrentData(T *padded_local_matrix, int nCPU){
      for (int i = 0; i < nCPU; i++) {
          current_data[i] = padded_local_matrix[i];
      }
}
  /**
   * @brief Set the GLOBAL index of the first element on the PLM to be processed
   * by the GPU.
   *
   * @param fi
   */
  void setFirstGPUIdx(int fi) { firstGPUIdx = fi; }

private:
  std::vector<T *> ptrs;
  typename std::vector<T *>::iterator it;
  T *current_data;
  int n;
  int m;
  int rows;
  int cols;
  int stencil_size;
  // First row of the local matrix in the global matrix
  int firstRow;
  int firstRowGPU;
  int firstGPUIdx;
  int tile_width;
  int tile_height;
  int width;
};

} // namespace msl
