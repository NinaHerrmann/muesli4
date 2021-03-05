
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
class PLMatrix : public ArgumentType {
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
  PLMatrix(int n, int m, int r, int c, int ss, int tw, int th)
      : ArgumentType(), current_data(0), shared_data(0), n(n), m(m), rows(r),
        cols(c), stencil_size(ss), firstRow(Muesli::proc_id * r),
        tile_width(tw), width(2 * stencil_size + tile_width) {}

  /**
   * \brief Default Constructor.
   */
  PLMatrix() {}
  /**
   * \brief Destructor.
   */
  ~PLMatrix() {}

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
   * \brief Returns the number of rows of the padded local matrix.
   *
   * @return The number of rows of the padded local matrix.
   */
  MSL_USERFUNC
  int getRows() const { return rows; }

  /**
   * \brief Returns the number of columns of the padded local matrix.
   *
   * @return The number of columns of the padded local matrix.
   */
  MSL_USERFUNC
  int getCols() const { return cols; }
  MSL_USERFUNC
  void printSharedMemory() {
    printf("Width: %d\n", width);
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < width; j++) {
        printf("%6.2f ", shared_data[i * width + j]);
      }
      printf("\n");
    }
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
    // GPU version: read from shared memory.
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int rowIndex = (row - firstRowGPU - r + stencil_size) + threadIdx.y;
    int colIndex = (col - c + stencil_size) + threadIdx.x;

    // printf("for %d, %d fetching %d, %d. GPU wide thread coords: (%d, %d)\n",
    //        row, col, rowIndex, colIndex, r, c);
      if ((col < 0) || (col >= m)) {
          return 75;
      } else {
          if ((row < 0)) {
              return 100;
          }
          if ((row > n)) {
              return 0;
          }
      }
      return current_data[rowIndex * width + colIndex];
      //return shared_data[rowIndex * width + colIndex];
//	  if ((col < 0) || (col >= m)) {
//	    // out of bounds -> return neutral value
//	    return neutral_value;
//	  } else { // in bounds -> return desired value
//      return current_data[(row-firstRow+stencil_size)*cols + col];
//	  }
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

#ifdef __CUDACC__

  /**
   * @brief Reads matrix to shared memory. Called by the mapStencil skeleton
   * kernel. Note that in this case, current_data will point to the GPU memory
   * it was pointing to at the moment this object was copied to the GPU memory.
   * IMPORTANT: Call on the root level of the kernel. The \em __syncthreads()
   * call at the end will cause the execution to get stuck if not all threads
   * reach it.
   *
   * @param r Global Row
   * @param c Global Col
   * @param tile_width
   * @return __device__
   */
  __device__ void readToSharedMem(int r, int c, int tile_width, int tile_height,
                                  int gpu_rows, int gpu_columns) {
    int tx = threadIdx.x;
    int abs_tx = blockIdx.x * blockDim.x + threadIdx.x;

    int ty = threadIdx.y;
    int abs_ty = blockIdx.y * blockDim.y + threadIdx.y;
    // Get the local row as it relates to the entire partition.
    int row = r - firstRowGPU;

    T *smem = SharedMemory<T>();
    // printf("Thread y: %d, x: %d. GPU data size %d x %d\n", abs_ty, abs_tx,
    //        gpu_rows, gpu_columns);
    if (abs_ty < gpu_rows && (abs_tx < gpu_columns)) {

      // read assigned value into shared memory
      smem[(ty + stencil_size) * width + tx + stencil_size] =
          current_data[(row + stencil_size) * cols +
                       c]; // the current data does not have padding on the
                           // left and right

      // read halo values
      // first row of tile needs to read upper stencil_size rows of halo values
      // corner will be skipped here because it will be read further down the
      // line
      if (ty == 0) {
        for (int i = 0; i < stencil_size; i++) {
          smem[i * width + stencil_size + tx] =
              current_data[(row + i) * cols + c];
        }
      }

      // last row of tile needs to read lower stencil_size rows of halo values
      if (ty == tile_width - 1 || abs_ty == gpu_rows - 1) {
        for (int i = 0; i < stencil_size; i++) {
          smem[(i + stencil_size + ty + 1) * width + stencil_size + tx] =
              current_data[(row + stencil_size + i + 1) * cols + c];
        }
      }

      // first column of tile needs to read left hand side stencil_size columns
      // of
      // halo values
      if (tx == 0) {
        for (int i = 0; i < stencil_size; i++) {
          if (c + i - stencil_size < 0) {
            smem[(ty + stencil_size) * width + i] = 100;
              //  neutral_value_functor(r, c + i - stencil_size);
          } else
            smem[(ty + stencil_size) * width + i] =
                current_data[(row + stencil_size) * cols + c + i -
                             stencil_size];
        }
      }

      // last column of tile needs to read right hand side stencil_size columns
      // of
      // halo values
      if (tx == tile_width - 1 || abs_tx == gpu_columns - 1) {
        for (int i = 0; i < stencil_size; i++) {
          if (c + i + 1 > m - 1)
            smem[(ty + stencil_size) * width + i + tx + 1 + stencil_size] = 100;
               // neutral_value_functor(r, c + i + 1);
          else
            smem[(ty + stencil_size) * width + i + tile_width + stencil_size] =
                current_data[(row + stencil_size) * cols + c + i + 1];
        }
      }

      // upper left corner
      if (tx == 0 && ty == 0) {
        for (int i = 0; i < stencil_size; i++) {
          for (int j = 0; j < stencil_size; j++) {
            if (c + j - stencil_size < 0)
              smem[i * width + j] = 100;
                  //neutral_value_functor(r, c + j - stencil_size);
            else
              smem[i * width + j] =
                  current_data[(row + i) * cols + c + j - stencil_size];
          }
        }
      }

      // upper right corner
      if ((tx == tile_width - 1 || abs_tx == gpu_columns - 1) && ty == 0) {
        for (int i = 0; i < stencil_size; i++) {
          for (int j = 0; j < stencil_size; j++) {
            if (c + j + 1 > m - 1)
              smem[i * width + j + stencil_size + tx + 1] = 100;
                 // neutral_value_functor(r, c + j + 1);
            else
              smem[i * width + j + stencil_size + tile_width] =
                  current_data[(row + i) * cols + c + j + 1];
          }
        }
      }

      // lower left corner
      if (tx == 0 && (ty == tile_width - 1 || abs_ty == gpu_rows - 1)) {
        for (int i = 0; i < stencil_size; i++) {
          for (int j = 0; j < stencil_size; j++) {
            if (c + j - stencil_size < 0)
              smem[(i + stencil_size + ty + 1) * width + j] = 0;
                  //neutral_value_functor(r, c + j - stencil_size);
            else
              smem[(i + stencil_size + tile_width) * width + j] =
                  current_data[(row + i + stencil_size + 1) * cols + c + j -
                               stencil_size];
          }
        }
      }

      // lower right corner
      if ((tx == tile_width - 1 || abs_tx == gpu_columns - 1) &&
          (ty == tile_width - 1 || abs_ty == gpu_rows - 1)) {
        for (int i = 0; i < stencil_size; i++) {
          for (int j = 0; j < stencil_size; j++) {
            if (c + j + 1 > m - 1)
              smem[(i + stencil_size + ty + 1) * width + j + stencil_size + tx + 1] = 0;
                      //neutral_value_functor(r, c + j + 1);
            else
              smem[(i + stencil_size + tile_width) * width + j + stencil_size +
                   tile_width] =
                  current_data[(row + i + stencil_size + 1) * cols + c + j + 1];
          }
        }
      }
    }

    __syncthreads();
    shared_data = smem;
  }
#endif

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
        void setCurentData(T element, int index) {
            current_data[index] = element;
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
  T *current_data, *shared_data;
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
