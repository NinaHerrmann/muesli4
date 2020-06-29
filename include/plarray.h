/*
 * plarray.h
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

#include "argtype.h"
#include "shared_mem.h"

namespace msl {

/**
 * \brief Class PLArray represents a padded local array (partition). It serves
 *        as input for the mapStencil skeleton and actually is a shallow copy that
 *        only stores the pointers to the data. The data itself is managed by the
 *        mapStencil skeleton. For the user, the only important part is the \em get
 *        function.
 *
 * @tparam T The element type.
 */
template <typename T>
class PLArray : public ArgumentType
{
public:
  /**
   * \brief Constructor: creates a PLArray.
   */
  PLArray(int n, int nl, int ss, int tw, T nv)
    : ArgumentType(),
      current_data(0),
      shared_data(0),
      n(n),
      nLocal(nl),
      stencil_size(ss),
      firstIndex(Muesli::proc_id*nLocal),
      firstIndexGPU(0),
      tile_width(tw),
      width(2*stencil_size+tile_width),
      neutral_value(nv)
  {
  }

  /**
   * \brief Destructor.
   */
  ~PLArray()
  {
  }

  /**
   * \brief Adds another pointer to data residing in GPU or in CPU memory,
   *        respectively.
   */
  void addDevicePtr(T* d_ptr)
  {
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
  void update()
  {
    if (++it == ptrs.end()) {
      it = ptrs.begin();
    }
    current_data = *it;
  }

  /**
   * \brief Returns the size of the padded local array.
   *
   * @return The size of the padded local array.
   */
  MSL_USERFUNC
  int getLocalSize() const
  {
    return nLocal;
  }

  /**
   * \brief Returns the element at the given global index \em index.
   *
   * @param index The global index.
   */
  MSL_USERFUNC
  T get(int index) const
  {
#ifdef __CUDA_ARCH__
    int i = blockIdx.x * blockDim.x;
    int shared_index = index-firstIndexGPU-i+stencil_size;
    return shared_data[shared_index];
#else
    // bounds check
    if (index < 0 || index > n) {
      // out of bounds -> return neutral value
      return neutral_value;
    } else { // in bounds -> return desired value
      return current_data[index - firstIndex + stencil_size];
    }
#endif
  }

#ifdef __CUDACC__
  /**
   * \brief Each thread block (on the GPU) reads elements from the padded local
   *        array to shared memory. Called by the mapStencil skeleton kernel.
   */
  __device__
  void readToSharedMem(int index, int tile_width)
  {
    int tx = threadIdx.x;
    int i = index - firstIndex;
    T *smem = SharedMemory<T>();

    // read assigned value into shared memory
    smem[tx+stencil_size] = current_data[i+stencil_size];

    // read halo values
    // first thread needs to read stencil_size halo values from left hand side
    if (tx == 0) {
      for (int j = 0; j < stencil_size; j++) {
        smem[tx+j] = current_data[i+j];
      }
    }

    // last thread needs to read stencil_size halo values from right hand side
    if (tx == tile_width-1) {
      for (int j = 0; j < stencil_size; j++) {
        smem[tx+j+stencil_size+1] = current_data[i+stencil_size+j+1];
      }
    }

    __syncthreads();

    shared_data = smem;
  }
#endif

  /**
   * \brief Sets the first index for the current device.
   *
   * @param fr The first index.
   */
  void setFirstIndexGPU(int index)
  {
    firstIndexGPU = index;
  }

private:
  std::vector<T*> ptrs;
  typename std::vector<T*>::iterator it;
  T* current_data, *shared_data;
  int n, nLocal, stencil_size, firstIndex, firstIndexGPU, tile_width, width;
  T neutral_value;
};

}






