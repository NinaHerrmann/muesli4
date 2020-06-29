/*
 * larray.h
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
#include "functor_base.h"

namespace msl {

template <typename T>
class DArray;

/**
 * \brief Class \em LArray represents a shallow copy of class \em DArray.
 *
 * Class \em LArray represents a shallow copy of class \em DArray. Instantiations of this
 * class can be added as additional arguments to a user functor in terms of data
 * members. It only stores pointers to the local partition (CPU and GPU partitions)
 * but not the data itself. Therefore, copy construction comes at lower costs compared
 * to copy constructing a DArray which would imply copying of (big) data. This is
 * important because when initializing data members, the copy constructor will be called.
 * The \em L in \em LArray stands for \em Local and indicates that the user functor only
 * has access to the local partition of a \em DArray.
 *
 * \tparam T The element type.
 */
template <typename T>
class LArray : public ArgumentType
{
public:

  /**
   * \brief Class \em Tile represents a small portion of data that is stored in
   *        GPU shared memory for faster access.
   *
   * Class \em Tile represents a small portion of data that is stored in GPU
   * shared memory for faster access. This is very beneficial when data is
   * often reused. The size of a tile is determined by the template parameter
   * \em tile_width. Note that the size of \em LArray must be divided by
   * \em tile_width without remainder.
   * Can only be used in device code and not in CPU code.
   *
   * @tparam tile_width The size of a tile, e.g. number of elements in a shared
   *                    memory block. Value must be known at compile time.
   */
  class Tile
  {
  public:
    /**
     * \brief Constructor loads data to shared memory. Each thread of a thread
     *        block loads a single data element to shared memory.
     *
     * @param tile The (static) shared memory buffer.
     * @param la LArray instance to read data from.
     * @param ti tileIndex Index of the tile.
     * @param tw The tile width.
     */
    MSL_USERFUNC
    Tile(T* tile_, const LArray<T>& la, int tileIndex, int tw)
      : tile(tile_), tile_width(tw), tileIndex(tw)
    {
#ifdef __CUDA_ARCH__
      // get thread id
      int tx = threadIdx.x;
      // load corresponding element to shared memory buffer
      tile[tx] = la.current_plan.d_Data[tileIndex*tw+tx];
      // synchronize threads (all loads have been completed)
      __syncthreads();
#else
      tile = &(la.getDArray()->getLocalPartition()[tileIndex * tw]);
#endif
    }

    /**
     * \brief Destructor.
     *
     * Synchronizes threads of a thread block in order to ensure that all memory
     * loads are completed.
     */
    MSL_USERFUNC
    ~Tile()
    {
#ifdef __CUDA_ARCH__
      // synchronize threads (all loads have been completed)
      __syncthreads();
#endif
    }

    /**
     * \brief Returns the element at index \em index. Note that 0 <= i < \em tile_width
     *        must hold.
     *
     * @param index The index of the requested element.
     * @return The element at index \em index.
     */
    MSL_USERFUNC
    T get(int index)
    {
      return tile[index];
    }

  private:
    T* tile;
    int tile_width, tileIndex;
  };

public:
  /**
   * \brief Constructor. Gathers all pointers (CPU + GPUs) pointing to a local
   *        partition of a given \em DArray.
   *
   * @param da The distributed array whose local partition will be accessed.
   * @param gpu_dist Specifies the distribution among GPUs. Default is distributed.
   *                 May also be copy distributed, in this case the local partition
   *                 of a distributed array is copy distributed among all GPUs.
   */
  LArray(msl::DArray<T>& da, Distribution gpu_dist = Distribution::DIST);

  LArray(msl::DArray<T>& da, detail::FunctorBase* f, Distribution gpu_dist = Distribution::DIST);

  /**
   * \brief Virtual destructor.
   */
  virtual ~LArray();

  /**
   * \brief Updates the pointer that is accessed within the get function to point
   *        to the correct memory.
   *
   * Updates the pointer that is accessed within the get function to point
   * to the correct memory. When accessed by the CPU, the pointer must point
   * to host main memory, when accessed by GPU \em i, the pointer must point
   * to device main memory of GPU \em i.
   */
  virtual void update();

  /**
   * \brief Returns the number of elements.
   *
   * @return The number of elements.
   */
  MSL_USERFUNC
  int getSize() const;

  /**
   * \brief Returns the element at index \em index. Uses local indices. Note that
   *        0 <= \em index < \em getSize() must hold (not checked for performance
   *        reasons).
   *
   * @param index The index of the requested element.
   * @return The requested element at index \em index.
   */
  MSL_USERFUNC
  T operator[](int index) const;

  /**
   * \brief Returns the element at index \em index. Uses global indices.
   *
   * @param index The index of the requested element.
   * @return The requested element at index \em index.
   */
  MSL_USERFUNC
  T get(int index) const;

  virtual int getSmemSize() const
  {
    if (tile_width == -1)
      return 0;
    else
      return sizeof(T)*tile_width;
  }

  /**
   * \brief Returns a shared memory tile.
   *
   * Returns a shared memory tile. A tile index denotes, which tile of a
   * \em LArray will be loaded to shared memory. Note that
   * 0 <= \em tileIndex < \em getSize()/tile_width must hold.
   *
   * @param tileIndex The index of the tile to be loaded into shared memory.
   * @param functor The functor.
   * @tparam F The functor type.
   */
  template <class F>
  MSL_USERFUNC
  Tile getTile(int tileIndex, F functor) const
  {
    int tw = functor->getTileWidth();
#ifdef __CUDA_ARCH__
    if (!valid_smem) {
      smem_ptr = functor->template getSmemPtr<T>(tw);
      valid_smem = 1;
    }
#else
    smem_ptr = 0;
#endif
    return Tile(smem_ptr, *this, tileIndex, tw);
  }

  MSL_USERFUNC
  DArray<T>* getDArray() const
  {
    return darray;
  }

private:
  GPUExecutionPlan<T> current_plan;
  int current_device;
  Distribution dist;
  DArray<T>* darray;
  mutable T* smem_ptr;
  mutable bool valid_smem;
};

}

#include "../src/larray.cpp"
