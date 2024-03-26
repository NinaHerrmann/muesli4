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

namespace msl {

template <typename T>
class DMatrix;

/**
 * \brief Class \em LMatrix represents a shallow copy of class \em DMatrix.
 *
 * Class \em LMatrix represents a shallow copy of class \em DMatrix. Instantiations of this
 * class can be added as additional arguments to a user functor in terms of data
 * members. It only stores pointers to the local partition (CPU and GPU partitions)
 * but not the data itself. Therefore, copy construction comes at lower costs compared
 * to copy constructing a DMatrix which would imply copying of (big) data. This is
 * important because when initializing data members, the copy constructor will be called.
 * The \em L in \em LMatrix stands for \em Local and indicates that the user functor only
 * has access to the local partition of a \em DMatrix.
 *
 * \tparam T The element type.
 */
template <typename T>
class LMatrix : public ArgumentType
{
public:
// TODO: Class Tile needs to be debugged. See class RowTile and ColTile.
//#ifdef __CUDACC__
//  /**
//   * \brief Class \em Tile represents a small portion of data that is stored in
//   *        GPU shared memory for faster access.
//   *
//   * Class \em Tile represents a small portion of data that is stored in GPU
//   * shared memory for faster access. This is very beneficial when data is
//   * often reused. The number of rows and columns of a tile is determined
//   * by the template parameter \em tile_width. Note that the number of rows and
//   * columns of a \em LMatrix must be divided by \em tile_width without remainder.
//   * Can only be used in device code and not in CPU code.
//   *
//   */
//  class Tile
//  {
//  public:
//    /**
//     * \brief Constructor loads data to shared memory. Each thread of a thread
//     *        block loads a single data element to shared memory.
//     *
//     * @param tile The (static) shared memory buffer.
//     * @param lm LMatrix instance to read data from.
//     * @param rowIndex Row index where the executing thread reads data from.
//     * @param colIndex Column index where the executing thread reads data from.
//     */
//    __device__
//    Tile(T* tile, const LMatrix<T>& lm, int rowIndex, int colIndex, int tw)
//      : _tile(tile), tile_width(tw)
//    {
//      // get thread ids
//      int tx = threadIdx.x, ty = threadIdx.y;
//      // load corresponding element to shared memory buffer
//      _tile[ty*tile_width + tx] = lm.data_gpu[rowIndex*lm.cols_gpu + colIndex];
//      // synchronize threads (all loads have been completed)
//      __syncthreads();
//    }
//
//    /**
//     * \brief Destructor.
//     *
//     * Synchronizes threads of a thread block in order to ensure that all memory
//     * loads are completed.
//     */
//    __device__
//    ~Tile()
//    {
//      // synchronize threads (all loads have been completed)
//      __syncthreads();
//    }
//
//    /**
//     * \brief Returns the element at indices (\em rowIndex, \em colIndex). Note
//     *        that 0 <= \em rowIndex, \em colIndex < \em tile_width must hold.
//     *
//     * @param rowIndex The row index of the requested element.
//     * @param colIndex The column index of the requested element.
//     * @return The requested element.
//     */
//    __device__
//    T get(int rowIndex, int colIndex)
//    {
//      return _tile[rowIndex*tile_width + colIndex];
//    }
//
//  private:
//    T* _tile;
//    int tile_width;
//  };
//#endif

  /**
   * \brief Class \em RowTile represents a small portion of data that is stored in
   *        GPU shared memory for faster access.
   *
   * Class \em RowTile represents a small portion of data that is stored in GPU
   * shared memory for faster access. This is very beneficial when data is
   * often reused. The number of rows and columns of a tile is determined
   * by the template parameter \em tile_width. Note that the number of rows and
   * columns of a \em LMatrix must be divided by \em tile_width without remainder.
   * Can only be used in device code and not in CPU code.
   *
   */
  class RowTile
  {
  public:

    /**
     * \brief Constructor loads data to shared memory. Each thread of a thread
     *        block loads a single data element to shared memory.
     *
     * @param tile The (static) shared memory buffer.
     * @param lm LMatrix instance to read data from.
     * @param tileIndex Index of the row tile.
     * @param rowIndex Row index of the thread.
     * @param tw The tile width.
     */
    MSL_USERFUNC
    RowTile(T* tile_, const LMatrix<T>& lm, int tileIndex, int rowIndex, int tw)
      : tile(tile_), tile_width(tw), mLocal(lm.current_plan.mLocal)
    {
#ifdef __CUDA_ARCH__
      // get thread ids
      int tx = threadIdx.x, ty = threadIdx.y;
      // load corresponding element to shared memory buffer
      tile[ty*tw + tx] = lm.current_plan.d_Data[rowIndex*mLocal + tileIndex*tw + tx];
      // calculate index offset for faster access
      offset = threadIdx.y*tile_width;
      // synchronize threads (all loads have been completed)
      __syncthreads();
#else
      tile = &(lm.getDMatrix()->getLocalPartition()[rowIndex*mLocal + tileIndex*tw]);
#endif
    }

    /**
     * \brief Destructor.
     *
     * Synchronizes threads of a thread block in order to ensure that all memory
     * loads are completed.
     */
    MSL_USERFUNC
    ~RowTile()
    {
#ifdef __CUDA_ARCH__
      __syncthreads();
#endif
    }

// TODO: Debug this! (Do we need this getter?)
//    /**
//     * \brief Returns the element at indices (\em rowIndex, \em colIndex). Note
//     *        that 0 <= \em rowsIndex, \em colIndex < \em tile_width must hold.
//     *
//     * @param rowIndex The row index of the requested element.
//     * @param colIndex The column index of the requested element.
//     * @return The requested element.
//     */
//    MSL_USERFUNC
//    T get(int rowIndex, int colIndex)
//    {
//#ifdef __CUDA_ARCH__
//      return tile[rowIndex*tile_width + colIndex];
//#else
//      return tile[colIndex];
//#endif
//    }

    /**
     * \brief Returns the element at indices (\em rowIndex). The column index is
     *        determined by the corresponding column thread id. Note that
     *        0 <= \em rowIndex < \em tile_width must hold.
     *
     * @param rowIndex The row index of the requested element.
     * @return The requested element.
     */
    MSL_USERFUNC
    T get(int index)
    {
#ifdef __CUDA_ARCH__
      return tile[offset + index];
#else
      return tile[index];
#endif
    }

  private:
    T* tile;
    int tile_width, offset, mLocal;
  };

  /**
   * \brief Class \em ColTile represents a small portion of data that is stored in
   *        GPU shared memory for faster access.
   *
   * Class \em ColTile represents a small portion of data that is stored in GPU
   * shared memory for faster access. This is very beneficial when data is
   * often reused. The number of rows and columns of a tile is determined
   * by the template parameter \em tile_width. Note that the number of rows and
   * columns of a \em LMatrix must be divided by \em tile_width without remainder.
   * Can only be used in device code and not in CPU code.
   *
   * @tparam tile_width The number of rows and columns of a tile, e.g. number of
   *                    elements in a shared memory block. Value must be known at compile time.
   */
  class ColTile
  {
  public:
    /**
     * \brief Constructor loads data to shared memory. Each thread of a thread
     *        block loads a single data element to shared memory.
     *
     * @param tile The (static) shared memory buffer.
     * @param lm LMatrix instance to read data from.
     * @param tileIndex Index of the column tile.
     * @param colIndex Column index of the thread.
     * @param tw The tile width.
     */
    MSL_USERFUNC
    ColTile(T* tile_, const LMatrix<T>& lm, int tileIndex, int colIndex, int tw)
      : tile(tile_), tile_width(tw), mLocal(lm.current_plan.mLocal)
    {
#ifdef __CUDA_ARCH__
      // get thread ids
      int tx = threadIdx.x, ty = threadIdx.y;
      // load corresponding element to shared memory buffer (transposed)
      tile[tx*tw + ty] = lm.current_plan.d_Data[tileIndex*tw*lm.current_plan.mLocal + colIndex + ty*lm.current_plan.mLocal];
      // calculate index offset for faster access
      offset = threadIdx.y*tile_width;
      // synchronize threads (all loads have been completed)
      __syncthreads();
#else
      mLocal = lm.getDMatrix()->getLocalCols();
      tile = &(lm.getDMatrix()->getLocalPartition()[tileIndex*tw*mLocal + colIndex]);
#endif
    }

    /**
     * \brief Destructor.
     *
     * Synchronizes threads of a thread block in order to ensure that all memory
     * loads are completed.
     */
    MSL_USERFUNC
    ~ColTile()
    {
#ifdef __CUDA_ARCH__
      __syncthreads();
#endif
    }

// TODO: Debug this! (Do we need this getter?)
//    /**
//     * \brief Returns the element at indices (\em rowIndex, \em colIndex). Note
//     *        that 0 <= \em rowsIndex, \em colIndex < \em tile_width must hold.
//     *
//     * @param rowIndex The row index of the requested element.
//     * @param colIndex The column index of the requested element.
//     * @return The requested element.
//     */
//    MSL_USERFUNC
//    T get(int rowIndex, int colIndex)
//    {
//#ifdef __CUDA_ARCH__
//      return tile[rowIndex*tile_width + colIndex];
//#else
//      return tile[rowIndex];
//#endif
//    }

    /**
     * \brief Returns the element at indices (\em colIndex). The row index is
     *        determined by the corresponding row thread id. Note that
     *        0 <= \em colIndex< \em tile_width must hold.
     *
     * @param index The index of the requested element.
     * @return The requested element.
     */
    MSL_USERFUNC
    T get(int index)
    {
#ifdef __CUDA_ARCH__
      return tile[offset + index];
#else
      return tile[index*mLocal];
#endif
    }

  private:
    T* tile;
    int tile_width, mLocal, offset;
  };

  template <int tile_width>
  class RowTileStatic
  {
  public:

    /**
     * \brief Constructor loads data to shared memory. Each thread of a thread
     *        block loads a single data element to shared memory.
     *
     * @param tile The (static) shared memory buffer.
     * @param lm LMatrix instance to read data from.
     * @param rowIndex Row index where the executing thread reads data from.
     * @param colIndex Column index where the executing thread reads data from.
     */
    MSL_USERFUNC
    RowTileStatic(T* tile, const LMatrix<T>& lm, int tileIndex, int rowIndex)
      : _tile(tile)
    {
#ifdef __CUDA_ARCH__
      // get thread ids
      int tx = threadIdx.x, ty = threadIdx.y;
      // load corresponding element to shared memory buffer
      _tile[ty*tile_width + tx] = lm.current_plan.d_Data[rowIndex*lm.current_plan.mLocal + tileIndex*tile_width + tx];
      // synchronize threads (all loads have been completed)
      __syncthreads();
#else
      _tile = &(lm.getDMatrix()->getLocalPartition()[rowIndex*lm.getDMatrix()->getLocalCols() + tileIndex*tile_width]);
#endif
    }

    /**
     * \brief Destructor.
     *
     * Synchronizes threads of a thread block in order to ensure that all memory
     * loads are completed.
     */
    MSL_USERFUNC
    ~RowTileStatic()
    {
#ifdef __CUDA_ARCH__
      __syncthreads();
#endif
    }

    /**
     * \brief Returns the element at indices (\em rowIndex, \em colIndex). Note
     *        that 0 <= \em rowsIndex, \em colIndex < \em tile_width must hold.
     *
     * @param rowIndex The row index of the requested element.
     * @param colIndex The column index of the requested element.
     * @return The requested element.
     */
//    __device__
//    T get(int rowIndex, int colIndex)
//    {
//      return _tile[rowIndex*tile_width + colIndex];
//    }

    /**
     * \brief Returns the element at indices (\em rowIndex). The column index is
     *        determined by the corresponding column thread id. Note that
     *        0 <= \em rowIndex < \em tile_width must hold.
     *
     * @param rowIndex The row index of the requested element.
     * @return The requested element.
     */
    MSL_USERFUNC
    T get(int index)
    {
#ifdef __CUDA_ARCH__
      return _tile[threadIdx.y*tile_width + index];
#else
      return _tile[index];
#endif
    }

  private:
    T* _tile;
  };

  /**
   * \brief Class \em ColTile represents a small portion of data that is stored in
   *        GPU shared memory for faster access.
   *
   * Class \em ColTile represents a small portion of data that is stored in GPU
   * shared memory for faster access. This is very beneficial when data is
   * often reused. The number of rows and columns of a tile is determined
   * by the template parameter \em tile_width. Note that the number of rows and
   * columns of a \em LMatrix must be divided by \em tile_width without remainder.
   * Can only be used in device code and not in CPU code.
   *
   * @tparam tile_width The number of rows and columns of a tile, e.g. number of
   *                    elements in a shared memory block. Value must be known at compile time.
   */
  template <int tile_width>
  class ColTileStatic
  {
  public:
    /**
     * \brief Constructor loads data to shared memory. Each thread of a thread
     *        block loads a single data element to shared memory.
     *
     * @param tile The (static) shared memory buffer.
     * @param lm LMatrix instance to read data from.
     * @param rowIndex Row index where the executing thread reads data from.
     * @param colIndex Column index where the executing thread reads data from.
     */
    MSL_USERFUNC
    ColTileStatic(T* tile, const LMatrix<T>& lm, int tileIndex, int colIndex)
      : _tile(tile), mLocal(lm.current_plan.mLocal)
    {
#ifdef __CUDA_ARCH__
      // get thread ids
      int tx = threadIdx.x, ty = threadIdx.y;
      // load corresponding element to shared memory buffer
      _tile[ty*tile_width + tx] = lm.current_plan.d_Data[tileIndex*tile_width*lm.current_plan.mLocal + colIndex + ty*lm.current_plan.mLocal];
      // synchronize threads (all loads have been completed)
      __syncthreads();
#else
      mLocal = lm.getDMatrix()->getLocalCols();
      _tile = &(lm.getDMatrix()->getLocalPartition()[tileIndex*tile_width*mLocal + colIndex]);
#endif
    }

    /**
     * \brief Destructor.
     *
     * Synchronizes threads of a thread block in order to ensure that all memory
     * loads are completed.
     */
    MSL_USERFUNC
    ~ColTileStatic()
    {
#ifdef __CUDA_ARCH__
      __syncthreads();
#endif
    }

    /**
     * \brief Returns the element at indices (\em rowIndex, \em colIndex). Note
     *        that 0 <= \em rowsIndex, \em colIndex < \em tile_width must hold.
     *
     * @param rowIndex The row index of the requested element.
     * @param colIndex The column index of the requested element.
     * @return The requested element.
     */
//    MSL_USERFUNC
//    T get(int rowIndex, int colIndex)
//    {
//      return _tile[rowIndex*tile_width + colIndex];
//    }

    /**
     * \brief Returns the element at indices (\em colIndex). The row index is
     *        determined by the corresponding row thread id. Note that
     *        0 <= \em colIndex< \em tile_width must hold.
     *
     * @param colIndex The column index of the requested element.
     * @return The requested element.
     */
    MSL_USERFUNC
    T get(int index)
    {
#ifdef __CUDA_ARCH__
      return _tile[index*tile_width + threadIdx.x];
#else
      return _tile[index*mLocal];
#endif
    }

  private:
    T* _tile;
    int mLocal;
  };

public:
  /**
   * \brief Constructor. Gathers all pointers (CPU + GPUs) pointing to a local
   *        partition of a given \em DMatrix.
   *
   * @param da The distributed matrix whose local partition will be accessed.
   * @param gpu_dist Specifies the distribution among GPUs. Default is distributed.
   *                 May also be copy distributed, in this case the local partition
   *                 of a distributed matrix is copy distributed among all GPUs.
   */
  LMatrix(DMatrix<T>& dm, Distribution gpu_dist = Distribution::DIST);

  /**
   * \brief Virtual destructor.
   */
  virtual ~LMatrix();

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
   * \brief Returns the number of rows.
   *
   * @return The number of rows.
   */
  MSL_USERFUNC
  int getRows() const;

  /**
   * \brief Returns the number of columns.
   *
   * @return The number of columns.
   */
  MSL_USERFUNC
  int getCols() const;

  /**
   * \brief Returns a pointer to row with index \em rowIndex. Uses a local index. Note that
   *        0 <= \em rowIndex < \em getRows() must hold (not checked for performance
   *        reasons).
   *
   * @param index The index of the requested element.
   * @return A pointer to the requested row of the local partition.
   */
  MSL_USERFUNC
  T* operator[](int rowIndex) const;

  /**
   * \brief Returns the element at indices (\em row, \em col). Uses global indices. Note that
   *        0 <= \em row < \em getRows() and 0 <= \em col < \em getCols() must hold (not
   *        checked for performance reasons).
   *
   * @param row The row index of the requested element.
   * @param col The column index of the requested element.
   * @return The requested element.
   */
  MSL_USERFUNC
  T get(int row, int col) const;

  /**
   * \brief Returns the shared memory size in bytes.
   *
   * @return The shared memory size in bytes.
   */
  virtual int getSmemSize() const
  {
    if (tile_width == -1)
      return 0;
    else
      return sizeof(T)*tile_width*tile_width;
  }

// TODO: Debug this!
//#ifdef __CUDACC__
//  /**
//   * \brief Returns a shared memory tile.
//   *
//   * Returns a shared memory tile. Must be passed a static shared memory buffer
//   * so that data can be loaded from device main memory to this buffer. A row and
//   * a column index denote which block of a \em LMatrix will be loaded to shared
//   * memory. Note that 0 <= \em rowIndex < \em getRows()/tile_width and
//   * 0 <= \em colIndex < \em getCols()/tile_width must hold. Also note that the
//   * size of the shared memory buffer smem must equal \em tile_width * \em tile_width.
//   *
//   * @param rowIndex The row index of the block to be loaded into shared memory.
//   * @param colIndex The column index of the block to be loaded into shared memory.
//   * @param smem The static shared memory buffer.
//   * @tparam tile_width The size of a dimension of a tile, e.g. number of rows and
//   *                    columns to load into shared memory.
//   */
//  template <class F>
//  MSL_GPUFUNC
//  Tile getTile(int rowIndex, int colIndex, F functor) const
//  {
//    int tw = functor->getTileWidth();
//    extern __shared__ T smem[];
//    if (!valid_smem) {
//      smem_pos = functor->getSmemPosition();
//      valid_smem = 1;
//    }
//    return Tile(&smem[smem_pos], *this, rowIndex, colIndex, tw);
//  }
//#endif

  /**
   * \brief Returns a shared memory tile.
   *
   * Returns a shared memory tile. A tile index denotes, which (row) tile of a
   * \em LMatrix will be loaded to shared memory.
   * Note that 0 <= \em tileIndex < \em getCols()/tile_width must hold.
   *
   * @param tileIndex The index of the tile to be loaded into shared memory.
   * @param rowIndex The row index.
   * @param functor The functor.
   * @tparam F The functor type.
   */
  template <class F>
  MSL_USERFUNC
  RowTile getRowTile(int tileIndex, int rowIndex, F functor) const
  {
    int tw = functor->getTileWidth();
#ifdef __CUDA_ARCH__
    if (!valid_smem) {
      smem_ptr = functor->template getSmemPtr<T>(tw*tw);
      valid_smem = 1;
    }
#else
    smem_ptr = 0;
#endif
    return RowTile(smem_ptr, *this, tileIndex, rowIndex, tw);
  }


  /**
   * \brief Returns a shared memory tile.
   *
   * Returns a shared memory tile. A tile index denotes, which (column) tile of a
   * \em LMatrix will be loaded to shared  memory.
   * Note that 0 <= \em tileIndex < \em getRows()/tile_width must hold.
   *
   * @param tileIndex The index of the tile to be loaded into shared memory.
   * @param colIndex The column index.
   * @param functor The functor.
   * @tparam F The functor type.
   */
  template <class F>
  MSL_USERFUNC
  ColTile getColTile(int tileIndex, int colIndex, F functor) const
  {
    int tw = functor->getTileWidth();
#ifdef __CUDA_ARCH__
    if (!valid_smem) {
      smem_ptr = functor->template getSmemPtr<T>(tw*tw);
      valid_smem = 1;
    }
#else
    smem_ptr = 0;
#endif
    return ColTile(smem_ptr, *this, tileIndex, colIndex, tw);
  }

  template <int tile_width>
  MSL_USERFUNC
  RowTileStatic<tile_width> getRowTile(int tileIndex, int rowIndex, T* smem) const
  {
    return RowTileStatic<tile_width>(smem, *this, tileIndex, rowIndex);
  }

  template <int tile_width>
  MSL_USERFUNC
  ColTileStatic<tile_width> getColTile(int tileIndex, int colIndex, T* smem) const
  {
    return ColTileStatic<tile_width>(smem, *this, tileIndex, colIndex);
  }

  MSL_USERFUNC
  DMatrix<T>* getDMatrix() const
  {
    return dmatrix;
  }

private:
  GPUExecutionPlan<T> current_plan;
  int current_device, colOffset;
  Distribution dist;
  DMatrix<T>* dmatrix;
  mutable T* smem_ptr;
  mutable bool valid_smem;
};

}

#include "../src/lmatrix.cpp"
