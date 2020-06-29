/*
 * functor_base.h
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

#include <vector>
#include "lmatrix.h"

namespace msl {
/**
 * \brief Namespace \em detail contains internally used classes. The user of Muesli
 *        should not get in touch with the contents of this namespace.
 */
namespace detail {

/**
 * \brief Class \em FunctorBase represents the base class for all functor classes.
 *
 * Class \em FunctorBase represents the base class for all functor classes. It
 * includes methods to configure a user implemented functor. Additionally it acts as
 * an observer for additional arguments.
 */
class FunctorBase
{
protected:
  std::vector<ArgumentType*> args;
  int tile_width;
  bool local_indices;
  mutable bool valid_smem = 0;
  mutable void* smem;

public:
  /**
   * \brief Default constructor.
   */
  FunctorBase()
    : tile_width(-1), local_indices(false), smem(0)
  {
  }

  /**
   * \brief Checks whether indices are local or global.
   *
   * @return True if indices passed to the functor are local indices, false otherwise.
   */
  bool useLocalIndices() const
  {
    return local_indices;
  }

  /**
   * \brief Use this function to configure your functor to use local indices instead
   *        of global indices.
   *
   * @param value True for local indices, false for global indices.
   */
  void setLocalIndices(bool value)
  {
    local_indices = value;
  }

  /**
   * \brief Returns the tile_width. If tiling is not used, -1 will be returned.
   *
   * @return The tile width.
   */
  MSL_USERFUNC
  int getTileWidth() const
  {
    return tile_width;
  }

  /**
   * \brief Sets the tile width.
   *
   * @param value The tile_width.
   */
  void setTileWidth(int value)
  {
    for (ArgumentType*& arg : args) {
      arg->setTileWidth(value);
    }
    tile_width = value;
  }

  /**
   * \brief Notifies all observed objects (additional arguments) to update.
   */
  void notify()
  {
    for (size_t i = 0; i < args.size(); i++) {
      args.at(i)->update();
    }
  }

  /**
   * \brief Adds an additional argument to the functor.
   *
   * @param arg The additional argument.
   */
  void addArgument(ArgumentType* arg)
  {
    arg->setTileWidth(tile_width);
    args.push_back(arg);
  }

  /**
   * \brief Returns the size in bytes of shared memory to be used.
   *
   * @return The size in bytes of shared memory.
   */
  int getSmemSize() const
  {
    int size = 0;
    for (ArgumentType* arg : args) {
      size += arg->getSmemSize();
    }
    return size;
  }

  /**
   * \brief Returns a pointer to an array declared in dynamic shared memroy.
   *
   * @return (Dynamic) shared memory pointer.
   */
  template <class T>
  MSL_GPUFUNC
  T* getSmemPtr(int size) const
  {
    T* res = 0;
#ifdef __CUDA_ARCH__
    if (!valid_smem) {
      extern __shared__ T smem_tmp[];
      smem = smem_tmp;
      valid_smem = 1;
    }
    res = static_cast<T*>(smem);
    smem = static_cast<T*>(smem) + size;
#endif
    return res;
  }

  /**
   * \brief Destructor.
   */
  virtual ~FunctorBase()
  {
  }
};

/**
 * \brief Class MatrixFunctorBase represents the base class for all functors to be
 *        used with any distributed matrix skeletons.
 */
class MatrixFunctorBase : public FunctorBase
{
public:
  /**
   * \brief Initializes its attributes.
   *
   * @param nl Number of local rows.
   * @param ml Number of local columns.
   * @param fr Index of first row of the local partition.
   * @param fc Index of first column of the local partition.
   */
  void init(int nl, int ml, int fr, int fc)
  {
    nLocal = nl;
    mLocal = ml;
    firstRow = fr;
    firstCol = fc;
  }

protected:
  int nLocal, mLocal, firstRow, firstCol;
};

/**
 * \brief Class ArrayFunctorBase represents the base class for all functors to be
 *        used with any distributed array skeletons.
 */
class ArrayFunctorBase : public FunctorBase
{
public:
  /**
   * \brief Initializes its attributes.
   *
   * @param nl Number of local rows.
   * @param f Index of first element of the local partition.
   */
  void init(int nl, int f)
  {
    nLocal = nl;
    first = f;
  }

protected:
  int nLocal, first;
};

/**
 * \brief Class FarmFunctorBase represents the base class for all functors to be
 *        used with the farm skeleton.
 */
class FarmFunctorBase : public FunctorBase
{
  // no extra functionality yet
};

}
}








