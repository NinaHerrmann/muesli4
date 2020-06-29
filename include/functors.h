/*
 * functors.h
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
#include "functor_base.h"
#include "lmatrix.h"

namespace msl {

/**************************************************************************
 * \brief Class Functor represents a unary functor
 *
 * @tparam T Input data type.
 * @tparam R Output data type.
 */
template <typename T, typename R>
class Functor{

public:
  /**
   * \brief Function call operator has to be implemented by the user. 
   *
   * @param value Input for the operator.
   * @return Output of the operator.
   */
  MSL_USERFUNC
  virtual R operator() (T value) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~Functor(){}
};


/**************************************************************************
 * \brief Class Functor2 represents a binary functor
 *
 * @tparam T1 1st input data type.
 * @tparam T2 2nd input data type.
 * @tparam R output data type.
 */
template <typename T1, typename T2, typename R>
class Functor2{

public:
  /**
   * \brief Function call operator has to be implemented by the user. 
   *
   * @param x 1st nput for the operator.
   * @param y 2nd nput for the operator.
   * @return Output of the operator.
   */
  MSL_USERFUNC
  virtual R operator() (T1 x, T2 y) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~Functor2(){}
};

/**************************************************************************
 * \brief Class MMapFunctor represents a functor for the map skeleton of the
 *        distributed matrix.
 *
 * @tparam T Input data type.
 * @tparam R Output data type.
 */
template <typename T, typename R>
class MMapFunctor : public detail::MatrixFunctorBase
{
public:
  /**
   * \brief Function call operator has to be implemented by the user. Here,
   *        the actual function is implemented.
   *
   * @param value Input for the map function.
   * @return Output of the map function.
   */
  MSL_USERFUNC
  virtual R operator() (T value) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~MMapFunctor()
  {
  }
};

/**************************************************************************
 * \brief Class MMapIndexFunctor represents a functor for the mapIndex skeleton of
 *        the distributed matrix.
 *
 * @tparam T Input data type.
 * @tparam R Output data type.
 */
template <typename T, typename R>
class MMapIndexFunctor : public detail::MatrixFunctorBase
{
public:
  /**
   * \brief Function call operator has to be implemented by the user. Here,
   *        the actual function is implemented.
   *
   * @param rowIndex Global row index of the input value.
   * @param colIndex Global column index of the input value.
   * @param value Input for the map function.
   * @return Output of the map function.
   */
  MSL_USERFUNC
  virtual R operator() (int rowIndex, int colIndex, T value) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~MMapIndexFunctor()
  {
  }
};

//************************************************************************
template <typename T>
class PLMatrix;

/**************************************************************************
 * \brief Class MMapStencilFunctor represents a functor for the mapStencil skeleton
 *        of the distributed matrix.
 *
 * @tparam T Input data type.
 * @tparam R Output data type.
 */
template <typename T, typename R>
class MMapStencilFunctor : public detail::MatrixFunctorBase
{
public:
  /**
   * \brief Default Constructor.
   *
   * Sets a default stencil size of 1.
   */
  MMapStencilFunctor()
	  : stencil_size(1)
  {
    this->setTileWidth(msl::DEFAULT_TILE_WIDTH);
  }

  /**
   * \brief Function call operator has to be implemented by the user. Here,
   *        the actual function is implemented.
   *
   * @param rowIndex Global row index of the input value.
   * @param colIndex Global column index of the input value.
   * @param input Input for the map stencil function.
   * @return Output of the map stencil function.
   */
  MSL_USERFUNC
  virtual R operator() (int rowIndex, int colIndex, const PLMatrix<T>& input) const = 0;

  /**
   * \brief Returns the stencil size.
   *
   * @return The stencil size.
   */
  int getStencilSize()
  {
    return stencil_size;
  }

  /**
   * \brief Sets the stencil size.
   *
   * @param value The new stencil size.
   */
  void setStencilSize(int value)
  {
    stencil_size = value;
  }

  /**
   * \brief Destructor.
   */
  virtual ~MMapStencilFunctor()
  {
  }

protected:
  int stencil_size;
};


/****************************************************************************
 * \brief Class MZipFunctor represents a functor for the zip skeleton of the
 *        distributed matrix.
 *
 * @tparam T1 Input data type of the first distributed matrix.
 * @tparam T2 Input data type of the second distributed matrix.
 * @tparam R Output data type.
 */
template <typename T1, typename T2, typename R>
class MZipFunctor : public detail::MatrixFunctorBase
{
public:
  /**
   * \brief Function call operator has to be implemented by the user. Here,
   *        the actual function is implemented.
   *
   * @param l_value Input value of the first distributed matrix.
   * @param r_value Input value of the second distributed matrix.
   * @return Output of the zip function.
   */
  MSL_USERFUNC
  virtual R operator() (T1 l_value, T2 r_value) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~MZipFunctor()
  {
  }
};

/************************************************************************
 * \brief Class MZipIndexFunctor represents a functor for the zipIndex skeleton
 *        of the distributed matrix.
 *
 * @tparam T1 Input data type of the first distributed matrix.
 * @tparam T2 Input data type of the second distributed matrix.
 * @tparam R Output data type.
 */
template <typename T1, typename T2, typename R>
class MZipIndexFunctor : public detail::MatrixFunctorBase
{
public:
  /**
   * \brief Function call operator has to be implemented by the user. Here,
   *        the actual function is implemented.
   *
   * @param rowIndex Global row index of the input values.
   * @param colIndex Global column index of the input values.
   * @param l_value Input value of the first distributed matrix.
   * @param r_value Input value of the second distributed matrix.
   * @return Output of the zipIndex function.
   */
  MSL_USERFUNC
  virtual R operator() (int rowIndex, int colIndex, T1 l_value, T2 r_value) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~MZipIndexFunctor()
  {
  }
};

/************************************************************************
 * \brief Class MFoldFunctor represents a functor for the fold skeleton
 *        of the distributed matrix.
 *
 * @tparam T Input data type.
 * @tparam R Output data type.
 */
template <typename T, typename R>
class MFoldFunctor : public detail::MatrixFunctorBase
{
public:
  /**
   * \brief Function call operator has to be implemented by the user. Here,
   *        the actual function is implemented.
   *
   * @param value1 Input value 1.
   * @param value2 Input value 2.
   * @return Output of the fold function.
   */
  MSL_USERFUNC
  virtual R operator() (T value1, T value2) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~MFoldFunctor()
  {
  }
};

/************************************************************************
 * \brief Class AMapFunctor represents a functor for the fold skeleton
 *        of the distributed array.
 *
 * @tparam T Input data type.
 * @tparam R Output data type.
 */
template <typename T, typename R>
class AMapFunctor : public detail::ArrayFunctorBase
{
public:
  /**
   * \brief Function call operator has to be implemented by the user. Here,
   *        the actual function is implemented.
   *
   * @param value Input for the map function.
   * @return Output of the map function.
   */
  MSL_USERFUNC
  virtual R operator() (T value) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~AMapFunctor()
  {
  }
};

/************************************************************************
 * \brief Class AMapIndexFunctor represents a functor for the mapIndex skeleton of
 *        the distributed array.
 *
 * @tparam T Input data type.
 * @tparam R Output data type.
 */
template <typename T, typename R>
class AMapIndexFunctor : public detail::ArrayFunctorBase
{
public:
  /**
   * \brief Function call operator has to be implemented by the user. Here,
   *        the actual function is implemented.
   *
   * @param index Global index of the input value.
   * @param value Input for the map function.
   * @return Output of the map function.
   */
  MSL_USERFUNC
  virtual R operator() (int index, T value) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~AMapIndexFunctor()
  {
  }
};

template <typename T>
class PLArray;

/*********************************************************************************
 * \brief Class AMapStencilFunctor represents a functor for the mapStencil skeleton
 *        of the distributed array.
 *
 * @tparam T Input data type.
 * @tparam R Output data type.
 */
template <typename T, typename R>
class AMapStencilFunctor : public detail::ArrayFunctorBase
{
public:
  /**
   * \brief Default Constructor.
   *
   * Sets a default stencil size of 1.
   */
  AMapStencilFunctor()
    : stencil_size(1)
  {
    this->setTileWidth(msl::DEFAULT_TILE_WIDTH);
  }

  /**
   * \brief Function call operator has to be implemented by the user. Here,
   *        the actual function is implemented.
   *
   * @param index Global index of the input value.
   * @param input Input for the map stencil function.
   * @return Output of the map stencil function.
   */
  MSL_USERFUNC
  virtual R operator() (int index, const PLArray<T>& input) const = 0;

  /**
   * \brief Returns the stencil size.
   *
   * @return The stencil size.
   */
  int getStencilSize()
  {
    return stencil_size;
  }

  /**
   * \brief Sets the stencil size.
   *
   * @param value The new stencil size.
   */
  void setStencilSize(int value)
  {
    stencil_size = value;
  }

  /**
   * \brief Destructor.
   */
  virtual ~AMapStencilFunctor()
  {
  }
protected:
  int stencil_size;
};

/***************************************************************************
 * \brief Class AZipFunctor represents a functor for the zip skeleton of the
 *        distributed array.
 *
 * @tparam T1 Input data type of the first distributed array.
 * @tparam T2 Input data type of the second distributed array.
 * @tparam R Output data type.
 */
template <typename T1, typename T2, typename R>
class AZipFunctor : public detail::ArrayFunctorBase
{
public:
  /**
   * \brief Function call operator has to be implemented by the user. Here,
   *        the actual function is implemented.
   *
   * @param l_value Input value of the first distributed array.
   * @param r_value Input value of the second distributed array.
   * @return Output of the zip function.
   */
  MSL_USERFUNC
  virtual R operator() (T1 l_value, T2 r_value) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~AZipFunctor()
  {
  }
};

/***********************************************************************************
 * \brief Class AZipIndexFunctor represents a functor for the zipIndex skeleton
 *        of the distributed array.
 *
 * @tparam T1 Input data type of the first distributed array.
 * @tparam T2 Input data type of the second distributed array.
 * @tparam R Output data type.
 */
template <typename T1, typename T2, typename R>
class AZipIndexFunctor : public detail::ArrayFunctorBase
{
public:
  /**
   * \brief Function call operator has to be implemented by the user. Here,
   *        the actual function is implemented.
   *
   * @param index Global index of the input values.
   * @param l_value Input value of the first distributed array.
   * @param r_value Input value of the second distributed array.
   * @return Output of the zipIndex function.
   */
  MSL_USERFUNC
  virtual R operator() (int index, T1 l_value, T2 r_value) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~AZipIndexFunctor()
  {
  }
};

/***********************************************************************************
 * \brief Class AFoldFunctor represents a functor for the fold skeleton
 *        of the distributed array.
 *
 * @tparam T Input data type.
 * @tparam R Output data type.
 */
template <typename T, typename R>
class AFoldFunctor : public detail::ArrayFunctorBase
{
public:
  /**
   * \brief Function call operator has to be implemented by the user. Here,
   *        the actual function is implemented.
   *
   * @param value1 Input value 1.
   * @param value2 Input value 2.
   * @return Output of the fold function.
   */
  MSL_USERFUNC
  virtual R operator() (T value1, T value2) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~AFoldFunctor()
  {
  }
};

/***********************************************************************************
 * \brief Class FarmFunctor represents a functor for the farm skeleton.
 *
 * @tparam T Input data type.
 * @tparam R Output data type.
 */
template <typename T, typename R>
class FarmFunctor : public detail::FarmFunctorBase
{
public:
  /**
   * \brief Function call operator has to be implemented by the user. Here,
   *        the actual function is implemented.
   *
   * @param value Input value.
   * @return Output value..
   */
  MSL_USERFUNC
  virtual R operator() (T value) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~FarmFunctor()
  {
  }
};

} // namespace msl
