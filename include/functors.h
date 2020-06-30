/*
 * functors.h
 *
 *      Author: Herbert Kuchen <kuchen@uni-muenster.de>
 * 
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2020 Herbert Kuchen <kuchen@uni-muenster.de>
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
} // namespace msl
