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

#include <type_traits>

#include "muesli.h"
#include "detail/exception.h"

#include "detail/conversion.h"
#include "detail/exec_plan.h"
/*#include "plmatrix.h"*/
#include <utility>

#include "argtype.h"
#include "detail/functor_base.h"
#include "plcube.h"

namespace msl {

/**************************************************************************
 * \brief Class Functor represents a unary functor
 *
 * @tparam T Input data type.
 * @tparam R Output data type.
 */
template <typename T, typename R> class Functor {

public:
  /**
   * \brief Function call operator has to be implemented by the user.
   *
   * @param value Input for the operator.
   * @return Output of the operator.
   */
  MSL_USERFUNC
  virtual R operator()(T value) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~Functor() {}
};

/**************************************************************************
 * \brief Class Functor2 represents a binary functor
 *
 * @tparam T1 1st input data type.
 * @tparam T2 2nd input data type.
 * @tparam R output data type.
 */
template <typename T1, typename T2, typename R> class Functor2 {

public:
  /**
   * \brief Function call operator has to be implemented by the user.
   *
   * @param x 1st nput for the operator.
   * @param y 2nd nput for the operator.
   * @return Output of the operator.
   */
  MSL_USERFUNC
  virtual R operator()(T1 x, T2 y) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~Functor2() {}
};

/**************************************************************************
 * \brief Class Functor3 represents a 3ary functor
 *
 * @tparam T1 1st input data type.
 * @tparam T2 2nd input data type.
 * @tparam T3 3rd input data type.
 * @tparam R output data type.
 */
template <typename T1, typename T2, typename T3, typename R> class Functor3 {

public:
  /**
   * \brief Function call operator has to be implemented by the user.
   *
   * @param x 1st nput for the operator.
   * @param y 2nd nput for the operator.
   * @param z 3rd nput for the operator.
   * @return Output of the operator.
   */
  MSL_USERFUNC
  virtual R operator()(T1 x, T2 y, T3 z) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~Functor3() {}
};

/**************************************************************************
 * \brief Class Functor4 represents a 4ary functor
 *
 * @tparam T1 1st input data type.
 * @tparam T2 2nd input data type.
 * @tparam T3 3rd input data type.
 * @tparam T4 4th input data type.
 * @tparam R output data type.
 */
template <typename T1, typename T2, typename T3, typename T4, typename R>
class Functor4 {

public:
  /**
   * \brief Function call operator has to be implemented by the user.
   *
   * @param x 1st input for the operator.
   * @param y 2nd input for the operator.
   * @param z 3rd input for the operator.
   * @param v 4th input for the operator.
   * @return Output of the operator.
   */
  MSL_USERFUNC
  virtual R operator()(T1 x, T2 y, T3 z, T4 v) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~Functor4() {}
};
/**************************************************************************
 * \brief Class Functor4 represents a 4ary functor
 *
 * @tparam T1 1st input data type.
 * @tparam T2 2nd input data type.
 * @tparam T3 3rd input data type.
 * @tparam T4 4th input data type.
 * @tparam T5 4th input data type.
 * @tparam R output data type.
 */
template <typename T1, typename T2, typename T3, typename T4,typename T5, typename R>
class Functor5 {

public:
    /**
     * \brief Function call operator has to be implemented by the user.
     *
     * @param x 1st input for the operator.
     * @param y 2nd input for the operator.
     * @param z 3rd input for the operator.
     * @param v 4th input for the operator.
     * @param l 5th input for the operator.
     * @return Output of the operator.
     */
    MSL_USERFUNC
    virtual R operator()(T1 x, T2 y, T3 z, T4 v, T5 l) const = 0;

    /**
     * \brief Destructor.
     */
    virtual ~Functor5() {}
};
/**************************************************************************
 * \brief Class Functor4P represents a 4ary functor with 2 Pointers
 *
 * @tparam T1 1st input data type.
 * @tparam T2 2nd input data type.
 * @tparam T3 3rd input data type.
 * @tparam T4 4th input data type.
 * @tparam R output data type.
 */
template <typename T1, typename T2, typename T3, typename T4, typename R>
class Functor4P {

public:
  /**
   * \brief Function call operator has to be implemented by the user.
   *
   * @param x 1st input for the operator.
   * @param y 2nd input for the operator.
   * @param z 3rd input for the operator.
   * @param v 4th input for the operator.
   * @return Output of the operator.
   */
  MSL_USERFUNC
  virtual void operator()(T1 x, T2 y, T3* z, T4* v) const = 0;

  /**
   * \brief Destructor.
   */
  virtual ~Functor4P() {}
};

template <typename T> class DM;
/*template <typename T> class PLMatrix;*/

namespace NVF {
    class NeutralValueFunctor2 : public Functor2<int, int, int> {
    public:
        NeutralValueFunctor2(int default_neutral)
                : default_neutral(default_neutral) {}

        MSL_USERFUNC
        int operator()(int x, int y) const { // here, x represents rows
            return default_neutral;
        }

    private:
       int default_neutral;
    };
}
/**
 * \brief Class MMapStencilFunctor represents a functor for the mapStencil
 * skeleton of the distributed matrix.
 *
 * @tparam T Input data type.
 * @tparam R Output data type.
 */

template <typename T, typename R, typename NeutralValueFunctor>
class MMapStencilFunctor : public detail::MatrixFunctorBase {
public:
  /**
   * \brief Default Constructor.
   *
   * Sets a default stencil size of 1.
   */
  MMapStencilFunctor() : stencil_size(1) {
    this->setTileWidth(msl::DEFAULT_TILE_WIDTH);
  }

  /**
   * \brief Returns the stencil size.
   *
   * @return The stencil size.
   */
  int getStencilSize() { return stencil_size; }

  /**
   * \brief Sets the stencil size.
   *
   * @param value The new stencil size.
   */
  void setStencilSize(int value) { stencil_size = value; }
  /**
   * \brief Sets the stencil size.
   *
   * @param value The new stencil size.
   */
  //void setNVF(NeutralValueFunctor &nv) { nvf = nv; }

  /**
   * \brief Destructor.
   */
  virtual ~MMapStencilFunctor() {}

protected:
  int stencil_size;
};

template <typename T, typename R>
class DMMapStencilFunctor : public detail::MatrixFunctorBase {
public:
    /**
     * \brief Default Constructor.
     *
     * Sets a default stencil size of 1.
     */
    DMMapStencilFunctor() : stencil_size(1) {
        this->setTileWidth(msl::DEFAULT_TILE_WIDTH);
    }

    /**
     * \brief Returns the stencil size.
     *
     * @return The stencil size.
     */
    int getStencilSize() { return stencil_size; }

    /**
     * \brief Sets the stencil size.
     *
     * @param value The new stencil size.
     */
    void setStencilSize(int value) { stencil_size = value; }
    /**
     * \brief Sets the stencil size.
     *
     * @param value The new stencil size.
     */
    bool getSharedMemory() { return shared_memory; }
/**
     * \brief Sets the stencil size.
     *
     * @param value The new stencil size.
     */
    void setSharedMemory(bool value) { shared_memory = value; }

    /**
     * \brief Destructor.
     */
    virtual ~DMMapStencilFunctor() {}

protected:
    int stencil_size;
    bool shared_memory = false;
};
template <typename T, typename R, typename NeutralValueFunctor>
class DMNVFMapStencilFunctor : public detail::MatrixFunctorBase {
    public:
        /**
         * \brief Default Constructor.
         *
         * Sets a default stencil size of 1.
         */
        DMNVFMapStencilFunctor() : stencil_size(1) {
            this->setTileWidth(msl::DEFAULT_TILE_WIDTH);
        }

        /**
         * \brief Returns the stencil size.
         *
         * @return The stencil size.
         */
        int getStencilSize() { return stencil_size; }

        /**
         * \brief Sets the stencil size.
         *
         * @param value The new stencil size.
         */
        void setStencilSize(int value) { stencil_size = value; }
        /**
         * \brief Sets the stencil size.
         *
         * @param value The new stencil size.
         */
        bool getSharedMemory() { return shared_memory; }
/**
         * \brief Sets the stencil size.
         *
         * @param value The new stencil size.
         */
        void setSharedMemory(bool value) { shared_memory = value; }

        /**
         * \brief Destructor.
         */
        virtual ~DMNVFMapStencilFunctor() {}

    protected:
        int stencil_size;
        bool shared_memory = false;
    };

template <typename T>
using DCMapStencilFunctor = T(*)(const PLCube<T> &cs, int x, int y, int z);

} // namespace msl
