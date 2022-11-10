/*
 * dc.h
 *
 *      Author: Nina Herrmann <nina.herrmann@uni-muenster.de>
 *              Herbert Kuchen <kuchen@uni-muenster.de>
 *
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2021-... 	Nina Herrmann <nina.herrmann@uni-muenster.de>,
 *                	Herbert Kuchen <kuchen@uni-muenster.de.
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
#include "ds.h"

#ifndef MUESLI_DC_H
#define MUESLI_DC_H
#include <type_traits>

#include "muesli.h"
#include "detail/exception.h"
#include "functors.h"
#include "detail/conversion.h"
#include "detail/exec_plan.h"
#include <utility>
#include "plmatrix.h"

#ifdef __CUDACC__
#include "detail/copy_kernel.cuh"
#include "detail/fold_kernels.cuh"
#include "detail/map_kernels.cuh"
#include "detail/properties.cuh"
#include "detail/zip_kernels.cuh"

#endif

namespace msl {
    /**
     * \brief Class DC represents a distributed Cuboid.
     *
     * A distributed Cuboid represents a one-dimensional parallel container and is
     * distributed among all MPI processes the application was started with. It
     * includes data parallel skeletons such as map, mapStencil, and zip as
     * well as variants of these skeletons.
     *
     * \tparam T Element type. Restricted to classes without pointer data members.
     */
template <typename T>
class DC : public msl::DS<T>{
public:
  //
  // CONSTRUCTORS / DESTRUCTOR
  //

  /**
   * \brief Default constructor.
   */
  DC();

  /**
   * \brief Creates an empty distributed matrix.
   *
   * @param size Size of the distributed array.
   * @param d Distribution of the distributed array.
   */
  DC(int row, int col, int depth);

  DC(int row, int col, int depth, bool rowComplete);

  /**
   * \brief Creates a distributed matrix with \em size elements equal to
   *        \em initial_value.
   *
   * @param size Size of the distributed array.
   * @param initial_value Initial value for all elements.
   */
  DC(int row, int col, int depth, const T &v);

  /**
   * @brief Creates a distributed matrix with \em size elements equal to
   *        \em initial_value.
   * @param col number or columns
   * @param row Number of rows
   * @param initial_value Initial value of the matrix
   * @param rowComplete if true, the matrix will be distributed between nodes in
   * full rows. If mapStencil will be used, this option needs to be set to true.
   */
  DC(int row, int col, int depth, const T &v, bool rowComplete);

//#pragma region Rule of five
  /**
   * For more details see https://cpppatterns.com/patterns/rule-of-five.html
   * The 5 functions here are needed to perform operations such as std::move.
   * See examples/jacobi.cu for a usage reference.
   */

  /**
   * @brief Copy constructor. Fully copies the object and it's data.
   *
   */
  DC(const DC<T> &other);

  /**
   * @brief Move constructor. Transfers ownership of resources allocated by \em
   * other to the object that is being created
   *
   * @param other
   */
  DC(DC<T> &&other) noexcept ;

  /**
   * @brief Copy assignment operator. Works the same as the copy constructor.
   *
   * @param other
   * @return DC<T>&
   */
  DC<T> &operator=(const DC<T> &other) noexcept;

  /**
   * @brief Move assignment operator. This assigs the object defined in \em
   * other to the left hand side of the operation without creating copies
   *
   * @param other
   * @return DC<T>&
   */
  DC<T> &operator=(DC<T> &&other) noexcept;

  /**
   * \brief Destructor.
   */
  ~DC();

//#pragma endregion


  //
  // SKELETONS / COMPUTATION
  //

  // SKELETONS / COMPUTATION / MAP

  /**
   * \brief Replaces each element a[i] of the distributed array with f(i, a[i]).
   *        Note that besides the element itself also its index is passed to the
   *        functor.
   *
   * @param f The mapIndex functor, must be of type \em AMapIndexFunctor.
   * @tparam MapIndexFunctor Functor type.
   */
  template <typename MapIndexFunctor> void mapIndexInPlace(MapIndexFunctor &f);

  /**
   * \brief Returns a new distributed array with a_new[i] = f(i, a[i]). Note
   *        that besides the element itself also its index is passed to the
   * functor.
   *
   * @param f The mapIndex functor, must be of type \em AMapIndexFunctor.
   * @tparam MapIndexFunctor Functor type.
   * @tparam R Return type.
   * @return The newly created distributed array.
   */
  template <typename MapIndexFunctor>
  void mapIndex(MapIndexFunctor &f, DC<T> &b); // should be return type DA<R>; debug
/*
    TODO Stencil Functor
  */
/**
   * \brief Replaces each element a[i] of the distributed array with f(i, a).
   *        Note that the index i and the local partition is passed to the
   *        functor.
   *
   * @param f The mapStencil functor, must be of type \em AMapStencilFunctor.
   * @tparam MapStencilFunctor Functor type.
   *//*

  template <typename MapStencilFunctor, typename NeutralValueFunctor>
  void mapStencilInPlace(MapStencilFunctor &f, NeutralValueFunctor &neutral_value_functor);
  template <typename T2, typename MapStencilFunctor, typename NeutralValueFunctor>
  void mapStencilMM(DC<T2> &result, MapStencilFunctor &f, NeutralValueFunctor &neutral_value_functor);
*/

  // SKELETONS / COMPUTATION / ZIP

  /**
   * \brief Replaces each element a[i] of the distributed array with f(i, a[i],
   * b[i]). Note that besides the elements themselves also the index is passed
   * to the functor.
   *
   * @param f The zipIndex functor, must be of type \em AZipIndexFunctor.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam ZipIndexFunctor Functor type.
   */
  template <typename T2, typename ZipIndexFunctor>
  void zipIndexInPlace(DC<T2> &b, ZipIndexFunctor &f);
  /**
   * \brief Replaces each element a[i] of the distributed array with f(i, a[i],
   * *b[]). Note that besides the elements themselves also the index is passed
   * to the functor. Also note that the whole column is passed.
   *
   * @param f The zipIndex functor, must be of type \em AZipIndexFunctor.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam ZipIndexFunctor Functor type.
   */
  template <typename T2, typename ZipIndexFunctor>
  void crossZipIndexInPlace(DC<T2> &b, ZipIndexFunctor &f);

  /**
   * \brief Non-inplace variant of the zipIndex skeleton.
   *
   * @param f The zipIndex functor, must be of type \em AZipIndexFunctor.
   * @tparam R Return type.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam ZipIndexFunctor Functor type.
   * @return The newly created distributed array.
   */
  template <typename T2, typename ZipIndexFunctor>
  void zipIndex(DC<T2> &b, DC <T2> &c, ZipIndexFunctor &f);

  /**
   * \brief TODO Replaces each element a[i] of the distributed array with f(i, a).
   *        Note that the index i and the local partition is passed to the
   *        functor.
   *
   * @param result DC to save the result
   * @param f MapStencilFuncotr
   * @param neutral_value_functor NeutralValueFunctor
   */
  template<typename MapStencilFunctor, typename NeutralValueFunctor>
  void mapStencilInPlace(MapStencilFunctor &f, NeutralValueFunctor &neutral_value_functor);
  /**
  * @brief TODO Non-inplace variant of the mapStencil skeleton.
  *
  * @tparam MapStencilFunctor Functor for the Stencil Calculation
  * @tparam NeutralValueFunctor Functor to return the NV
  * @param result DC to save the result
  * @param f MapStencilFuncotr
  * @param neutral_value_functor NeutralValueFunctor
  */
  template<typename MapStencilFunctor, typename NeutralValueFunctor>
  void mapStencil(DC<T> &result, MapStencilFunctor &f, NeutralValueFunctor &neutral_value_functor);

  //
  // SKELETONS / COMMUNICATION
  //

  // SKELETONS / COMMUNICATION / BROADCAST PARTITION

  // SKELETONS / COMMUNICATION / PERMUTE PARTITION

  /**
   * \brief Permutes the partitions of the distributed array according to the
   *        given function \em f. \em f must be bijective and return the ID
   *        of the new process p_i to store the partition, with 0 <= i < np.
   *
   * @param f bijective functor
   * @tparam F Function type for \em f.
   */
  template <typename Functor> inline void permutePartition(Functor &f);

  /**
   * \brief Permutes the partitions of the distributed array according to the
   *        given function \em f. \em f must be bijective and return the the
   *        ID of the new process p_i to store the partition, with 0 <= i < np.
   *
   * @param f The bijective function.
   */
   inline void permutePartition(int (*f)(int));

  //
  // GETTERS AND SETTERS
  //

  /**
   * \brief Returns the element at the given global index \em row, col.
   *
   * @param row The row index.
   * @param col The col index.
   * @return The element at the given global index.
   */
    MSL_USERFUNC
    T get3D(int row, int col, int depth, int gpu) const;
    /**
     * \brief Returns the element at the given row \em row and column \em column.
     *
     * @param row The global row.
     * @param column The global column.
     * @return The element at the given global index.
     */
    T get_shared(int row, int column) const;

      /**
      * \brief Override DS function to initialize depth, col, row.
      */
    void initGPUs();

  /**
   * \brief Manually download the local partition from GPU memory.
   */
  //void downloadupperpart(int paddingsize);
  /**
   * \brief Manually download the local partition from GPU memory.
   */
  //void downloadlowerpart(int paddingsize);

  /**
   * \brief Prints the local partion of the root processor of the distributed
   * array to standard output. Optionally, the user may pass a description that
   * will be printed with the output. Just useful for debugging.
   *
   * @param descr The description string.
   */
  void showLocal(const std::string &descr);

  /**
   * \brief Prints the distributed array to standard output. Optionally, the
   * user may pass a description that will be printed with the output.
   *
   * @param descr The description string.
   */
  void show(const std::string &descr = std::string());



private:
  //
  // Attributes
  //

  // local partition

  // Number of local rows. If the distribution is not row complete, a row will
  // be counted if one or more elements from that row are part of this
  // partition.
  int nlocalRows;

  // number of cols
  int ncol;
  // number of rows
  int nrow;
  // depth
  int depth;
  // first (global) row in local partition
  int firstRow;

  // Indicates whether the matrix should be distributed in full rows between
  // the nodes. The map stencil functor needs this type of distribution
  bool rowComplete;

};

} // namespace msl

#include "../src/dc.cpp"
#endif