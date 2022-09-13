/*
 * DMatrix.h
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

#include "muesli.h"
#include "exception.h"
#include "functors.h"
#include "lmatrix.h"
#include "plmatrix.h"
#include "exec_plan.h"
#ifdef __CUDACC__
#include "map_kernels.cuh"
#include "zip_kernels.cuh"
#include "fold_kernels.cuh"
#include "copy_kernel.cuh"
#include "properties.cuh"
#endif

namespace msl {

/**
 * \brief Class DMatrix represents a distributed matrix.
 *
 * A distributed matrix represents a parallel two-dimensional container and is
 * distributed among all MPI processes the application was started with. It
 * includes data parallel skeletons such as map, mapStencil, zip, and fold as
 * well as variants of these skeletons.
 *
 * \tparam T Element type. Restricted to classes without pointer data members.
 */
template<typename T>
class DMatrix {
 public:

  //
  // CONSTRUCTORS / DESTRUCTOR
  //

  /**
   * \brief Default constructor.
   */
  DMatrix();

  /**
   * \brief Creates an empty distributed matrix with \em rows * \em cols elements.
   *
   * @param n0 Number of rows.
   * @param m0 Number of columns.
   * @param rows Number of blocks per column.
   * @param cols Number of blocks per row.
   * @param d Distribution of the distributed matrix.
   */
  DMatrix(int n0, int m0, int rows, int cols, Distribution d = DIST);

  /**
   * \brief Creates a distributed matrix with \em rows * \em cols elements
   *        equal to \em initial_value.
   *
   * @param n0 Number of rows.
   * @param m0 Number of columns.
   * @param rows Number of blocks per column.
   * @param cols Number of blocks per row.
   * @param initial_value Initial value for all elements.
   * @param d Distribution of the distributed matrix.
   */
  DMatrix(int n0, int m0, int rows, int cols, const T& initial_value,
          Distribution d = DIST);

  /**
   * \brief Creates a distributed matrix with \em rows * \em cols elements.
   *        Elements are copied from \em initial_array. Note that the length of
   *        \em initial_matrix must equal \em rows * \em cols.
   *
   * @param n0 Number of rows.
   * @param m0 Number of columns.
   * @param rows Number of blocks per column.
   * @param cols Number of blocks per row.
   * @param initial_matrix Initial matrix to copy elements from.
   * @param d Distribution of the distributed matrix.
   * @param root_init true: the array is only available for the root process, false: array is available for all processes
   */
  DMatrix(int n0, int m0, int rows, int cols, T* const initial_matrix,
          Distribution d = DIST, bool root_init = false);

//  DMatrix(int n0, int m0, int rows, int cols, const T* const * const initial_matrix, Distribution d = DIST);

  /**
   * \brief Creates a distributed matrix with \em rows * \em cols elements.
   *        Initializes all elements via the given function \em f. Note that global
   *        indices are pass to this function as arguments.
   *
   * @param n0 Number of rows.
   * @param m0 Number of columns.
   * @param rows Number of blocks per column.
   * @param cols Number of blocks per row.
   * @param f Function to initialize the elements of the distributed matrix.
   * @param d Distribution of the distributed matrix.
   */
  DMatrix(int n0, int m0, int rows, int cols, T (*f)(int, int), Distribution d =
              DIST);

  /**
   * \brief Creates a distributed matrix with \em rows * \em cols elements.
   *        Initializes all elements via the given functor \em f. Note that global
   *        indices are pass to this function as arguments.
   *
   * @param n0 Number of rows.
   * @param m0 Number of columns.
   * @param rows Number of blocks per column.
   * @param cols Number of blocks per row.
   * @param f Functor to initialize the elements of the distributed matrix.
   * @param d Distribution of the distributed matrix.
   */
  template<typename F2>
  DMatrix(int n0, int m0, int rows, int cols, const F2& f,
          Distribution d = DIST);

  /**
   * \brief Creates an empty copy distributed distributed matrix with
   *        \em rows * \em cols elements.
   *
   * @param n0 Number of rows.
   * @param m0 Number of columns.
   */
  DMatrix(int n0, int m0);

  /**
   * \brief Creates a copy distributed distributed matrix with \em rows * \em cols
   *        elements equal to \em initial_value.
   *
   * @param n0 Number of rows.
   * @param m0 Number of columns.
   * @param initial_value Initial value for all elements.
   */
  DMatrix(int n0, int m0, const T& initial_value);

  /**
   * \brief Creates a copy distributed distributed matrix with \em rows * \em cols
   *        elements. Elements are copied from \em initial_array. Note that the
   *        length of \em initial_matrix must equal \em rows * \em cols.
   *
   * @param n0 Number of rows.
   * @param m0 Number of columns.
   * @param initial_matrix Initial matrix to copy elements from.
   */
  DMatrix(int n0, int m0, T* const initial_matrix);

//  DMatrix(int n0, int m0, const T* const * const initial_matrix);

  /**
   * \brief Creates a copy distributed distributed matrix with \em rows * \em cols
   *        elements. Initializes all elements via the given function \em f. Note
   *        that global indices are pass to this function as arguments.
   *
   * @param n0 Number of rows.
   * @param m0 Number of columns.
   * @param f Function to initialize the elements of the distributed matrix.
   */
  DMatrix(int n0, int m0, T (*f)(int, int));

  /**
   * \brief Creates a copy distributed distributed matrix with \em rows * \em cols
   *        elements. Initializes all elements via the given functor \em f. Note
   *        that global indices are pass to this function as arguments.
   *
   * @param n0 Number of rows.
   * @param m0 Number of columns.
   * @param f Functor to initialize the elements of the distributed matrix.
   * @tparam F2 Functor type.
   */
  template<typename F2>
  DMatrix(int n0, int m0, const F2& f);

  /**
   * \brief Copy constructor.
   */
  DMatrix(const DMatrix<T>& cs);

  /**
   * \brief Destructor.
   */
  ~DMatrix();

  // ASSIGNMENT OPERATOR
  /**
   * \brief Assignment operator.
   */
  DMatrix<T>& operator=(const DMatrix<T>& rhs);

  //
  // FILL
  //

  /**
   * \brief Initializes the elements of the distributed matrix with the value \em
   *        value.
   *
   * @param value The value.
   */
  void fill(const T& value);

  /**
   * \brief Initializes the elements of the distributed matrix with the elements
   *        of the given array of values. Note that the length of \em values must
   *        match the size of the distributed matrix (not checked).
   *
   * @param values The array of values.
   */
  void fill(T* const values);

  /**
   * \brief Initializes the elements of the distributed matrix with the elements
   *        of the given 2D array of values. Note that the length of \em values must
   *        match the size of the distributed matrix (not checked).
   *
   * @param values The array of values.
   */
  void fill(T** const values);

  /**
   * \brief Initializes the elements of the distributed matrix via the given
   *        function \em f. Note that global indices are pass to this function
   *        as arguments.
   *
   * @param f The initializer function.
   */
  void fill(T (*f)(int, int));

  /**
   * \brief Initializes the elements of the distributed matrix via the given
   *        functor \em f. Note that global indices are pass to this functor
   *        as arguments.
   *
   * @param f The initializer functor.
   * @tparam F2 Functor type. Must expect 2 arguments: row index and column index.
   */
  template<typename F2>
  void fill(const F2& f);

  /**
   * \brief Initializes the elements of the distributed matrix with the elements
   *        of the given array of values. Note that the length of \em values must
   *        match the size of the distributed matrix (not checked).
   *
   * @param values The array of values.
   */
  void fill_root_init(T* const values);

  //
  // SKELETONS / COMPUTATION
  //

  // SKELETONS / COMPUTATION / MAP

  /**
   * \brief Replaces each element m[i][j] of the distributed matrix with f(m[i][j]).
   *
   * @param f The map functor, must be of type \em MMapFunctor.
   * @tparam MapFunctor Functor type.
   */
  template<typename MapFunctor>
  void mapInPlace(MapFunctor& f);

  /**
   * \brief Replaces each element m[i][j] of the distributed matrix with f(i, j, m[i][j]).
   *        Note that besides the element itself also its indices are passed to the
   *        functor.
   *
   * @param f The mapIndex functor, must be of type \em MMapIndexFunctor.
   * @tparam MapIndexFunctor Functor type.
   */
  template<typename MapIndexFunctor>
  void mapIndexInPlace(MapIndexFunctor& f);

  /**
   * \brief Returns a new distributed matrix with m_new[i][j] = f(m[i][j]).
   *
   * @param f The map functor, must be of type \em MMapFunctor.
   * @tparam MapFunctor Functor type.
   * @tparam R Return type.
   * @return The newly created distributed matrix.
   */
  template<typename R, typename MapFunctor>
  msl::DMatrix<R> map(MapFunctor& f);

  /**
   * \brief Returns a new distributed matrix with m_new[i] = f(i, j, m[i][j]). Note
   *        that besides the element itself also its indices are passed to the functor.
   *
   * @param f The mapIndex functor, must be of type \em MMapIndexFunctor.
   * @tparam MapIndexFunctor Functor type.
   * @tparam R Return type.
   * @return The newly created distributed matrix.
   */
  template<typename R, typename MapIndexFunctor>
  DMatrix<R> mapIndex(MapIndexFunctor& f);

  /**
   * \brief Replaces each element m[i][j] of the distributed matrix with f(i, j, m).
   *        Note that the index i and the local partition is passed to the
   *        functor.
   *
   * @param f The mapStencil functor, must be of type \em MMapStencilFunctor.
   * @tparam MapStencilFunctor Functor type.
   */
  template<typename MapStencilFunctor>
  void mapStencilInPlace(MapStencilFunctor& f, T neutral_value);

  /**
   * \brief Non-inplace variant of the mapStencil skeleton.
   *
   * @see mapStencilInPlace()
   * @param f The mapStencil functor, must be of type \em MMapStencilFunctor.
   * @tparam MapStencilFunctor Functor type.
   * @tparam R Return type.
   * @return The newly created distributed matrix.
   */
  template<typename R, typename MapStencilFunctor>
  DMatrix<R> mapStencil(MapStencilFunctor& f, T neutral_value);

#ifndef __CUDACC__

  // SKELETONS / COMPUTATION / MAP / INPLACE
  /**
   * \brief Replaces each element m[i][j] of the distributed matrix with f(m[i][j]).
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The map functor, must be a 'curried' function pointer.
   * @tparam F Function type.
   */
  template<typename F>
  void mapInPlace(const msl::Fct1<T, T, F>& f);

  /**
   * \brief Replaces each element m[i][j] of the distributed matrix with f(m[i][j]).
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The mapIndex function.
   */
  void mapInPlace(T (*f)(T));

  // SKELETONS / COMPUTATION / MAP / INDEXINPLACE
  /**
   * \brief Replaces each element m[i][j] of the distributed matrix with f(i, j, m[i][j]).
   *        Note that besides the element itself also its indices are passed to the
   *        functor. Also note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The mapIndex functor, must be a 'curried' function pointer.
   * @tparam F Function type.
   */
  template<typename F>
  void mapIndexInPlace(const msl::Fct3<int, int, T, T, F>& f);

  /**
   * \brief Replaces each element m[i][j] of the distributed array with f(i, j, m[i][j]).
   *        Note that besides the element itself also its indices are passed to the
   *        functor. Also note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The mapIndex function.
   */
  void mapIndexInPlace(T (*f)(int, int, T));

  // SKELETONS / COMPUTATION / MAP
  /**
   * \brief Non-inplace variant of the map skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The map functor, must be a 'curried' function pointer.
   * @tparam R Return type.
   * @tparam F Function type.
   * @return The newly created distributed matrix.
   */
  template<typename R, typename F>
  msl::DMatrix<R> map(const msl::Fct1<T, R, F>& f);

  /**
   * \brief Non-inplace variant of the map skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The map function.
   * @tparam R Return type.
   */
  template<typename R>
  msl::DMatrix<R> map(R (*f)(T));

  // SKELETONS / COMPUTATION / MAP / INDEX
  /**
   * \brief Non-inplace variant of the mapIndex skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The mapIndex functor, must be a 'curried' function pointer.
   * @tparam R Return type.
   * @tparam F Function type.
   * @return The newly created distributed matrix.
   */
  template<typename R, typename F>
  DMatrix<R> mapIndex(const msl::Fct3<int, int, T, R, F>& f);

  /**
   * \brief Non-inplace variant of the mapIndex skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The mapIndex function.
   * @tparam R Return type.
   * @return The newly created distributed matrix.
   */
  template<typename R>
  DMatrix<R> mapIndex(R (*f)(int, int, T));

#endif

  // SKELETONS / COMPUTATION / ZIP

  /**
   * \brief Replaces each element m[i][j] of the distributed array with
   *        f(m[i][j], b[i][j]) with \em b being another distributed matrix of
   *        the same size.
   *
   * @param f The zip functor, must be of type \em MZipFunctor.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam ZipFunctor Functor type.
   */
  template<typename T2, typename ZipFunctor>
  void zipInPlace(DMatrix<T2>& b, ZipFunctor& f);

  /**
   * \brief Replaces each element m[i][j] of the distributed matrix with
   *        f(i, j, m[i][j], b[i][j]). Note that besides the elements
   *        themselves also the indices are passed to the functor.
   *
   * @param f The zipIndex functor, must be of type \em MZipIndexFunctor.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam ZipIndexFunctor Functor type.
   */
  template<typename T2, typename ZipIndexFunctor>
  void zipIndexInPlace(DMatrix<T2>& b, ZipIndexFunctor& f);

  /**
   * \brief Non-inplace variant of the zip skeleton.
   *
   * @param f The zip functor, must be of type \em MZipFunctor.
   * @return The newly created distributed matrix.
   * @tparam R Return type.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam ZipFunctor Functor type.
   */
  template<typename R, typename T2, typename ZipFunctor>
  DMatrix<R> zip(DMatrix<T2>& b, ZipFunctor& f);

  /**
   * \brief Non-inplace variant of the zipIndex skeleton.
   *
   * @param f The zipIndex functor, must be of type \em MZipIndexFunctor.
   * @tparam R Return type.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam ZipIndexFunctor Functor type.
   * @return The newly created distributed matrix.
   */
  template<typename R, typename T2, typename ZipIndexFunctor>
  DMatrix<R> zipIndex(DMatrix<T2>& b, ZipIndexFunctor& f);

#ifndef __CUDACC__

  // SKELETONS / COMPUTATION / ZIP / INPLACE
  /**
   * \brief Replaces each element m[i][j] of the distributed matrix with
   *        f(m[i][j], b[i][j]) with \em b being another distributed matrix
   *        of the same size. Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zip functor, must be a 'curried' function pointer.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam F Function type.
   */
  template<typename T2, typename F>
  void zipInPlace(DMatrix<T2>& b, const Fct2<T, T2, T, F>& f);

  /**
   * \brief Replaces each element m[i][j] of the distributed matrix with
   *        f(m[i][j], b[i][j]) with \em b being another distributed matrix
   *        of the same size. Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zip function.
   * @tparam T2 Element type of the distributed matrix to zip with.
   */
  template<typename T2>
  void zipInPlace(DMatrix<T2>& b, T (*f)(T, T2));

  // SKELETONS / COMPUTATION / ZIP / INDEXINPLACE
  /**
   * \brief Replaces each element m[i][j] of the distributed array with
   *        f(i, j, m[i][j], b[i][j]). Note that besides the elements
   *        themselves also the indices are passed to the functor. Note
   *        that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zipIndex functor, must be a 'curried' function pointer.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam F Function type.
   */
  template<typename T2, typename F>
  void zipIndexInPlace(DMatrix<T2>& b, const Fct4<int, int, T, T2, T, F>& f);

  /**
   * \brief Replaces each element m[i][j] of the distributed array with
   *        f(i, j, m[i][j], b[i][j]). Note that besides the elements
   *        themselves also the indices are passed to the functor. Note
   *        that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zipIndex function.
   * @tparam T2 Element type of the distributed matrix to zip with.
   */
  template<typename T2>
  void zipIndexInPlace(DMatrix<T2>& b, T (*f)(int, int, T, T2));

  // SKELETONS / COMPUTATION / ZIP
  /**
   * \brief Non-inplace variant of the zip skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zip functor, must be a 'curried' function pointer.
   * @tparam R Return type.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam F Function type.
   * @return The newly created distributed matrix.
   */
  template<typename R, typename T2, typename F>
  DMatrix<R> zip(DMatrix<T2>& b, const Fct2<T, T2, R, F>& f);

  /**
   * \brief Non-inplace variant of the zip skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zip function.
   * @tparam R Return type.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @return The newly created distributed matrix.
   */
  template<typename R, typename T2>
  DMatrix<R> zip(DMatrix<T2>& b, R (*f)(T, T2));

  // SKELETONS / COMPUTATION / ZIP / INDEX
  /**
   * \brief Non-inplace variant of the zipIndex skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zipIndex functor, must be a 'curried' function pointer.
   * @tparam R Return type.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam F Function type.
   * @return The newly created distributed matrix.
   */
  template<typename R, typename T2, typename F>
  DMatrix<R> zipIndex(DMatrix<T2>& b, const Fct4<int, int, T, T2, R, F>& f);

  /**
   * \brief Non-inplace variant of the zipIndex skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zipIndex function.
   * @tparam R Return type.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @return The newly created distributed matrix.
   */
  template<typename R, typename T2>
  DMatrix<R> zipIndex(DMatrix<T2>& b, R (*f)(int, int, T, T2));
#endif

  // SKELETONS / COMPUTATION / FOLD

  /**
   * \brief Reduces all elements of the distributed matrix to a single element by
   *        successively applying the given functor \em f. Note that \em f needs to
   *        be a commutative function.
   *
   * @param f The fold functor, must be of type \em MFoldFunctor.
   * @param final_fold_on_cpu Specifies whether the final fold steps are done by the CPU.
   *        Default is false. Passing true may increase performance.
   * @tparam FoldFunctor Functor type.
   * @return The reduce value.
   */
  template<typename FoldFunctor>
  T fold(FoldFunctor& f, bool final_fold_on_cpu = 0);

#ifndef __CUDACC__

  /**
   * \brief Reduces all elements of the distributed matrix to a single element by
   *        successively applying the given functor \em f. Note that \em f needs to
   *        be a commutative function.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The fold functor, must be of type \em MFoldFunctor.
   * @param final_fold_on_cpu Specifies whether the final fold steps are done by the CPU.
   *        Default is true and since this is the CPU version of this skeleton, passing
   *        false will have no effect.
   * @tparam FoldFunctor Functor type.
   * @return The reduced value.
   */
//  template <typename FoldFunctor>
//  T fold(FoldFunctor& f, bool final_fold_on_cpu = 1);
  /**
   * \brief Reduces all elements of the distributed matrix to a single element by
   *        successively applying the given functor \em f. Note that \em f needs to
   *        be a commutative function.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The fold functor, must be a 'curried' function pointer.
   * @tparam F Function type.
   * @return The reduced value.
   */
  template<typename F>
  T fold(const Fct2<T, T, T, F>& f);

  /**
   * \brief Reduces all elements of the distributed matrix to a single element by
   *        successively applying the given function \em f. Note that \em f needs to
   *        be a commutative function.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The fold function.
   * @return The reduced value.
   */
  T fold(T (*f)(T, T));
#endif

  /**
   * \brief Reduces all elements of each row of the distributed matrix to a distributed array with
   *        length(darray) == #rows(dmatrix).
   *        The elements are folded by successively applying the given functor \em f.
   *        Note that \em f needs to be a commutative function.
   *
   * @param f The fold functor, must be of type \em MFoldFunctor.
   * @param final_fold_on_cpu Specifies whether the final fold steps are done by the CPU.
   *        Default is false. Passing true may increase performance.
   * @tparam FoldFunctor Functor type.
   * @return The reduce darray.
   */
  template<typename FoldFunctor>
  DArray<T> foldRows(FoldFunctor& f);

#ifndef __CUDACC__

  DArray<T> foldRows(T (*f)(T, T));
#endif

  /**
   * \brief Reduces all elements of each column of the distributed matrix to a distributed array with
   *        length(darray) == #cols(dmatrix).
   *        The elements are folded by successively applying the given functor \em f.
   *        Note that \em f needs to be a commutative function.
   *
   * @param f The fold functor, must be of type \em MFoldFunctor.
   * @param final_fold_on_cpu Specifies whether the final fold steps are done by the CPU.
   *        Default is false. Passing true may increase performance.
   * @tparam FoldFunctor Functor type.
   * @return The reduce darray.
   */
  template<typename FoldFunctor>
  DArray<T> foldCols(FoldFunctor& f);

#ifndef __CUDACC__
  DArray<T> foldCols(T (*f)(T, T));
#endif

  //
  // SKELETONS / COMMUNICATION
  //

  // SKELETONS / COMMUNICATION / BROADCAST PARTITION

  /**
   * \brief Broadcasts the partition with index (\em blockRow, \em blockCol to
   *        all processes. Afterwards, each partition of the distributed matrix
   *        stores the same values. Note that 0 <= \em blockRow < n and
   *        0 <= blockCol < m.
   *
   * @param blockRow The row index of the partition to broadcast.
   * @param blockCol The column index of the partition to broadcast.
   */
  void broadcastPartition(int blockRow, int blockCol);

  // SKELETONS / COMMUNICATION / GATHER

  /**
   * \brief Transforms a distributed matrix to an ordinary (two-dimnesional)
   *        array by copying each element to the given (two-dimensional) array
   *        \em b. \em b must match the size of the distributed matrix.
   *
   * @param b The (two-dimensional) array to store the elements of the distributed
   *          matrix.
   */
  void gather(T** b);

  /**
   * \brief Transforms a distributed matrix to a copy distributed distributed
   *        matrix by copying each element to the given distributed matrix \em dm.
   *        \em dm must be copy distributed.
   *
   * @param dm The (copy distributed) distributed matrix to stores the elements
   *            of the distributed matrix.
   */
  void gather(DMatrix<T>& dm);

  // SKELETONS / COMMUNICATION / PERMUTE PARTITION

  /**
   * \brief Permutes the partitions of the distributed array according to the
   *        given functions \em newRow and \em newCol. Both functions must be
   *        bijective and return the new row/column index. Note that
   *        0 <= \em newRow < \em blocksInCol and 0 <= \em newCol < \em blocksInRow.
   *
   * @param newRow The bijective function to calculate the new row index, must be
   *               a curried function pointer.
   * @param newCol The bijective function to calculate the new column index, must be
   *               a curried function pointer.
   * @tparam F1 Function type for function \em newRow
   * @tparam F2 Function type for function \em newCol
   */
  template<class F1, class F2>
  void permutePartition(const Fct2<int, int, int, F1>& newRow,
                        const Fct2<int, int, int, F2>& newCol);

  /**
   * \brief Permutes the partitions of the distributed array according to the
   *        given functions \em newRow and \em newCol. Both functions must be
   *        bijective and return the new row/column index. Note that
   *        0 <= \em newRow < \em blocksInCol and 0 <= \em newCol < \em blocksInRow.
   *
   * @param newRow The bijective function to calculate the new row index.
   * @param newCol The bijective function to calculate the new column index.
   *
   */
  void permutePartition(int (*f)(int, int), int (*g)(int, int));

  /**
   * \brief Permutes the partitions of the distributed array according to the
   *        given functions \em f and \em g. Both functions must be
   *        bijective and return the new row/column index. Note that
   *        0 <= \em f < \em blocksInCol and 0 <= \em g < \em blocksInRow.
   *
   * @param f The bijective function to calculate the new row index.
   * @param g The bijective function to calculate the new column index, must be
   *               a curried function pointer.
   * @tparam F Function type for \em newCol.
   */
  template<class F>
  void permutePartition(int (*f)(int, int), const Fct2<int, int, int, F>& g);

  /**
   * \brief Permutes the partitions of the distributed array according to the
   *        given functions \em f and \em g. Both functions must be
   *        bijective and return the new row/column index. Note that
   *        0 <= \em f < \em g and 0 <= \em newCol < \em blocksInRow.
   *
   * @param f The bijective function to calculate the new row index, must be
   *               a curried function pointer.
   * @param g The bijective function to calculate the new column index.
   * @tparam F Function type for \em newCol
   */
  template<class F>
  void permutePartition(const Fct2<int, int, int, F>& f, int (*g)(int, int));

  template<class F1, class F2>
  void permutePartition(F1& f1, F2& f2);

  // SKELETONS / COMMUNICATION / ROTATE

  // SKELETONS / COMMUNICATION / ROTATE / ROTATE COLUMNS

  template<typename F>
  void shiftCols(F& f);

  template<typename F>
  void shiftRows(F& f);

  /**
   * \brief Rotates the partitions of the distributed matrix cyclically in vertical
   *        direction.
   *
   * Rotates the partitions of the distributed matrix cyclically in vertical
   * direction. The number of steps depends on the given function f that
   * calculates this number for each column. Negative numbers correspond to
   * cyclic rotations upwards, positive numbers correspond to cyclic rotations
   * downward.
   *
   * @param f The function to calculate the number of steps, must be a curried
   *          function pointer.
   * @tparam F Function type for \em f.
   */
  template<class F>
  void rotateCols(const Fct1<int, int, F>& f);

  /**
   * \brief Rotates the partitions of the distributed matrix cyclically in vertical
   *        direction.
   *
   * Rotates the partitions of the distributed matrix cyclically in vertical
   * direction. The number of steps depends on the given function f that
   * calculates this number for each column. Negative numbers correspond to
   * cyclic rotations upwards, positive numbers correspond to cyclic rotations
   * downward.
   *
   * @param f The function to calculate the number of steps.
   */
  void rotateCols(int (*f)(int));

  /**
   * \brief Rotates the partitions of the distributed matrix cyclically in vertical
   *        direction.
   *
   * Rotates the partitions of the distributed matrix cyclically in vertical
   * direction. The number of steps is determined by \em rows. Negative numbers
   * correspond to cyclic rotations upwards, positive numbers correspond to
   * cyclic rotations downward.
   *
   * @param rows The number of steps to rotate.
   */
  void rotateCols(int rows);

  // SKELETONS / COMMUNICATION / ROTATE / ROTATE ROWS

  /**
   * \brief Rotates the partitions of the distributed matrix cyclically in horizontal
   *        direction.
   *
   * Rotates the partitions of the distributed matrix cyclically in horizontal
   * direction. The number of steps depends on the given function f that
   * calculates this number for each row. Negative numbers correspond to
   * cyclic rotations to the left, positive numbers correspond to cyclic rotations
   * to the right.
   *
   * @param f The function to calculate the number of steps, must be a curried
   *          function pointer.
   * @tparam F Function type for \em f.
   */
  template<class F>
  void rotateRows(const Fct1<int, int, F>& f);

  /**
   * \brief Rotates the partitions of the distributed matrix cyclically in horizontal
   *        direction.
   *
   * Rotates the partitions of the distributed matrix cyclically in horizontal
   * direction. The number of steps depends on the given function f that
   * calculates this number for each row. Negative numbers correspond to
   * cyclic rotations to the left, positive numbers correspond to cyclic rotations
   * to the right.
   *
   * @param f The function to calculate the number of steps.
   */
  void rotateRows(int (*f)(int));

  /**
   * \brief Rotates the partitions of the distributed matrix cyclically in horizontal
   *        direction.
   *
   * Rotates the partitions of the distributed matrix cyclically in horizontal
   * direction. The number of steps is determined by \em rows. Negative numbers
   * correspond to cyclic rotations to the left, positive numbers correspond to
   * cyclic rotations to the right.
   *
   * @param rows The number of steps to rotate.
   */
  void rotateRows(int cols);

  /**
   * \brief Transposes the local partition. Currently only implemented for
   *        \em nLocal == \em mLocal
   */
  void transposeLocalPartition();

  //
  // GETTERS AND SETTERS
  //

  /**
   * \brief Returns the local partition.
   *
   * @return The local partition.
   */
  T* getLocalPartition() const;
  /**
   * \brief Returns the element at the given global indices (\em row, \em col).
   *
   * @param row The global row index.
   * @param col The global column index.
   * @return The element at the given global indices.
   */
  T get(size_t row, size_t col) const;
  /**
   * \brief Sets the element at the given global indices (\em row, \em col) to the
   *        given value \em v.
   *
   * @param globalIndex The global index.
   * @param v The new value.
   */
  void set(int row, int col, const T& v);

  /**
   * \brief Returns the index of the first row of the local partition.
   *
   * @return Index of the first row of the local partition.
   */
  int getFirstRow() const;

  /**
   * \brief Returns the index of the first column of the local partition.
   *
   * @return Index of the first column of the local partition.
   */
  int getFirstCol() const;

  /**
   * \brief Returns the number of columns of the local partition.
   *
   * @return Number of columns of the local partition.
   */
  int getLocalCols() const;

  /**
   * \brief Returns the number of rows of the local partition.
   *
   * @return Number of rows of the local partition.
   */
  int getLocalRows() const;

  /**
   * \brief Returns the size of the local partition.
   *
   * @return Size of the local partition.
   */
  int getLocalSize() const;

  /**
   * \brief Returns the number of rows of the distributed matrix.
   *
   * @return Number of rows of the distributed matrix.
   */
  int getRows() const;

  /**
   * \brief Returns the number of columns of the distributed matrix.
   *
   * @return Number of columns of the distributed matrix.
   */
  int getCols() const;

  /**
   * \brief Returns the number of blocks (local partitions) in a column.
   *
   * @return Number of blocks (local partitions) in a column.
   */
  int getBlocksInCol() const;

  /**
   * \brief Returns the number of blocks (local partitions) in a row.
   *
   * @return Number of blocks (local partitions) in a row.
   */
  int getBlocksInRow() const;

  /**
   * \brief Checks whether the element at the given global indices (\em row, \em col)
   *        is locally stored.
   *
   * @param row The global row index.
   * @param col The global column index.
   * @return True if the element is locally stored.
   */
  bool isLocal(int row, int col) const;

  /**
   * \brief Returns the element at the given local indices (\em row, \em col).
   *        Note that 0 <= \em row < \em nLocal and 0 <= col < \em mLocal (will
   *        not be checked, for reasons of performance)
   *
   * @param row The local row index.
   * @param col The local column index.
   * @return The element at the given indices.
   */
  T getLocal(int row, int col) const;

  /**
   * \brief Sets the element at the given local indices (\em row, \em col) to the
   *        given value \em v.
   *
   * @param row The local row index.
   * @param col The local column index.
   * @param v The new value.
   */
  void setLocal(int row, int col, const T& v);

  /**
   * \brief Returns the GPU execution plans that store information about size, etc.
   *        for the GPU partitions. For internal purposes.
   *
   * @return The GPU execution plans.
   */
  std::vector<GPUExecutionPlan<T> > getExecPlans();

  /**
   * \brief Returns the GPU execution plan for device \em device.
   *        For internal purposes.
   *
   * @param device The device to get the execution plan for.
   * @return The GPU execution plan for device \em device.
   */
  GPUExecutionPlan<T> getExecPlan(int device);

  /**
   * \brief Switch the distribution scheme from distributed to copy distributed.
   */
  void setCopyDistribution();

  /**
   * \brief Switch the distribution scheme from copy distributed to distributed.
   *        Note that \em rows * \em cols = \em numProcesses must hold.
   *
   * @param rows The number of blocks per row.
   * @param cols The number of blocks per col.
   */
  void setDistribution(int rows, int cols);

  //
  // AUXILIARY
  //

  /**
   * \brief Manually upload the local partition to GPU memory.
   *
   * @param allocOnly Specifies whether data is actually uploaded.
   * @return Set of pointers to GPU memory, one pointer for each GPU.
   */
  std::vector<T*> upload(bool allocOnly = 0);

  /**
   * \brief Manually download the local partition from GPU memory.
   */
  void download();

  /**
   * \brief Manually free device memory.
   */
  void freeDevice();

  /**
   * \brief Set how the local partition is distributed among the GPUs. Current
   *        distribution schemes are: distributed, copy distributed.
   *
   * @param dist The GPU distribution scheme.
   */
  void setGpuDistribution(Distribution dist);

  /**
   * \brief Returns the current GPU distribution scheme.
   *
   * @return The GPU distribution scheme.
   */
  Distribution getGpuDistribution();

  /**
   * \brief Prints the distributed array to standard output. Optionally, the user
   *        may pass a description that will be printed with the output.
   *
   * @param descr The description string.
   */
  void show(const std::string& descr = std::string());

  /**
   * \brief Each process prints its local partition of the distributed array.
   */
  void printLocal();

  std::vector<double> getStencilTimes();

 private:

  //
  // Attributes
  //

  // local partition
  T* localPartition;
  // position of processor in data parallel group of processors; zero-base
  int id;
  // number of rows
  int n;
  // number of columns
  int m;
  // number of local rows
  int nLocal;
  // number of local columns
  int mLocal;
  // nLocal * mLocal;
  int localsize;
  // number of (local) partitions per row
  int blocksInRow;
  // number of (local) partitions per column
  int blocksInCol;
  // X position of processor in data parallel group of processors
  int localColPosition;
  // Y position of processor in data parallel group of processors
  int localRowPosition;
  // position of processor in data parallel group of processors
  int localPosition;
  // first row index in local partition; assuming division mode: block
  int firstRow;
  // first column index in local partition; assuming division mode: block
  int firstCol;
  // index of first row in next partition
  int nextRow;
  // index of first column in next partition
  int nextCol;
  // first (global) index of local partition
  int firstIndex;
  // total number of MPI processes
  int np;
  // checks whether data is up to date in main (cpu) memory
  bool cpuMemoryFlag;
  // execution plans for each gpu
  GPUExecutionPlan<T>* plans = 0;
  // checks whether data is copy distributed among all processes
  Distribution dist;
  // checks whether data is copy distributed among all gpus
  bool gpuCopyDistributed = 0;
  // stencil timing
  double upload_time = 0.0;
  double padding_time = 0.0;
  double kernel_time = 0.0;

  //
  // Skeletons
  //

  template<typename MapFunctor>
  void mapInPlace(MapFunctor& f, Int2Type<true>);

  template<typename MapFunctor>
  void mapInPlace(MapFunctor& f, Int2Type<false>);

  template<typename MapIndexFunctor>
  void mapIndexInPlace(MapIndexFunctor& f, Int2Type<true>);

  template<typename MapIndexFunctor>
  void mapIndexInPlace(MapIndexFunctor& f, Int2Type<false>);

  template<typename R, typename MapFunctor>
  DMatrix<R> map(MapFunctor& f, Int2Type<true>);

  template<typename R, typename MapFunctor>
  DMatrix<R> map(MapFunctor& f, Int2Type<false>);

  template<typename R, typename MapIndexFunctor>
  DMatrix<R> mapIndex(MapIndexFunctor& f, Int2Type<true>);

  template<typename R, typename MapIndexFunctor>
  DMatrix<R> mapIndex(MapIndexFunctor& f, Int2Type<false>);

  template<typename T2, typename ZipFunctor>
  void zipInPlace(DMatrix<T2>& b, ZipFunctor& f, Int2Type<true>);

  template<typename T2, typename ZipFunctor>
  void zipInPlace(DMatrix<T2>& b, ZipFunctor& f, Int2Type<false>);

  template<typename T2, typename ZipIndexFunctor>
  void zipIndexInPlace(DMatrix<T2>& b, ZipIndexFunctor& f, Int2Type<true>);

  template<typename T2, typename ZipIndexFunctor>
  void zipIndexInPlace(DMatrix<T2>& b, ZipIndexFunctor& f, Int2Type<false>);

  template<typename R, typename T2, typename ZipFunctor>
  DMatrix<R> zip(DMatrix<T2>& b, ZipFunctor& f, Int2Type<true>);

  template<typename R, typename T2, typename ZipFunctor>
  DMatrix<R> zip(DMatrix<T2>& b, ZipFunctor& f, Int2Type<false>);

  template<typename R, typename T2, typename ZipIndexFunctor>
  DMatrix<R> zipIndex(DMatrix<T2>& b, ZipIndexFunctor& f, Int2Type<true>);

  template<typename R, typename T2, typename ZipIndexFunctor>
  DMatrix<R> zipIndex(DMatrix<T2>& b, ZipIndexFunctor& f, Int2Type<false>);

  template<typename FoldFunctor>
  T fold(FoldFunctor& f, Int2Type<true>, bool final_fold_on_cpu);

  template<typename FoldFunctor>
  T fold(FoldFunctor& f, Int2Type<false>, bool final_fold_on_cpu);

  template<typename FoldFunctor>
  DArray<T> foldRows(FoldFunctor& f, Int2Type<true>);

  template<typename FoldFunctor>
  DArray<T> foldRows(FoldFunctor& f, Int2Type<false>);

  template<typename FoldFunctor>
  DArray<T> foldCols(FoldFunctor& f, Int2Type<true>);

  template<typename FoldFunctor>
  DArray<T> foldCols(FoldFunctor& f, Int2Type<false>);

  //
  // AUXILIARY
  //

  // initializes distributed matrix (used in constructors).
  void init(int rows, int cols);

  // initializes the GPU execution plans.
  void initGPU();

  // returns the GPU id that locally stores the element at index index.
  int getGpuId(int row, int col) const;

};

}  // namespace msl

#include "../src/dmatrix_common.cpp"

#ifdef __CUDACC__
#include "../src/dmatrix.cu"
#else
#include "../src/dmatrix.cpp"
#endif

