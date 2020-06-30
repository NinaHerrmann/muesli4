/*
 * da.h
 *
 *      Author: Steffen Ernsting <s.ernsting@uni-muenster.de>
 *              Herbert Kuchen <kuchen@uni-muenster.de>
 * 
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2014-2020 	Steffen Ernsting <s.ernsting@uni-muenster.de>,
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

#include <type_traits>

#include "muesli.h"
#include "exception.h"
#include "functors.h"
#include "larray.h"
#include "plarray.h"
#include "exec_plan.h"
#include "conversion.h"
#ifdef __CUDACC__
#include "map_kernels.cuh"
#include "zip_kernels.cuh"
#include "fold_kernels.cuh"
#include "copy_kernel.cuh"
#include "properties.cuh"
#include "exec_plan.h"
#endif

namespace msl {
/**
 * \brief Class DA represents a distributed array.
 *
 * A distributed array represents a one-dimensional parallel container and is
 * distributed among all MPI processes the application was started with. It
 * includes data parallel skeletons such as map, mapStencil, zip, and fold as
 * well as variants of these skeletons.
 *
 * \tparam T Element type. Restricted to classes without pointer data members.
 */
template <typename T>
class DA
{
public:

  //
  // CONSTRUCTORS / DESTRUCTOR
  //

  /**
   * \brief Default constructor.
   */
  DA();

  /**
   * \brief Creates an empty distributed array.
   *
   * @param size Size of the distributed array.
   * @param d Distribution of the distributed array.
   */
  DA(int size);

  /**
   * \brief Creates a distributed array with \em size elements equal to
   *        \em initial_value.
   *
   * @param size Size of the distributed array.
   * @param initial_value Initial value for all elements.
   */
  DA(int size, const T& initial_value);

  /**
   * \brief Creates a distributed array with \em size elements. Elements are
   *        copied from \em initial_array. Note that the length of \em initial_array
   *        must equal \em size.
   *
   * @param size Size of the distributed array.
   * @param initial_array Initial array to copy elements from.

   * @param root_init true: the array is only available for the root process, false: array is available for all processes
   */
//  DA(int size, T* const initial_array, Distribution d = DIST, bool root_init = false);

  /**
   * \brief Creates a distributed array with \em size elements. Initializes all
   *        elements via the given function \em f. Note that global indices
   *        are pass to this function as arguments.
   *
   * @param size Size of the distributed array.
   * @param f Function to initialize the elements of the distributed array.
   * @param d Distribution of the distributed array.
   */
//  DA(int size, T (*f)(int), Distribution d = DIST);

  /**
   * \brief Creates a distributed array with \em size elements. Initializes all
   *        elements via the given functor \em f. Note that global indices
   *        are pass to this functor as arguments.
   *
   * @param size Size of the distributed array.
   * @param f Functor to initialize the elements of the distributed array.
   * @param d Distribution of the distributed array.
   * @tparam F2 Functor type.
   */
 // template <typename F>
 // DA(int size, const F& f, Distribution d = DIST);

  /**
   * \brief Copy constructor.
   *
   * @param cs The copy source.
   */
 // DA(const DA<T>& cs);

  /**
   * \brief Destructor.
   */
  ~DA();

  // ASSIGNMENT OPERATOR
  /**
   * \brief Assignment operator.
   *
   * @param rhs Right hand side of assignment operator.
   */
//  DA<T>& operator=(const DA<T>& rhs);


  //
  // FILL
  //

  /**
   * \brief Initializes the elements of the distributed array with the value \em
   *        value.
   *
   * @param value The value.
   */
  void fill(const T& value);

  /**
   * \brief Initializes the elements of the distributed array with the elements
   *        of the given array of values. Note that the length of \em values must
   *        match the size of the distributed array (not checked).
   *
   * @param values The array of values.
   */
  void fill(T* const values);

  /**
   * \brief Initializes the elements of the distributed array via the given
   *        function \em f. Note that global indices are pass to this function
   *        as arguments.
   *
   * @param f The initializer function.
   */
  void fill(T (*f)(int));

  /**
   * \brief Initializes the elements of the distributed array via the given
   *        functor \em f. Note that global indices are pass to this functor
   *        as arguments.
   *
   * @param f The initializer functor.
   * @tparam F2 Functor type.
   */
  template <typename F>
  void fill(const F& f);

  /**
   * \brief Initializes the elements of the distributed array with the elements
   *        of the given array of values. Note that the length of \em values must
   *        match the size of the distributed array (not checked).
   *        The array is only read by the root process, and afterwards the data is
   *        distributed among all processes.
   *
   * @param values The array of values.
   */
  void fill_root_init(T* const values);


  //
  // SKELETONS / COMPUTATION
  //

  // SKELETONS / COMPUTATION / MAP

  /**
   * \brief Replaces each element a[i] of the distributed array with f(a[i]).
   *
   * @param f The map functor, must be of type \em AMapFunctor.
   * @tparam MapFunctor Functor type.
   */
  template <typename MapFunctor>
  void mapInPlace(MapFunctor& f);

  /**
   * \brief Replaces each element a[i] of the distributed array with f(i, a[i]).
   *        Note that besides the element itself also its index is passed to the
   *        functor.
   *
   * @param f The mapIndex functor, must be of type \em AMapIndexFunctor.
   * @tparam MapIndexFunctor Functor type.
   */
  template <typename MapIndexFunctor>
  void mapIndexInPlace(MapIndexFunctor& f);

  /**
   * \brief Returns a new distributed array with a_new[i] = f(a[i]).
   *
   * @param f The map functor, must be of type \em AMapFunctor.
   * @tparam MapFunctor Functor type.
   * @tparam R Return type.
   * @return The newly created distributed array.
   */
  template <typename F>
  msl::DA<T> map(F& f);  // preliminary simplification, in order to avoid type error
  // should be: msl::DA<R> map(F& f);

  /**
   * \brief Returns a new distributed array with a_new[i] = f(i, a[i]). Note
   *        that besides the element itself also its index is passed to the functor.
   *
   * @param f The mapIndex functor, must be of type \em AMapIndexFunctor.
   * @tparam MapIndexFunctor Functor type.
   * @tparam R Return type.
   * @return The newly created distributed array.
   */
  template <typename R, typename MapIndexFunctor>
  DA<R> mapIndex(MapIndexFunctor& f);

  /**
   * \brief Replaces each element a[i] of the distributed array with f(i, a).
   *        Note that the index i and the local partition is passed to the
   *        functor.
   *
   * @param f The mapStencil functor, must be of type \em AMapStencilFunctor.
   * @tparam MapStencilFunctor Functor type.
   */
  template <typename MapStencilFunctor>
  void mapStencilInPlace(MapStencilFunctor& f, T neutral_value);

  /**
   * \brief Non-inplace variant of the mapStencil skeleton.
   *
   * @see mapStencilInPlace()
   * @param f The mapStencil functor, must be of type \em AMapStencilFunctor.
   * @tparam MapFunctor Functor type.
   * @tparam R Return type.
   * @return The newly created distributed array.
   */
  template <typename R, typename MapStencilFunctor>
  DA<R> mapStencil(MapStencilFunctor& f, T neutral_value);

#ifndef __CUDACC__

  // SKELETONS / COMPUTATION / MAP / INPLACE
  /**
   * \brief Replaces each element a[i] of the distributed array with f(a[i]).
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The map functor, must be a 'curried' function pointer.
   * @tparam F Function type.
   */
  template <typename F>
  void mapInPlace(const msl::Fct1<T, T, F>& f);

  /**
   * \brief Replaces each element a[i] of the distributed array with f(a[i]).
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The mapIndex function.
   */
  void mapInPlace(T(*f)(T));

  // SKELETONS / COMPUTATION / MAP / INDEXINPLACE
  /**
   * \brief Replaces each element a[i] of the distributed array with f(i, a[i]).
   *        Note that besides the element itself also its index is passed to the
   *        functor. Also note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The mapIndex functor, must be a 'curried' function pointer.
   * @tparam F Function type.
   */
  template <typename F>
  void mapIndexInPlace(const msl::Fct2<int, T, T, F>& f);

  /**
   * \brief Replaces each element a[i] of the distributed array with f(i, a[i]).
   *        Note that besides the element itself also its index is passed to the
   *        functor. Also note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The mapIndex function.
   */
  void mapIndexInPlace(T(*f)(int, T));

  // SKELETONS / COMPUTATION / MAP
  /**
   * \brief Non-inplace variant of the map skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The map functor, must be a 'curried' function pointer.
   * @tparam R Return type.
   * @tparam F Function type.
   * @return The newly created distributed array.
   */
  template <typename R, typename F>
  msl::DA<R> map(const msl::Fct1<T, R, F>& f);

  /**
   * \brief Non-inplace variant of the map skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The map function.
   * @tparam R Return type.
   */
  template <typename R>
  msl::DA<R> map(R(*f)(T));

  // SKELETONS / COMPUTATION / MAP / INDEX
  /**
   * \brief Non-inplace variant of the mapIndex skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The mapIndex functor, must be a 'curried' function pointer.
   * @tparam R Return type.
   * @tparam F Function type.
   * @return The newly created distributed array.
   */
  template <typename R, typename F>
  DA<R> mapIndex(const msl::Fct2<int, T, R, F>& f);

  /**
   * \brief Non-inplace variant of the mapIndex skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The mapIndex function.
   * @tparam R Return type.
   * @return The newly created distributed array.
   */
  template <typename R>
  DA<R> mapIndex(R(*f)(int, T));

#endif

  // SKELETONS / COMPUTATION / ZIP

  /**
   * \brief Replaces each element a[i] of the distributed array with f(a[i], b[i])
   *        with \em b being another distributed array of the same size.
   *
   * @param f The zip functor, must be of type \em AZipFunctor.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam ZipFunctor Functor type.
   */
  template <typename T2, typename ZipFunctor>
  void zipInPlace(DA<T2>& b, ZipFunctor& f);

  /**
   * \brief Replaces each element a[i] of the distributed array with f(i, a[i], b[i]).
   *        Note that besides the elements themselves also the index is passed to the
   *        functor.
   *
   * @param f The zipIndex functor, must be of type \em AZipIndexFunctor.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam ZipIndexFunctor Functor type.
   */
  template <typename T2, typename ZipIndexFunctor>
  void zipIndexInPlace(DA<T2>& b, ZipIndexFunctor& f);

  /**
   * \brief Non-inplace variant of the zip skeleton.
   *
   * @param f The zip functor, must be of type \em AZipFunctor.
   * @tparam R Return type.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam ZipFunctor Functor type.
   * @return The newly created distributed array.
   */
  template <typename R, typename T2, typename ZipFunctor>
  DA<R> zip(DA<T2>& b, ZipFunctor& f);

  /**
   * \brief Non-inplace variant of the zipIndex skeleton.
   *
   * @param f The zipIndex functor, must be of type \em AZipIndexFunctor.
   * @tparam R Return type.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam ZipIndexFunctor Functor type.
   * @return The newly created distributed array.
   */
  template <typename R, typename T2, typename ZipIndexFunctor>
  DA<R> zipIndex(DA<T2>& b, ZipIndexFunctor& f);

#ifndef __CUDACC__

  // SKELETONS / COMPUTATION / ZIP / INPLACE
  /**
   * \brief Replaces each element a[i] of the distributed array with f(a[i], b[i])
   *        with \em b being another distributed array of the same size.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zip functor, must be a 'curried' function pointer.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam F Function type.
   */
  template <typename T2, typename F>
  void zipInPlace(DA<T2>& b, const Fct2<T, T2, T, F>& f);

  /**
   * \brief Replaces each element a[i] of the distributed array with f(a[i], b[i])
   *        with \em b being another distributed array of the same size.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zip function.
   * @tparam T2 Element type of the distributed matrix to zip with.
   */
  template <typename T2>
  void zipInPlace(DA<T2>& b, T(*f)(T, T2));

  // SKELETONS / COMPUTATION / ZIP / INDEXINPLACE
  /**
   * \brief Replaces each element a[i] of the distributed array with f(i, a[i], b[i]).
   *        Note that besides the elements themselves also the index is passed to the
   *        functor. Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zipIndex functor, must be a 'curried' function pointer.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam F Function type.
   */
  template <typename T2, typename F>
  void zipIndexInPlace(DA<T2>& b, const Fct3<int, T, T2, T, F>& f);

  /**
   * \brief Replaces each element a[i] of the distributed array with f(i, a[i], b[i]).
   *        Note that besides the elements themselves also the index is passed to the
   *        functor. Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zipIndex function.
   * @tparam T2 Element type of the distributed matrix to zip with.
   */
  template <typename T2>
  void zipIndexInPlace(DA<T2>& b, T(*f)(int, T, T2));

  // SKELETONS / COMPUTATION / ZIP
  /**
   * \brief Non-inplace variant of the zip skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zip functor, must be a 'curried' function pointer.
   * @tparam R Return type.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam F Function type.
   * @return The newly created distributed array.
   */
  template <typename R, typename T2, typename F>
  DA<R> zip(DA<T2>& b, const Fct2<T, T2, R, F>& f);

  /**
   * \brief Non-inplace variant of the zip skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zip function.
   * @tparam R Return type.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @return The newly created distributed array.
   */
  template <typename R, typename T2>
  DA<R> zip(DA<T2>& b, R(*f)(T, T2));

  // SKELETONS / COMPUTATION / ZIP / INDEX
  /**
   * \brief Non-inplace variant of the zipIndex skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zipIndex functor, must be a 'curried' function pointer.
   * @tparam R Return type.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam F Function type.
   * @return The newly created distributed array.
   */
  template <typename R, typename T2, typename F>
  DA<R> zipIndex(DA<T2>& b, const Fct3<int, T, T2, R, F>& f);

  /**
   * \brief Non-inplace variant of the zipIndex skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The zipIndex function.
   * @tparam R Return type.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @return The newly created distributed array.
   */
  template <typename R, typename T2>
  DA<R> zipIndex(DA<T2>& b, R(*f)(int, T, T2));
#endif

  // SKELETONS / COMPUTATION / FOLD

#ifndef __CUDACC__

  /**
   * \brief Reduces all elements of the distributed array to a single element by
   *        successively applying the given functor \em f. Note that \em f needs to
   *        be a commutative function.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The fold functor, must be of type \em AFoldFunctor.
   * @param final_fold_on_cpu Specifies whether the final fold steps are done by the CPU.
   *        Default is true and since this is the CPU version of this skeleton, passing
   *        false will have no effect.
   * @tparam FoldFunctor Functor type.
   * @return The reduced value.
   */
  template <typename FoldFunctor>
  T fold(FoldFunctor& f, bool final_fold_on_cpu = 1);

  /**
   * \brief Reduces all elements of the distributed array to a single element by
   *        successively applying the given functor \em f. Note that \em f needs to
   *        be a commutative function.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The fold functor, must be a 'curried' function pointer.
   * @tparam F Function type.
   * @return The reduced value.
   */
  template <typename F>
  T fold(const Fct2<T, T, T, F>& f);

  /**
   * \brief Reduces all elements of the distributed array to a single element by
   *        successively applying the given function \em f. Note that \em f needs to
   *        be a commutative function.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The fold function.
   * @return The reduced value.
   */
  T fold(T(*f)(T, T));

#else

  /**
   * \brief Reduces all elements of the distributed array to a single element by
   *        successively applying the given functor \em f. Note that \em f needs to
   *        be a commutative function.
   *
   * @param f The fold functor, must be of type \em AFoldFunctor.
   * @param final_fold_on_cpu Specifies whether the final fold steps are done by the CPU.
   *        Default is false. Passing true may increase performance.
   * @tparam FoldFunctor Functor type.
   * @return The reduce value.
   */
  template <typename FoldFunctor>
  T fold(FoldFunctor& f, bool final_fold_on_cpu = 0);

#endif


  //
  // SKELETONS / COMMUNICATION
  //

  // SKELETONS / COMMUNICATION / BROADCAST PARTITION

  /**
   * \brief Broadcasts the partition with index \em partitionIndex to all processes.
   *        Afterwards, each partition of the distributed array stores the same
   *        values. Note that 0 <= \em partitionIndex <= size/numProcesses.
   *
   * @param partitionIndex The index of the partition to broadcast.
   */
  void broadcastPartition(int partitionIndex);

  // SKELETONS / COMMUNICATION / GATHER

  /**
   * \brief Transforms a distributed array to an ordinary array by copying each
   *        element to the given array \em b. \em b must at least be of length
   *        \em size.
   *
   * @param b The array to store the elements of the distributed array.
   */
  void gather(T* b);

  /**
   * \brief Transforms a distributed array to a copy distributed distributed array
   *        by copying each element to the given distributed array \em da. \em da
   *        must be copy distributed, otherwise this function immediately returns.
   *
   * @param da The (copy distributed) distributed array to stores the elements of the
   *           distributed array.
   */
  void gather(msl::DA<T>& da);

  // SKELETONS / COMMUNICATION / PERMUTE PARTITION

  /**
   * \brief Permutes the partitions of the distributed array according to the
   *        given function \em f. \em f must be bijective and return the ID
   *        of the new process p_i to store the partition, with 0 <= i < np.
   *
   * @param f bijective functor
   * @tparam F Function type for \em f.
   */
  template <typename Functor>
  inline void permutePartition(Functor& f);

  /**
   * \brief Permutes the partitions of the distributed array according to the
   *        given function \em f. \em f must be bijective and return the the
   *        ID of the new process p_i to store the partition, with 0 <= i < np.
   *
   * @param f The bijective function.
   */
  // inline void permutePartition(int (*f)(int));


  //
  // GETTERS AND SETTERS
  //

  /**
   * \brief Returns the local partition.
   *
   * @return The local partition.
   */
  T* getLocalPartition();

  /**
   * \brief Returns the element at the given global index \em index.
   *
   * @param index The global index.
   * @return The element at the given global index.
   */
  T get(int index) const;

  /**
   * \brief Sets the element at the given global index \em globalIndex to the
   *        given value \em v, with 0 <= globalIndex < size.
   *
   * @param globalIndex The global index.
   * @param v The new value.
   */
  void set(int globalIndex, const T& v);

  /**
   * \brief Returns the global size of the distributed array.
   *
   * @return The global size.
   */
  int getSize() const;

  /**
   * \brief Returns the size of local partitions of the distributed array.
   *
   * @return The size of local partitions.
   */
  int getLocalSize() const;

  /**
   * \brief Returns the first (global) index of the local partition.
   *
   * @return The first (global) index.
   */
  int getFirstIndex() const;

   /**
   * \brief Setter for cpuMemoryInSync. 
   *
   * @param b new value of cpuMemoryInSync
   */
  void setCpuMemoryInSync(bool b);

  /**
   * \brief Checks whether the element at the given global index \em index is
   *        locally stored.
   *
   * @param index The global index.
   * @return True if the element is locally stored.
   */
  bool isLocal(int index) const;

  /**
   * \brief Returns the element at the given local index \em index. Note that
   *        0 <= \em index < getLocalSize() must hold (will not be checked, for
   *        reasons of performance).
   *
   * @param index The local index.
   */
  T getLocal(int localIndex) const;

  /**
   * \brief Sets the element at the given local index \em localIndex to the
   *        given value \em v.
   *
   * @param localIndex The local index.
   * @param v The new value.
   */
  void setLocal(int localIndex, const T& v);

  /**
   * \brief Returns the GPU execution plans that store information about size, etc.
   *        for the GPU partitions. For internal purposes.
   *
   * @return The GPU execution plans.
   */
  GPUExecutionPlan<T>* getExecPlans();

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
   */
  void setDistribution();


  //
  // AUXILIARY
  //

  /**
   * \brief Manually upload the local partition to GPU memory.
   *
   * @return void
   */

  void upload();

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
   * \brief Prints the local partion of the root processor of the distributed array to standard output. Optionally, the user
   *        may pass a description that will be printed with the output. Just useful for debugging.
   *
   * @param descr The description string.
   */
  void showLocal(const std::string& descr);

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

private:

  //
  // Attributes
  //

  // local partition
  T* localPartition;
  // position of processor in data parallel group of processors; zero-base
  int id;
  // number of elements
  int n;
  // number of local elements
  int nLocal;
  // first (global) index of local partition
  int firstIndex;
  // total number of MPI processes
  int np;
  // tells, whether data is up to date in main (cpu) memory; true := up-to-date, false := newer data on GPU
  bool cpuMemoryInSync;
  // execution plans for each gpu
  GPUExecutionPlan<T>* plans = 0;
  // checks whether data is copy distributed among all processes
  Distribution dist;
  // checks whether data is copy distributed among all gpus
  bool gpuCopyDistributed = 0;
  // number of GPUs per node (= Muesli::num_gpus)
  int ng;  
  // number of elements per GPU (all the same!)                   
  int nGPU;  
  // number of elements on CPU                 
  int nCPU;                    

  //
  // Skeletons
  //

  template <typename MapFunctor>
  void mapInPlace(MapFunctor& f, Int2Type<true>);

  template <typename MapFunctor>
  void mapInPlace(MapFunctor& f, Int2Type<false>);

  template <typename MapIndexFunctor>
  void mapIndexInPlace(MapIndexFunctor& f, Int2Type<true>);

  template <typename MapIndexFunctor>
  void mapIndexInPlace(MapIndexFunctor& f, Int2Type<false>);

  template <typename R, typename MapFunctor>
  DA<R> map(MapFunctor& f, Int2Type<true>);

  template <typename R, typename MapFunctor>
  DA<R> map(MapFunctor& f, Int2Type<false>);

  template <typename R, typename MapIndexFunctor>
  DA<R> mapIndex(MapIndexFunctor& f, Int2Type<true>);

  template <typename R, typename MapIndexFunctor>
  DA<R> mapIndex(MapIndexFunctor& f, Int2Type<false>);

  template <typename T2, typename ZipFunctor>
  void zipInPlace(DA<T2>& b, ZipFunctor& f, Int2Type<true>);

  template <typename T2, typename ZipFunctor>
  void zipInPlace(DA<T2>& b, ZipFunctor& f, Int2Type<false>);

  template <typename T2, typename ZipIndexFunctor>
  void zipIndexInPlace(DA<T2>& b, ZipIndexFunctor& f, Int2Type<true>);

  template <typename T2, typename ZipIndexFunctor>
  void zipIndexInPlace(DA<T2>& b, ZipIndexFunctor& f, Int2Type<false>);

  template <typename R, typename T2, typename ZipFunctor>
  DA<R> zip(DA<T2>& b, ZipFunctor& f, Int2Type<true>);

  template <typename R, typename T2, typename ZipFunctor>
  DA<R> zip(DA<T2>& b, ZipFunctor& f, Int2Type<false>);

  template <typename R, typename T2, typename ZipIndexFunctor>
  DA<R> zipIndex(DA<T2>& b, ZipIndexFunctor& f, Int2Type<true>);

  template <typename R, typename T2, typename ZipIndexFunctor>
  DA<R> zipIndex(DA<T2>& b, ZipIndexFunctor& f, Int2Type<false>);

  template <typename FoldFunctor>
  T fold(FoldFunctor& f, Int2Type<true>, bool final_fold_on_cpu);

  template <typename FoldFunctor>
  T fold(FoldFunctor& f, Int2Type<false>, bool final_fold_on_cpu);

  //
  // AUXILIARY
  //

  // initializes distributed matrix (used in constructors).
  void init();

  // initializes the GPU execution plans.
  void initGPUs();

  // returns the GPU id that locally stores the element at index index.
  int getGpuId(int index) const;
};

} // namespace msl

#include "../src/da.cpp"




