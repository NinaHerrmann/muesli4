/*
 * dm.h
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

#include "detail/exception.h"
#include "functors.h"
#include "muesli.h"

#include "detail/conversion.h"
#include "detail/exec_plan.h"
#include "plmatrix.h"
#include <utility>

#ifdef __CUDACC__
#include "detail/copy_kernel.cuh"

#include "detail/fold_kernels.cuh"
#include "detail/map_kernels.cuh"
#include "detail/properties.cuh"
#include "detail/zip_kernels.cuh"

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
class DM{
public:
  //
  // CONSTRUCTORS / DESTRUCTOR
  //

  /**
   * \brief Default constructor.
   */
  DM();

  /**
   * \brief Creates an empty distributed matrix.
   *
   * @param size Size of the distributed array.
   * @param d Distribution of the distributed array.
   */
  DM(int row, int col);

  DM(int row, int col, bool rowComplete);

  /**
   * \brief Creates a distributed matrix with \em size elements equal to
   *        \em initial_value.
   *
   * @param size Size of the distributed array.
   * @param initial_value Initial value for all elements.
   */
  DM(int row, int col, const T &initial_value);

  /**
   * @brief Creates a distributed matrix with \em size elements equal to
   *        \em initial_value.
   * @param col number or columns
   * @param row Number of rows
   * @param initial_value Initial value of the matrix
   * @param rowComplete if true, the matrix will be distributed between nodes in
   * full rows. If mapStencil will be used, this option needs to be set to true.
   */
  DM(int row, int col, const T &initial_value, bool rowComplete);

#pragma region Rule of five
  /**
   * For more details see https://cpppatterns.com/patterns/rule-of-five.html
   * The 5 functions here are needed to perform operations such as std::move.
   * See examples/jacobi.cu for a usage reference.
   */

  /**
   * @brief Copy constructor. Fully copies the object and it's data.
   *
   */
  DM(const DM<T> &other);

  /**
   * @brief Move constructor. Transfers ownership of resources allocated by \em
   * other to the object that is being created
   *
   * @param other
   */
  DM(DM<T> &&other);

  /**
   * @brief Copy assignment operator. Works the same as the copy constructor.
   *
   * @param other
   * @return DM<T>&
   */
  DM<T> &operator=(const DM<T> &other);

  /**
   * @brief Move assignment operator. This assigs the object defined in \em
   * other to the left hand side of the operation without creating copies
   *
   * @param other
   * @return DM<T>&
   */
  DM<T> &operator=(DM<T> &&other);

  /**
   * \brief Destructor.
   */
  ~DM();

#pragma endregion

  /**
   * \brief Initializes the elements of the distributed array with the value \em
   *        value.
   *
   * @param value The value.
   */
  void fill(const T &value);

  /**
   * \brief Initializes the elements of the distributed matrix with the elements
   *        of the given array of values. Note that the length of \em values
   * must match the size of the distributed array (not checked).
   *
   * @param values The array of values.
   */
  void fill(T *const values);

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
  template <typename F> void fill(const F &f);

  /**
   * \brief Initializes the elements of the distributed array with the elements
   *        of the given array of values. Note that the length of \em values
   * must match the size of the distributed array (not checked). The array is
   * only read by the root process, and afterwards the data is distributed among
   * all processes.
   *
   * @param values The array of values.
   */
  void fill_root_init(T *const values);

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
  template <typename MapFunctor> void mapInPlace(MapFunctor &f);

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
   * \brief Returns a new distributed array with a_new[i] = f(a[i]).
   *
   * @param f The map functor, must be of type \em AMapFunctor.
   * @tparam MapFunctor Functor type.
   * @tparam R Return type.
   * @return The newly created distributed array.
   */
  template <typename F>
  msl::DM<T>
  map(F &f); // preliminary simplification, in order to avoid type error
  // should be: msl::DA<R> map(F& f);

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
  DM<T> mapIndex(MapIndexFunctor &f); // should be return type DA<R>; debug

  /**
   * \brief Replaces each element a[i] of the distributed array with f(i, a).
   *        Note that the index i and the local partition is passed to the
   *        functor.
   *
   * @param f The mapStencil functor, must be of type \em AMapStencilFunctor.
   * @tparam MapStencilFunctor Functor type.
   */
  template <typename MapStencilFunctor, typename NeutralValueFunctor>
  void mapStencilInPlace(MapStencilFunctor &f, NeutralValueFunctor &neutral_value_functor);
  template <typename T2, typename MapStencilFunctor, typename NeutralValueFunctor>
  void mapStencilMM(DM<T2> &result, MapStencilFunctor &f, NeutralValueFunctor &neutral_value_functor);


  /**
   * @brief Non-inplace variant of the mapStencil skeleton.
   *
   * @tparam R type of the resulting matrix
   * @tparam MapStencilFunctor
   * @tparam NeutralValueFunctor
   * @param f
   * @param neutral_value_functor
   * @return DM<R>
   */
  template <typename MapStencilFunctor, typename NeutralValueFunctor>
  DM<T> mapStencil(MapStencilFunctor &f,
                   NeutralValueFunctor &neutral_value_functor);
#ifndef __CUDACC__

  // SKELETONS / COMPUTATION / MAP / INPLACE
  /**
   * \brief Replaces each element a[i] of the distributed array with f(a[i]).
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The map functor, must be a 'curried' function pointer.
   * @tparam F Function type.
   */
  template <typename F> void mapInPlace(const msl::Fct1<T, T, F> &f);

  /**
   * \brief Replaces each element a[i] of the distributed array with f(a[i]).
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The mapIndex function.
   */
  void mapInPlace(T (*f)(T));

  // SKELETONS / COMPUTATION / MAP / INDEXINPLACE
  /**
   * \brief Replaces each element a[i] of the distributed array with f(i, a[i]).
   *        Note that besides the element itself also its index is passed to the
   *        functor. Also note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The mapIndex functor, must be a 'curried' function pointer.
   * @tparam F Function type.
   */
  template <typename F> void mapIndexInPlace(const msl::Fct2<int, T, T, F> &f);

  /**
   * \brief Replaces each element a[i] of the distributed array with f(i, a[i]).
   *        Note that besides the element itself also its index is passed to the
   *        functor. Also note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The mapIndex function.
   */
  void mapIndexInPlace(T (*f)(int, T));

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
  template <typename R, typename F> msl::DM<R> map(const msl::Fct1<T, R, F> &f);

  /**
   * \brief Non-inplace variant of the map skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The map function.
   * @tparam R Return type.
   */
  template <typename R> msl::DM<R> map(R (*f)(T));

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
  DM<R> mapIndex(const msl::Fct2<int, T, R, F> &f);

  /**
   * \brief Non-inplace variant of the mapIndex skeleton.
   *        Note that this is a <b>CPU only</b> skeleton.
   *
   * @param f The mapIndex function.
   * @tparam R Return type.
   * @return The newly created distributed array.
   */
  template <typename R> DM<R> mapIndex(R (*f)(int, T));

#endif

  // SKELETONS / COMPUTATION / ZIP

  /**
   * \brief Replaces each element a[i] of the distributed array with f(a[i],
   * b[i]) with \em b being another distributed array of the same size.
   *
   * @param f The zip functor, must be of type \em AZipFunctor.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam ZipFunctor Functor type.
   */
  template <typename T2, typename ZipFunctor>
  void zipInPlace(DM<T2> &b, ZipFunctor &f);

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
  void zipIndexInPlace(DM<T2> &b, ZipIndexFunctor &f);

  /**
   * \brief Non-inplace variant of the zip skeleton.
   *
   * @param f The zip functor, must be of type \em AZipFunctor.
   * @tparam R Return type.
   * @tparam T2 Element type of the distributed matrix to zip with.
   * @tparam ZipFunctor Functor type.
   * @return The newly created distributed array.
   */
  template <typename T2, typename ZipFunctor>
  DM<T> zip(DM<T2> &b, ZipFunctor &f); // should have result type DA<R>; debug

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
  DM<T> zipIndex(DM<T2> &b, ZipIndexFunctor &f);

  /**
   * \brief Replaces each element a[i,j] of the distributed matrix by f(a[i,j],
   * b[i,j], c[i,j]) with \em b and \em c being other distributed matrices of
   * the same size.
   *
   * @param f The zip functor, must be of type \em AZipFunctor.
   * @tparam T2 Element type of the 1st distributed matrix to zip with.
   * @tparam T3 Element type of the 2nd distributed matrix to zip with.
   * @tparam ZipFunctor Functor type.
   */
  template <typename T2, typename T3, typename ZipFunctor>
  void zipInPlace3(DM<T2> &b, DM<T3> &c, ZipFunctor &f);

  // /**
  //  * \brief Replaces each element a[i,j] of the distributed matrix a by
  //  * f(a[i,j], b[i], c[i], d[i,j]), with \em b and \em c being distributed
  //  * arrays with a number of elements corresponding to the number of rows of
  //  a
  //  *        and d being another distributed matrix of the same size as a
  //  *
  //  * @param f The zip functor, must be of type \em AZipFunctor.
  //  * @tparam T2 Element type of the 1st distributed array b
  //  * @tparam T3 Element type of the 2nd distributed array c
  //  * @tparam T4 Element type of the other distributed matrix d
  //  * @tparam ZipFunctor Functor type.
  //  */
  // template <typename T2, typename T3, typename T4, typename ZipFunctor>
  // void zipInPlaceAAM(DA<T2> &b, DA<T3> &c, DM<T4> &d, ZipFunctor &f);

  /**
   * \brief fold skeleton.
   *
   * @param f The fold functor
   * @tparam T Element type of the distributed matrix to zip with.
   * @tparam ZipIndexFunctor Functor type.
   * @return the result of combining all elements of the arra by the binary,
   * associative and commutativ operation f
   */
  template <typename FoldFunctor>
  T fold(FoldFunctor &f, bool final_fold_on_cpu);

  //
  // SKELETONS / COMMUNICATION
  //

  // SKELETONS / COMMUNICATION / BROADCAST PARTITION

  /**
   * \brief Broadcasts the partition with index \em partitionIndex to all
   * processes. Afterwards, each partition of the distributed array stores the
   * same values. Note that 0 <= \em partitionIndex <= size/numProcesses.
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
  void gather(T *b);

  /**
   * \brief Transforms a distributed array to a copy distributed distributed
   * array by copying each element to the given distributed array \em da. \em da
   *        must be copy distributed, otherwise this function immediately
   * returns.
   *
   * @param da The (copy distributed) distributed array to stores the elements
   * of the distributed array.
   */
  void gather(msl::DM<T> &dm);

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
  // inline void permutePartition(int (*f)(int));

  //
  // GETTERS AND SETTERS
  //

  /**
   * \brief Returns the local partition.
   *
   * @return The local partition.
   */
  T *getLocalPartition();

  /**
   * \briefs Sets the local partition.
   *
   * @param elements for the local partition.
   */
  void setLocalPartition(T *elements);

  /**
   * \brief Returns the element at the given global index \em index.
   *
   * @param index The global index.
   * @return The element at the given global index.
   */
  T get(int index) const;
  /**
   * \brief Returns the element at the given global index \em row, col.
   *
   * @param row The row index.
   * @param col The col index.
   * @return The element at the given global index.
   */
    MSL_USERFUNC
    T get2D(int row, int col, int gpu) const;
    /**
     * \brief Returns the element at the given row \em row and column \em column.
     *
     * @param row The global row.
     * @param column The global column.
     * @return The element at the given global index.
     */
    T get_shared(int row, int column) const;
  /**
   * \brief Sets the element at the given global index \em globalIndex to the
   *        given value \em v, with 0 <= globalIndex < size.
   *
   * @param globalIndex The global index.
   * @param v The new value.
   */
  void set(int globalIndex, const T &v);

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
  T getLocal(int localIndex);

  /**
   * \brief Sets the element at the given local index \em localIndex to the
   *        given value \em v.
   *
   * @param localIndex The local index.
   * @param v The new value.
   */
  void setLocal(int localIndex, const T &v);

  /**
   * \brief Returns the GPU execution plans that store information about size,
   * etc. for the GPU partitions. For internal purposes.
   *
   * @return The GPU execution plans.
   */
  GPUExecutionPlan<T> *getExecPlans();

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
   * \brief Manually download the local partition from GPU memory.
   */
  void downloadupperpart(int paddingsize);
  /**
   * \brief Manually download the local partition from GPU memory.
   */
  void downloadlowerpart(int paddingsize);

  /**
   * \brief Manually free device memory.
   */
  void freeDevice();
  /**
   * \brief Print stencil time.
   */
  void printTime();

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

  /**
   * \brief Each process prints its local partition of the distributed array.
   */
  void printLocal();

private:
  //
  // Attributes
  //

  // local partition
  T *localPartition;
  // position of processor in data parallel group of processors; zero-base
  int id;

  // number of elements
  int n;

  // number of local elements
  int nLocal;

  // Number of local rows. If the distribution is not row complete, a row will
  // be counted if one or more elements from that row are part of this
  // partition.
  int nlocalRows;

  // number of cols
  int ncol;
  // number of rows
  int nrow;

  // first (global) index of local partition
  int firstIndex;
  // first (global) row in local partition
  int firstRow;
  // total number of MPI processes
  int np;
  // tells, whether data is up to date in main (cpu) memory; true := up-to-date,
  // false := newer data on GPU
  bool cpuMemoryInSync;
  // execution plans for each gpu
  GPUExecutionPlan<T> *plans = 0;
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
  // firstIndex caclulated by GPU
  int indexGPU;
  std::vector<PLMatrix<T>*> vplm;

  // Indicates whether the matrix should be distributed in full rows between
  // the nodes. The map stencil functor needs this type of distribution
  bool rowComplete;
  bool plinitMM = false; // pl matrix initialized?

  std::vector<T*> d_dm;
  T* padding_stencil;
  cudaEvent_t start, stop;
  float t0 = 0, t1 = 0, t2= 0, t3= 0, t4= 0, t5= 0, t6= 0, t7= 0, t8= 0, t9=0, t10=0;

  //
  // AUXILIARY
  //

  // initializes distributed matrix (used in constructors).
  void init();

  // initializes the GPU execution plans.
  void initGPUs();

  // returns the GPU id that locally stores the element at index index.
  int getGpuId(int index) const;

  int getFirstGpuRow() const;

  void copyLocalPartition(const DM<T> &other);
  void freeLocalPartition();

  void freePlans();
};

} // namespace msl

#include "../src/dm.cpp"
