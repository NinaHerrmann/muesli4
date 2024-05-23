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
#include <cstring>

#include "muesli.h"
#include "detail/exception.h"
#include "functors.h"
#include "detail/conversion.h"
#include "detail/exec_plan.h"
#include <utility>
#include "plcube.h"

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

  //DC(int row, int col, int depth, bool rowComplete);

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
  void mapIndex(MapIndexFunctor &f, DC<T>& b); // should be return type DA<R>; debug
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
  * @param result DC to save the result
  * @param f MapStencilFuncotr
  * @param neutral_value_functor NeutralValueFunctor
  */
  template<msl::DCMapStencilFunctor<T> f>
  void mapStencil(DC<T> &result, size_t stencilSize, T neutralValue);

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
   * \brief Returns the element at the given global index \em row, col.
   *
   * @param row The row index.
   * @param col The col index.
   * @param depth The z index.
   * @return The element at the given global index.
   */
    MSL_USERFUNC
    void set(int row, int col, int depth, T value) const;  /**
   * \brief Returns the element at the given global index \em row, col.
   *
   * @param row The row index.
   * @param col The col index.
   * @param depth The z index.
   * @return The element at the given global index.
   */
    MSL_USERFUNC
    T get(int row, int col, int depth) const;
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

    void initPLCubes(int stencilSize, T neutralValue) {
        plCubes = std::vector<PLCube<T>>();
#ifdef __CUDACC__

        plCubes.reserve(this->ng);
        for (int i = 0; i < this->ng; i++) {
            plCubes.push_back(PLCube<T>(
                    this->ncol, this->nrow, this->depth,
                    {this->plans[i].firstCol, this->plans[i].firstRow, this->plans[i].firstDepth},
                    {this->plans[i].lastCol, this->plans[i].lastRow, this->plans[i].lastDepth},
                    i,
                    stencilSize,
                    neutralValue,
                    this->plans[i].d_Data
            ));
        }

#else
     int slicePerProcess = this->depth / msl::Muesli::num_total_procs;
     int slicethisProcess = slicePerProcess * msl::Muesli::proc_id;

     plCubes.push_back(PLCube<T>(
                this->ncol, this->nrow, this->depth,
                {0, 0, slicethisProcess},
                {this->ncol-1, this->nrow-1, (slicethisProcess + slicePerProcess)-1},
                0,
                stencilSize,
                neutralValue,
                this->localPartition
        ));
#endif
        if (msl::Muesli::num_total_procs > 1) {
            size_t topPaddingElements = stencilSize * this->ncol * this->nrow * sizeof(T);
            nodeBottomPadding = new T[topPaddingElements];
            nodeTopPadding = new T[topPaddingElements];
        }

        supportedStencilSize = stencilSize;
    }

    void freePLCubes() {
        supportedStencilSize = -1;
    }

    void syncPLCubes(int stencilSize, T neutralValue) {
        if (stencilSize > supportedStencilSize) {
            freePLCubes();
            initPLCubes(stencilSize, neutralValue);
        }
#ifdef __CUDACC__
        if (stencilSize > supportedStencilSize) {
            freePLCubes();
            initPLCubes(stencilSize, neutralValue);
        }
        for (int i = 1; i < this->ng; i++) {
            size_t bottomPaddingSize = plCubes[i - 1].getBottomPaddingElements() * sizeof(T);
            (cudaMemcpyAsync(
                    plCubes[i - 1].bottomPadding, plCubes[i].data, bottomPaddingSize, cudaMemcpyDeviceToDevice, Muesli::streams[i - 1]
            ));

            size_t topPaddingSize = plCubes[i].getTopPaddingElements() * sizeof(T);
            (cudaMemcpyAsync(
                    plCubes[i].topPadding, plCubes[i - 1].data + (this->plans[i - 1].size - plCubes[i].getTopPaddingElements()), topPaddingSize,
                    cudaMemcpyDeviceToDevice, Muesli::streams[i]
            ));
        }
        msl::syncStreams();
#endif
    }
    void updateNodePaddingGPU(int topPaddingSize) {
#ifdef __CUDACC__
        int lastgpu = this->ng - 1;
        int firstgpu = 0;
        int topPaddingElements = topPaddingSize / sizeof(T);

        if (msl::Muesli::proc_id < msl::Muesli::num_total_procs - 1) {
            (cudaMemcpyAsync(
                    this->localPartition + (this->nLocal - topPaddingElements),
                    plCubes[lastgpu].data + (this->plans[lastgpu].size - topPaddingElements),
                    topPaddingSize,
                    cudaMemcpyDefault, Muesli::streams[lastgpu]
            ));
        }
        if (msl::Muesli::proc_id > 0) {
            (cudaMemcpyAsync(
                    this->localPartition,
                    plCubes[firstgpu].data,
                    topPaddingSize,
                    cudaMemcpyDefault, Muesli::streams[firstgpu]
            ));
        }
#endif
    }
    void updateGPUPaddingNode(int topPaddingSize) {
#ifdef __CUDACC__
        int lastgpu = this->ng - 1;
        int firstgpu = 0;
        if (msl::Muesli::proc_id < msl::Muesli::num_total_procs - 1) {
            (cudaMemcpyAsync(
                    plCubes[lastgpu].bottomPadding,
                    nodeBottomPadding,
                    topPaddingSize,
                    cudaMemcpyDefault, Muesli::streams[lastgpu]
            ));
        }
        if (msl::Muesli::proc_id > 0) {
            (cudaMemcpyAsync(
                    plCubes[firstgpu].topPadding,
                    nodeTopPadding,
                    topPaddingSize,
                    cudaMemcpyDefault, Muesli::streams[firstgpu]
            ));
        }
#else
        if (msl::Muesli::proc_id < msl::Muesli::num_total_procs - 1) {
            memcpy(plCubes[0].bottomPadding, nodeBottomPadding, topPaddingSize);
        }
        if (msl::Muesli::proc_id > 0) {
            memcpy(plCubes[0].topPadding, nodeTopPadding, topPaddingSize);
        }
#endif
    }
    void syncPLCubesMPI(int stencilSize) {
        if (msl::Muesli::num_total_procs <= 1) {
            //printf("Only one process no need to sync MPI\n");
            return;
        }
        size_t topPaddingElements = stencilSize * this->ncol * this->nrow;
        size_t topPaddingSize = stencilSize * this->ncol * this->nrow * sizeof(T);

        // Update from GPU
        updateNodePaddingGPU(topPaddingSize);
        MPI_Status statstart;
        MPI_Request reqstart;
        MPI_Status statbottom;
        MPI_Request reqbottom;

        if (msl::Muesli::proc_id < msl::Muesli::num_total_procs - 1) {
            // Send ending parts. NON BLOCKING
            MSL_ISend(Muesli::proc_id + 1,
                      this->localPartition + this->nLocal - topPaddingElements,
                      reqstart, topPaddingElements,
                      msl::MYTAG);
        }

        if (msl::Muesli::proc_id > 0) {
            // SEND STARTING PARTS NON BLOCKING
            MSL_ISend(Muesli::proc_id - 1,
                      this->localPartition,
                      reqbottom, topPaddingElements,
                      msl::MYADULTTAG);
            // Receive upper parts BLOCKING
            MSL_Recv(Muesli::proc_id - 1,
                     nodeTopPadding,
                     statstart, topPaddingElements,
                     msl::MYTAG);
        }
        if (msl::Muesli::proc_id < msl::Muesli::num_total_procs - 1) {
            MSL_Recv(Muesli::proc_id + 1,
                     nodeBottomPadding,
                     statbottom, topPaddingElements,
                     msl::MYADULTTAG);
        }

        // Update to GPU
        updateGPUPaddingNode(topPaddingSize);
    }

    void prettyPrint() {
#ifdef __CUDACC__
        this->updateHost();
        // Does not work for sequential or host
        for (int z = 0; z < this->depth; z++) {
            for (int y = 0; y < this->nrow; y++) {
                for (int x = 0; x < this->ncol; x++) {
                    printf("%02f ", this->plans[0].h_Data[(z * (this->nrow) + y) * this->ncol + x]);
                }
                printf("\n");
            }
            printf("\n");
        }
#endif
        // Does not work for sequential or host
        for (int z = 0; z < this->depth; z++) {
            for (int y = 0; y < this->nrow; y++) {
                for (int x = 0; x < this->ncol; x++) {
                    printf("%02f ", this->localPartition[(z * (this->nrow) + y) * this->ncol + x]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
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
  void show(const std::string &descr = std::string(), int limited = 0);

  void printnCPU();

  int getCols() {
      return ncol;
  }

  int getRows() {
      return nrow;
  }

  int getDepth() {
      return depth;
  }
        std::vector<PLCube<T>> plCubes;
private:
  //
  // Attributes
  //

  // local partition

  // Number of local rows. If the distribution is not row complete, a row will
  // be counted if one or more elements from that row are part of this
  // partition.
  int nlocalRows{};

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
  bool rowComplete{};

  // Padding to save data calculated from other cpu.
  T * nodeTopPadding, * nodeBottomPadding;

  int supportedStencilSize = -1;

};

} // namespace msl

#include "../src/dc.cpp"
#endif