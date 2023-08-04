/*
 * dm.h
 *
 *      Author: Steffen Ernsting <s.ernsting@uni-muenster.de>
 *              Herbert Kuchen <kuchen@uni-muenster.de>
 *              Nina Herrmann <nina.herrmann@uni-muenster.de>
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2014-2020 	Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *                	    Herbert Kuchen <kuchen@uni-muenster.de.
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
#include "ds.h"
#include "da.h"

#ifndef MUESLI_DM_H
#define MUESLI_DM_H

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
 * \brief Class DM represents a distributed array.
 *
 * A distributed array represents a one-dimensional parallel container and is
 * distributed among all MPI processes the application was started with. It
 * includes data parallel skeletons such as map, mapStencil, zip, and fold as
 * well as variants of these skeletons.
 *
 * \tparam T Element type. Restricted to classes without pointer data members.
 */
template <typename T>
class DM : public msl::DS<T>{
public:
    //
    // CONSTRUCTORS / DESTRUCTOR
    //

    /**
     * \brief Default constructor.
     */
    DM();


    /**
     * \brief Creates a distributed matrix with \em size elements equal to
     *        \em initial_value.
     *
     * @param size Size of the distributed array.
     * @param initial_value Initial value for all elements.
     */
    DM(int row, int col, const T &v);

    /**
     * @brief Creates a distributed matrix with \em size elements equal to
     *        \em initial_value.
     * @param col number or columns
     * @param row Number of rows
     * @param initial_value Initial value of the matrix
     * @param rowComplete if true, the matrix will be distributed between nodes in
     * full rows. If mapStencil will be used, this option needs to be set to true.
     */
    DM(int row, int col, const T &v, bool rowComplete);

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
    DM(const DM<T> &other);

    /**
     * @brief Move constructor. Transfers ownership of resources allocated by \em
     * other to the object that is being created
     *
     * @param other
     */
    DM(DM<T> &&other) noexcept;

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
    DM<T> &operator=(DM<T> &&other) noexcept;

    /**
     * \brief Destructor.
     */
    ~DM();

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
    template<typename MapIndexFunctor>
    void mapIndexInPlace(MapIndexFunctor &f);

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
    template<typename MapIndexFunctor>
    void mapIndex(MapIndexFunctor &f, DM<T> &result); // should be return type DA<R>; debug

    /**
     * \brief TODO Replaces each element a[i] of the distributed array with f(i, a).
     *        Note that the index i and the local partition is passed to the
     *        functor.
     *
     * @param f The mapStencil functor, must be of type \em AMapStencilFunctor.
     * @tparam MapStencilFunctor Functor type.
     */
    template<typename MapStencilFunctor, typename NeutralValueFunctor>
    void mapStencilInPlace(MapStencilFunctor &f, NeutralValueFunctor &neutral_value_functor);

    /**
    * @brief TODO Non-inplace variant of the mapStencil skeleton.
    *
    * @tparam R type of the resulting matrix
    * @tparam MapStencilFunctor
    * @tparam NeutralValueFunctor
    * @param result the dm to save the result
    * @param f
    * @param neutral_value_functor
    * @return void
    */
    template<typename T2, typename MapStencilFunctor, typename NeutralValueFunctor>
    void mapStencilMM(DM<T2> &result, MapStencilFunctor &f, NeutralValueFunctor &neutral_value_functor);

    /**
    * @brief Methods to generalize MapStencil Code snippets.
    *
    * @tparam MapStencilFunctor
    * @param int stencil_size
    * @param int padding_size
    * @param int kw
    * @param f MapStencilFunctor
    * @param int rowoffset
    * @param int coloffset
    * @return void
    */
    template<typename MapStencilFunctor>
    void initializeConstantsStencil(int &stencil_size, int &padding_size, int &col_size, int &kw, MapStencilFunctor &f,
                                                    int &rowoffset, int &coloffset);
    /**
    * @brief Methods to communicate borders between nodes.
    *
    * @tparam MapStencilFunctor
    * @param int stencil_size
    * @param int padding_size
    * @param int kw
    * @param int rowoffset
    * @param int coloffset
    * @return void
    */
    void communicateNodeBorders(int col_size, int stencil_size, int padding_size);
/**
    * @brief TODO Non-inplace variant of the mapStencil skeleton.
    *
    * @tparam R type of the resulting matrix
    * @tparam the dm to save the result
    * @tparam MapStencilFunctor
    * @tparam NeutralValueFunctor
    * @param f
    * @param neutral_value_functor
    * @return DM<R>
    */
    template<typename T2, typename MapStencilFunctor>
    void mapStencilMM(DM<T2> &result, MapStencilFunctor &f, T neutral_value);

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
    template<typename T2, typename ZipIndexFunctor>
    void zipIndexInPlace(DM<T2> &b, ZipIndexFunctor &f);

    /**
     * \brief Replaces each element a[i] of the distributed array with f(i, a[i],
     * b[i]). Note that besides the elements themselves also the index is passed
     * to the functor.
     *
     * @param f The zipIndex functor, must be of type \em AZipIndexFunctor.
     * @tparam T2 Element type of the distributed matrix to zip with.
     * @tparam ZipIndexFunctor Functor type.
     */
    template<typename T2, typename T3, typename ZipIndexFunctor>
    void zipInPlace3(DM<T2> &b, DM<T3> &c, ZipIndexFunctor &f);

    /**
     * \brief Replaces each element a[i] of the distributed array with f(i, a[i],
     * b[i]). Note that besides the elements themselves also the index is passed
     * to the functor.
     *
     * @param f The zipIndex functor, must be of type \em AZipIndexFunctor.
     * @tparam T2 Element type of the distributed matrix to zip with.
     * @tparam ZipIndexFunctor Functor type.
     */
    template<typename T2, typename T3, typename T4, typename ZipIndexFunctor>
    void zipInPlaceAAM(DA<T2> &b, DA<T3> &c, DM<T4> &d, ZipIndexFunctor &f);

    /**
     * \brief Replaces each element a[i] of the distributed array with f(i, a[i],
     * *b[]). Note that besides the elements themselves also the index is passed
     * to the functor. Also note that the whole column is passed.
     *
     * @param f The zipIndex functor, must be of type \em AZipIndexFunctor.
     * @tparam T2 Element type of the distributed matrix to zip with.
     * @tparam ZipIndexFunctor Functor type.
     */
    template<typename T2, typename ZipIndexFunctor>
    void crossZipIndexInPlace(DM<T2> &b, ZipIndexFunctor &f);

    /**
     * \brief Non-inplace variant of the zipIndex skeleton.
     *
     * @param f The zipIndex functor, must be of type \em AZipIndexFunctor.
     * @tparam R Return type.
     * @tparam T2 Element type of the distributed matrix to zip with.
     * @tparam ZipIndexFunctor Functor type.
     * @return The newly created distributed array.
     */
    template<typename T2, typename ZipIndexFunctor>
    void zipIndex(DM<T2> &b, DM<T2> &result, ZipIndexFunctor &f);

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


    // SKELETONS / COMMUNICATION / PERMUTE PARTITION
    /**
       * \brief Rotates the partitions of the distributed matrix cyclically in vertical
       *        direction.
       *
       * Rotates the partitions of the distributed matrix cyclically in vertical direction.
       * The number of steps depends on the given Integer a. Negative numbers correspond to
       * cyclic rotations upwards, positive numbers correspond to cyclic rotations downwards.
       *
       * @param a Integer - Positive rotate down negative rotate up.
       */
    void rotateRows(int a);

    /**
     * \brief Rotates the partitions of the distributed matrix cyclically in horizontal
     *        direction.
     *
     * Rotates the partitions of the distributed matrix cyclically in horizontal direction.
     * The number of steps depends on the given Integer a. Negative numbers correspond to
     * cyclic rotations to the left, positive numbers correspond to cyclic rotations to the
     * right.
     *
     * @param a Integer - Positive rotate down negative rotate up.
     */
    void rotateCols(int a);


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
    T get2D(int row, int col) const;

    /**
     * \brief Manually download the local partition from GPU memory.
     */
    void updateDeviceupperpart(int paddingsize);

    /**
     * \brief Manually download the local partition from GPU memory.
     */
    void updateDevicelowerpart(int paddingsize);

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

    // number of cols
    int ncol;
    // number of rows
    int nrow;
    // Number of local rows. If the distribution is not row complete, a row will
    // be counted if one or more elements from that row are part of this
    // partition.
    int nlocalRows{};
    // first (global) row in local partition
    int firstRow{};

    std::vector<PLMatrix<T> *> vplm;

    // Indicates whether the matrix should be distributed in full rows between
    // the nodes. The map stencil functor needs this type of distribution
    bool rowComplete;

    std::vector<T *> d_dm;

    /**
     * \brief Malloc the necessary space for all GPUs and generates the necessary GPU plans.
     */
    void initGPUs();

    /**
     * \brief Malloc the necessary space for all GPUs and generates the necessary GPU plans.
     */
    void DMinit();
};

} // namespace msl
#include "../src/dm.cpp"

#endif

