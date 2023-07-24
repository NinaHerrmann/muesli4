/*
 * da.h
 *
 *      Author: Steffen Ernsting <s.ernsting@uni-muenster.de>
 *              Herbert Kuchen <kuchen@uni-muenster.de>
 *              Nina Herrmann <nina.herrmann@uni-muenster.de>
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
#include "ds.h"

#ifndef MUESLI_DA_H
#define MUESLI_DA_H

#include "muesli.h"
#include "exception.h"
#include "functors.h"
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
    template<typename T>
    class DA : public DS<T>{
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
        DA(int size, const T &v);

        /**
         * \brief Copy constructor.
         *
         * @param cs The copy source.
         */
        // DA(const DA<T>& cs);
        // ASSIGNMENT OPERATOR
        /**
         * \brief Assignment operator.
         *
         * @param rhs Right hand side of assignment operator.
         */
        //  DA<T>& operator=(const DA<T>& rhs);


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
         *        that besides the element itself also its index is passed to the functor.
         *
         * @param f The mapIndex functor, must be of type \em AMapIndexFunctor.
         * @tparam MapIndexFunctor Functor type.
         * @tparam R Return type.
         * @return The newly created distributed array.
         */
        template<typename MapIndexFunctor>
        void mapIndex(MapIndexFunctor &f, DA<T> &result);  // should be return type DA<R>; debug

        /**
         * \brief Replaces each element a[i] of the distributed array with f(i, a).
         *        Note that the index i and the local partition is passed to the
         *        functor.
         *
         * @param f The mapStencil functor, must be of type \em AMapStencilFunctor.
         * @tparam MapStencilFunctor Functor type.
         */
        template<typename MapStencilFunctor>
        void mapStencilInPlace(MapStencilFunctor &f, T neutral_value);

        /**
         * \brief Non-inplace variant of the mapStencil skeleton.
         *
         * @see mapStencilInPlace()
         * @param f The mapStencil functor, must be of type \em AMapStencilFunctor.
         * @tparam MapFunctor Functor type.
         * @tparam R Return type.
         * @return The newly created distributed array.
         */
        template<typename R, typename MapStencilFunctor>
        DA<R> mapStencil(MapStencilFunctor &f, T neutral_value);

        // SKELETONS / COMPUTATION / ZIP ****************************************

        /**
         * \brief Replaces each element a[i] of the distributed array by f(i, a[i], b[i]).
         *
         * @param f The zipIndex functor, must be of type Functor3
         * @tparam T2 Element type of the distributed array to zip with.
         * @tparam ZipIndexFunctor Functor3 type.
         */
        template<typename T2, typename ZipIndexFunctor>
        void zipIndexInPlace(DA<T2> &b, ZipIndexFunctor &f);

        /**
         * \brief Non-inplace variant of the zipIndex skeleton.
         *
         * @param f The zipIndex functor, must be of type Functor3
         * @tparam R Return type.
         * @tparam T2 Element type of the distributed array to zip with.
         * @tparam ZipIndexFunctor Functor type.
         * @return The newly created distributed array.
         */
        template<typename T2, typename ZipIndexFunctor>
        void zipIndex(DA<T2> &b, DA<T2> &result, ZipIndexFunctor &f);  // should be return type DA<R>; debug!

        /**
          * \brief Replaces each element a[i] of the distributed array by f(a[i], b[i], c[i])
          *        with \em b and \em c being other distributed arrays of the same size.
          *
          * @param f The zip functor, must be of type Functor3.
          * @tparam T2 Element type of the distributed array to zip with.
          * @tparam ZipFunctor Functor3 type.
          */
        template<typename T2, typename T3, typename ZipFunctor>
        void zipInPlace3(DA<T2> &b, DA<T3> &c, ZipFunctor &f);

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


        // SKELETONS / COMMUNICATION / PERMUTE PARTITION

        /**
         * \brief Permutes the partitions of the distributed array according to the
         *        given function \em f. \em f must be bijective and return the ID
         *        of the new process p_i to store the partition, with 0 <= i < np.
         *
         * @param f bijective functor
         * @tparam F Function type for \em f.
         */
        template<typename Functor>
        inline void permutePartition(Functor &f);


        /**
          * \brief Prints the local partion of the root processor of the distributed array to standard output. Optionally, the user
          *        may pass a description that will be printed with the output. Just useful for debugging.
          *
          * @param descr The description string.
          */
        void showLocal(const std::string &descr);

        /**
         * \brief Prints the distributed array to standard output. Optionally, the user
         *        may pass a description that will be printed with the output.
         *
         * @param descr The description string.
         */
        void show(const std::string &descr = std::string());

    };

} // namespace msl

#include "../src/da.cpp"
#endif