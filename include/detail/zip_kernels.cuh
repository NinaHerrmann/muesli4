/*
 * zip_kernels.h
 *
 *      Authors: Steffen Ernsting <s.ernsting@uni-muenster.de>
 *               Herbert Kuchen <kuchen@uni-muenster.de>
 *               Nina Herrmann <nina.herrmann@uni-muenster.de.
 *
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2020 Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *                Herbert Kuchen <kuchen@uni-muenster.de>.
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

#include "exec_plan.h"

namespace msl::detail {

        /**
      * \brief Zip function for Index less Variants of DA and DM.
      * \em Remark: this function could also be called with a DC in case threads are started 1D
      *
      * @param in1 Pointer to gpu memory of datastructure (DA or DM) which provides the data to calcuate on.
      * @param in2 Pointer to gpu memory of datastructure (DA or DM) which provides the data to calcuate on.
      * @param out Pointer to gpu memory of datastructure (DA or DM) where the data is written to.
      * @param size of the data.
      * @param func functor to be called.
      */
        template<typename T1, typename T2, typename R, typename FCT2>
        __global__ void
        zipKernel(const T1 *in0,
                  const T2 *in1,
                  R *out,
                  size_t n,
                  FCT2 func);


        template<typename T1, typename T2,  typename T3, typename R, typename FCT3>
        __global__ void
        zipIndexKernelDMMA(const T1 *in0,
                           const T2 *in1,
                           const T3 *in2,
                           R *out,
                           size_t n,
                           int first,
                           FCT3 func,
                           int nCols);

        /**
       * \brief From HK Zip Skeleton for Index Variants of DMs.
       *
       * @param in1 Pointer to gpu memory of DM which provides the data to calculate on.
       * @param in2 Pointer to gpu memory of DM which provides the data to calculate on.
       * @param out Pointer to gpu memory of DM where the data is written to.
       * @param size of the data.
       * @param first index of first element
       * @param func functor to be called.
       * @param nCols number of columns
       */
        template<typename T1, typename T2, typename R, typename FCT3>
        __global__ void
        zipIndexKernelDM(T1 *in1,
                       T2 *in2,
                       R *out,
                       size_t n,
                       int first,
                       FCT3 func,
                       int nCols);
        /**
       * \brief Zip Skeleton for 3 DS.
       *
       * @param in1 Pointer to gpu memory of DM which provides the data to calculate on.
       * @param in2 Pointer to gpu memory of DM which provides the data to calculate on.
       * @param out Pointer to gpu memory of DM where the data is written to.
       * @param size of the data.
       * @param first index of first element
       * @param func functor to be called.
       * @param nCols number of columns
       */
        template<typename T1, typename T2, typename R, typename FCT3>
        __global__ void
        zipIndexKernel3(T1 *in1,
                       T2 *in2,
                       T2 *in3,
                       R *out,
                       size_t n,
                       int first,
                       FCT3 func,
                       int nCols);

        /**
         * \brief Original usage read data from another point than the current index for DMs
         * - e.g. \em examples/archiveMuesli2/gaussian.cu
         * TODO In my opinion crosszip does not really capture what the skeleton is doing
         * TODO as merely two data structures can be read. This would be more suitable if
         * an area which is read is defined (zipStencil?)
         *
         * @param in1 Pointer to gpu memory of DM which provides the data to calculate on.
         * @param in2 Pointer to gpu memory of DM which provides the data to calculate on.
         * @param out Pointer to gpu memory of DM where the data is written to.
         * @param size of the data.
         * @param first index of first element
         * @param func functor to be called.
         * @param nCols number of columns
         */
        template<typename T1, typename T2, typename R, typename FCT3>
        __global__ void
        crossZipIndexKernel(T1 *in1,
                            T2 *in2,
                            R *out,
                            size_t n,
                            int first,
                            FCT3 func,
                            int ncols);

        /**
         * \brief Original usage read data from another point than the current index for DMs(inPlace)
         * - e.g. \em examples/archiveMuesli2/gaussian.cu
         * TODO In my opinion crosszip does not really capture what the skeleton is doing
         * TODO as merely two data structures can be read. This would be more suitable if
         * an area which is read is defined (zipStencil?)
         *
         * @param in1 Pointer to gpu memory of DM which provides the data to calculate on.
         * @param in2 Pointer to gpu memory of DM which provides the data to calculate on.
         * @param out Pointer to gpu memory of DM where the data is written to.
         * @param size of the data.
         * @param first index of first element
         * @param func functor to be called.
         * @param nCols number of columns
         */
        template<typename T1, typename T2, typename FCT3>
        __global__ void
        crossZipInPlaceIndexKernel(T1 *in1,
                                   T2 *in2,
                                   size_t n,
                                   int first,
                                   FCT3 func,
                                   int nCols);

        /**
        * \brief HK 20.11.2020 Zip a DM two DAs and a DM - assumed all have the same number of elements
        * - e.g. \em examples/archiveMuesli2/gaussian.cu
        * TODO Very specific - it would be nice to find a more generic way to pass multiple arguments.
        *
        * @param in1 Pointer to gpu memory of DM which provides the data to calculate on.
        * @param in2 Pointer to gpu memory of DA which provides the data to calculate on.
        * @param in3 Pointer to gpu memory of DA which provides the data to calculate on.
        * @param in4 Pointer to gpu memory of DM which provides the data to calculate on.
        * @param out Pointer to gpu memory of DM where the data is written to.
        * @param size of the data.
        * @param first index of first element
        * @param first2 is used to calculate an index for arrays. This could be troublesome when having
         * an unknown number of arguments.
        * @param func functor to be called.
        * @param nCols number of columns
        */
        template<typename T1, typename T2, typename T3, typename T4, typename R, typename FCT3>
        __global__ void
        zipKernelAAM(T1 *in1,
                     T2 *in2,
                     T3 *in3,
                     T4 *in4,
                     R *out,
                     size_t n,
                     int first,
                     int first2,
                     FCT3 func,
                     int nCols);

        /**
       * \brief ZipIndex Skeleton for DA (one index Argument)
       *
       * @param in1 Pointer to gpu memory of DM which provides the data to calculate on.
       * @param in2 Pointer to gpu memory of DM which provides the data to calculate on.
       * @param out Pointer to gpu memory of DM where the data is written to.
       * @param size of the data.
       * @param first index of first element
       * @param func functor to be called.
       */
        template<typename T1, typename T2, typename R, typename FCT3>
        __global__ void
        zipIndexKernelDA(T1 *in1,
                       T2 *in2,
                       R *out,
                       size_t n,
                       int first,
                       FCT3 func);

        /**
       * \brief ZipIndex Skeleton for DA (one index Argument)
       *
       * @param in1 Pointer to gpu memory of DM which provides the data to calculate on.
       * @param in2 Pointer to gpu memory of DM which provides the data to calculate on.
       * @param out Pointer to gpu memory of DM where the data is written to.
       * @param size of the data.
       * @param first index of first element
       * @param func functor to be called.
       */
        template<typename T1, typename T2, typename R, typename FCT4>
        __global__ void
        zipIndexKernelDC(const T1 *in1,
                       const T2 *in2,
                       R *out,
                       FCT4 func,
                       int gpuRow, int gpuCol,
                       int offset, int elements);
    }

#include "../../src/zip_kernels.cu"

