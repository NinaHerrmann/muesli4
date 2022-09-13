/*
 * map_kernels.h
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

#include "exec_plan.h"
// #include "plmatrix.h"

// #include "plarray.h"

namespace msl {

    namespace detail {

        /**
        * \brief Map function for DA and DM.
        *
        * @param in Pointer to gpu memory of datastructure (DA or DM) which provides the data to calcuate on.
        * @param out Pointer to gpu memory of datastructure (DA or DM) where the data is written to.
        * @param size of the data.
        * @param func functor to be called.
        */
        template<typename T, typename R, typename F>
        __global__ void mapKernel(T *in, R *out, size_t size, F func);

        /**
        * \brief Map function for DC. \em Includes index calculations and checks to start 3D kernels.
        *
        * @param in Pointer to gpu memory of datastructure (DC) which provides the data to calcuate on.
        * @param out Pointer to gpu memory of datastructure (DC) where the data is written to.
        * @param size of the data.
        * @param func functor to be called. (One Param)
        * @param gpuRows Rows per GPU.
        * @param gpuCols Cols per GPU.
        * @param gpuDepth Depth per GPU.
        */
        template<typename T, typename R, typename F>
        __global__ void mapKernel3D(T *in, R *out, F func, int gpuRows, int gpuCols,
                                    int gpuDepth);

        /**
        * \brief MapIndex function for DM. \em Includes index calculations but is startes 1D. TODO:better start 2D?
        *
        * @param in Pointer to gpu memory of datastructure (DM) which provides the data to calcuate on.
        * @param out Pointer to gpu memory of datastructure (DM) where the data is written to.
        * @param size of the data.
        * @param first Offset for the first element.
        * @param func functor to be called.
        * @param ncols number of cols is used to calc the indices.
        */
        template<typename T, typename R, typename F>
        __global__ void mapIndexKernelDM(T *in, R *out, size_t size, size_t first, F func,
                                         int ncols);
        /**
        * \brief MapIndex function for DA. \em Calls Functor which has two Arguments: the index and the value at the index.
        *
        * @param in Pointer to gpu memory of datastructure (DA) which provides the data to calcuate on.
        * @param out Pointer to gpu memory of datastructure (DA) where the data is written to.
        * @param size of the data.
        * @param first functor to be called.
        * @param func functor to be called.
        * @param localIndices Evaluate - is this feature necessary? Can be used to calculate with local indices.
        */
        template<typename T, typename R, typename F>
        __global__ void mapIndexKernelDA(T *in, R *out, size_t size, size_t first, F func);

        /**
        * \brief MapIndex function for DC. \em Calls Functor which has four Arguments: the three indeces and the value at the index.
        *
        * @param in Pointer to gpu memory of datastructure (DC) which provides the data to calcuate on.
        * @param out Pointer to gpu memory of datastructure (DC) where the data is written to.
        * @param gpuRows Rows per GPU.
        * @param gpuCols Cols per GPU.
        * @param gpuDepth Depth per GPU.
        * @param func functor to be called.
        * @param localIndices Evaluate - is this feature necessary? Can be used to calculate with local indices.
        */
        template<typename T, typename R, typename F>
        __global__ void mapIndexKernelDC(T *in, R *out, int gpuRows, int gpuCols,
                                         int gpuDepth, int firstRow, int firstCol, int firstDepth, F func);

        /**
        * \brief MapInPlace function for DC. \em Calls Functor which has one Arguments: the value at the index.
        *
        * @param inout Pointer to gpu memory of datastructure (DC) which provides the data to calcuate on.
        * @param gpuRows Rows per GPU.
        * @param gpuCols Cols per GPU.
        * @param gpuDepth Depth per GPU.
        * @param func functor to be called.
        */
        template<typename T, typename F>
        __global__ void mapInPlaceKernelDC(T *inout, int gpuRows, int gpuCols,
                                           int gpuDepth, F func);

/*
template <typename T, typename R, typename F, typename NeutralValueFunctor>
__global__ void mapStencilKernel(R *out, GPUExecutionPlan<T> plan,
                                 PLMatrix<T> *input,
                                 F func, int tile_width, int tile_height, NeutralValueFunctor nv);
template <typename T, typename R, typename F>
__global__ void mapStencilMMKernel(R *out,int gpuRows, int gpuCols, int firstCol, int firstRow,  PLMatrix<T> *dm, T * current_data,
                                 F func, int tile_width, int reps, int kw);
template <typename T, typename R, typename F>
__global__ void mapStencilGlobalMem(R *out, GPUExecutionPlan<T> plan, PLMatrix<T> *dm,
                                 F func, int i);
template <typename T, typename R, typename F>
__global__ void mapStencilGlobalMem_rep(R *out, GPUExecutionPlan<T> plan, PLMatrix<T> *dm,
                                        F func, int i, int rep, int tile_width);

template <typename T> __global__ void fillsides(T *A, int paddingoffset, int gpuRows, int ss);
template <typename T> __global__ void fillcore(T *destination, T *source, int paddingoffset, int gpuCols, int ss);*/

    } // namespace detail
} // namespace msl

#include "../../src/map_kernels.cu"
