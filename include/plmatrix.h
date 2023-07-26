/*
 * plmatrix.h
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

#include "argtype.h"
#include "shared_mem.h"

namespace msl {

/**
 * \brief Class PLMatrix represents a padded local matrix (partition). It serves
 *        as input for the mapStencil skeleton and actually is a shallow copy that
 *        only stores the pointers to the data. The data itself is managed by the
 *        mapStencil skeleton. For the user, the only important part is the \em get
 *        function.
 *
 * @tparam T The element type.
 */
    template <typename T>
    class PLMatrix : public ArgumentType
    {
    public:
        /**
         * \brief Constructor: creates a PLMatrix.
         * Called with //nrow, ncol, plans[0].gpuRows, plans[0].gpuCols, stencil_size, f.getTileWidth()
         */
        PLMatrix(int n, int m, int r, int c, int ss, int tw, int rep)
                : ArgumentType(), current_data(0), shared_data(0), n(n), m(m), rows(r),
                cols(c), stencil_size(ss), firstRow(Muesli::proc_id*n), firstRowGPU(0), reps(rep), kw(2*ss),
                padding_size(ss * (c + (2*ss))), tile_width(tw), new_tile_width(tw+(ss*2)), inside_elements(tw * tw),
                smoffset(((tw+(ss*2)) * (ss)) + ss), init(0)
        {
        }

        /**
         * \brief Destructor.
         */
        ~PLMatrix()
        {
        }

        /**
         * \brief Adds another pointer to data residing in GPU or in CPU memory,
         *        respectively.
         */
        void addDevicePtr(T* d_ptr)
        {
            ptrs_data.push_back(d_ptr);
            if (it_data!= ptrs_data.begin()) {
                it_data = ptrs_data.begin();
                current_data = *it_data;
            }
        }

        /**
         * \brief Updates the pointer to point to current data (that resides in one of
         *        the GPUs memory or in CPU memory, respectively.
         */
        void update() {
            if (++it_data == ptrs_data.end()) {
                it_data = ptrs_data.begin();
            }
            current_data = *it_data;
        }

        /**
         * \brief Returns the number of rows of the padded local matrix.
         *
         * @return The number of rows of the padded local matrix.
         */
        MSL_USERFUNC
        int getRows() const
        {
            return rows;
        }

        /**
         * \brief Returns the number of columns of the padded local matrix.
         *
         * @return The number of columns of the padded local matrix.
         */
        MSL_USERFUNC
        int getCols() const
        {
            return cols;
        }
        /**
       * \brief Returns the number of columns of the padded local matrix.
       *
       * @return The number of columns of the padded local matrix.
       */
        MSL_USERFUNC
        int getStencil() const
        {
            return stencil_size;
        }

        /**
         * \brief Returns the element at the given global indices (\em row, \em col).
         *
         * @param row The global row index.
         * @param col The global col index.
         */
        MSL_USERFUNC
        T get(int row, int col) {
#ifdef __CUDA_ARCH__
// GPU version: read from shared memory.
                if (padding_size + ((row) * (cols+kw)) + col + stencil_size >= 0 && padding_size + ((row) * (cols+kw)) + col + stencil_size < ((cols+kw)*(rows+kw))){

                    return current_data[padding_size + ((row) * (cols+kw)) + col + stencil_size];
                } else {// SHOULD not happen
                    return neutral_value;
                }
                // TODO If GPU first GPU top nvf
#else
            // CPU version: read from main memory.
	  // bounds check
    if ((col < 0) || (col >= m)) {
      // out of bounds -> return neutral value
      return neutral_value;
    } else { // in bounds -> return desired value
      return current_data[(row-firstRow+stencil_size)*cols + col];
    }
#endif
        }

        MSL_USERFUNC
        void printcurrentData(int row){
            for (int i = 0; i < row * cols; i++){
                if(i%cols == 0){
                    printf("\n");
                }
                printf("%d;", current_data[i]);
            }
        }

        MSL_USERFUNC
        void printSM(int size) const {
#ifdef __CUDA_ARCH__
            for (int i = 1; i < size; i++) {
                if (i%new_tile_width==0){printf("\n%d:",i/new_tile_width);}
                printf("%d;", shared_data[i]);
            }
#endif
        }
#ifdef __CUDACC__
        MSL_USERFUNC
        void readToGlobalMemory() {
            shared_mem = false;
        }

      /**
         * \brief Load (tile_width+stencil)*(tile_height+stencil)
         */
        __device__
        void readToSM(int r, int c, int reps) {
            //T *smem = SharedMemory<T>();
            extern __shared__ int s[];
            shared_data = SharedMemory<T>();
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int g_row = (reps*tile_width * (x / tile_width)) + (threadIdx.x%tile_width);
            int global_col = blockIdx.y * blockDim.y + threadIdx.y;
            const int newsize = new_tile_width * ((reps*tile_width)+kw);
            const int iterations = (newsize / (inside_elements)) + 1;

            for (int rr = 0; rr <= iterations; ++rr) {
                int local_index = (rr * (inside_elements)) + (threadIdx.x) * tile_width + ( threadIdx.y);
                int row = local_index / new_tile_width;
                int firstcol = global_col -  threadIdx.y;
                int g_col = firstcol + ((local_index) % new_tile_width);
                int readfrom = (((g_row-threadIdx.x) + row) * (cols+kw)) + g_col;
                if (local_index <= newsize) {
                    s[local_index] = current_data[readfrom];
                }
            }
            shared_mem = true;
            __syncthreads();
            //shared_data = smem;
        }
#endif

        /**
         * \brief Sets the first row index for the current device.
         *
         * @param fr The first row index.
         */
        void setFirstRowGPU(int fr)
        {
            firstRowGPU = fr;
        }
        MSL_USERFUNC
        T* getcurrentData(){
            return current_data;
        }
        MSL_USERFUNC
        void printfcoutner(){
            printf("Coutner %d;", counter);
        }

    private:
        std::vector<T*> ptrs_data;// ptrs_top, ptrs_bottom;
        typename std::vector<T*>::iterator it_data;// it_top, it_bottom;
        T* current_data, *shared_data; //*data_bottom, *data_top;
        int n, m, rows, cols, stencil_size, firstRow, firstRowGPU, tile_width, new_tile_width, inside_elements, reps,padding_size, kw, smoffset, init;
        T neutral_value;
        bool shared_mem;
        int counter;
    };

}