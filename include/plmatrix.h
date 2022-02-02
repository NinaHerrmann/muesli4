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
                cols(c), stencil_size(ss), firstRow(Muesli::proc_id*n), firstRowGPU(0), reps(rep),
                  tile_width(tw), new_tile_width(tw+(stencil_size*2)), inside_elements(tw * tw), total_elements(tw+stencil_size*tw+stencil_size)
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
        void addDevicePtr(T* d_ptr, T* d_ptr1, T* d_ptr2)
        {
            ptrs_data.push_back(d_ptr);
            if (it_data!= ptrs_data.begin()) {
                it_data= ptrs_data.begin();
                current_data = *it_data;
            }
            ptrs_top.push_back(d_ptr1);
            if (it_top != ptrs_top.begin()) {
                it_top = ptrs_top.begin();
                data_top = *it_top;
            }
            ptrs_bottom.push_back(d_ptr2);
            if (it_bottom != ptrs_bottom.begin()) {
                it_bottom = ptrs_bottom.begin();
                data_bottom = *it_bottom;
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
            if (++it_top == ptrs_top.end()) {
                it_top = ptrs_top.begin();
            }
            data_top = *it_top;
            if (++it_bottom == ptrs_bottom.end()) {
                it_bottom = ptrs_bottom.begin();
            }
            data_bottom = *it_data;
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
            if(shared_mem) {
                // TODO : For runtime purposes this really really has to be shorter
                // TODO goal move logic to read to shared mem (only called once) and then access shared data
                // TODO (cols + stencil) * row + col + upper offset
                size_t g_row = blockIdx.x * blockDim.x + threadIdx.x;
                size_t g_col = blockIdx.y * blockDim.y + threadIdx.y;
                int previous_rows = (reps*tile_width * (g_row / tile_width));
                // TODO 2 is a hotfix
                int smallrow = row - previous_rows;
                int smallcol = col % tile_width;
                const int offset = new_tile_width * (stencil_size);
                if ((offset + (smallrow * new_tile_width) + smallcol) > 0 && (offset + (smallrow * new_tile_width) + smallcol) < 612){
                    return shared_data[offset + (smallrow * new_tile_width) + smallcol];
                } else {// not happening
                    return 255;}
            } else {
                // TODO If GPU first GPU top nvf
                if ((col < 0) || (col >= m) || (row < 0 && firstRowGPU == 0) || (row+firstRowGPU) >= n) {
                    // out of bounds -> return neutral value
                    return 0;
                } else if ((row) >= (rows)) {
                    return data_bottom[col + stencil_size];
                } else if (row < 0) {
                    return data_top[col + stencil_size];
                } else {
                    return current_data[(row) * cols + col];
                }
            }

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
            shared_data = SharedMemory<T>();
            int x = blockIdx.x * blockDim.x + threadIdx.x;

            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int g_row = (reps*tile_width * (x / tile_width)) + (threadIdx.x);
            size_t g_col = blockIdx.y * blockDim.y + threadIdx.y;

            // TODO offset x GPUs?
            int global_col = blockIdx.y * blockDim.y + threadIdx.y;

            const int offset = new_tile_width * (stencil_size);
            // you are the first row of blocks?
            if ((g_row-threadIdx.x) == firstRowGPU) {
                for (int i = 0; i < (stencil_size); i++) {
                    shared_data[ty + (i*new_tile_width)] = data_top[ty + (i*cols)];
                }
                for (int i = 0; i < (stencil_size); i++) {
                    shared_data[ty + (i*new_tile_width) + stencil_size] = data_top[ty + (i*cols) +stencil_size];
                    shared_data[ty + (i*new_tile_width) + (stencil_size*2)] = data_top[ty + (i*cols) +(stencil_size*2)];
                }
            } else {
                for (int i = 0; i < stencil_size; i++) {
                    shared_data[ty + (i*new_tile_width) + stencil_size] = current_data[((g_row - threadIdx.x)*cols) - ((i + 1) * cols) + global_col];
                }
                for (int i = 0; i < (stencil_size); i++) {
                    shared_data[ty + (i*new_tile_width) + stencil_size] = current_data[((g_row - threadIdx.x)*cols) - ((i + 1) * cols) + global_col + stencil_size];
                    shared_data[ty + (i*new_tile_width) + (stencil_size*2)] = current_data[((g_row - threadIdx.x)*cols) - ((i + 1) * cols) + global_col + (stencil_size*2)];
                }
            }
            for (int j = 0; j < reps; j++) {
                shared_data[offset + ((tx + (j*tile_width)) * new_tile_width) + ty + (stencil_size)] = current_data[(g_row + (j*tile_width)) * cols + global_col];
            }
            if (g_col < tile_width){
                for (int i = 0; i < reps; i++) {
                    // TODO nvf
                    shared_data[offset + ((tx+(i*tile_width)) * new_tile_width)] = 0;
                }
            } else {
                for (int i = 0; i < reps; i++) {
                    for (int j = 0; j < (stencil_size); j++) {
                        shared_data[offset + ((tx+(i*tile_width)) * new_tile_width) + j] = current_data[(g_row + (i*tile_width)) * cols + global_col - (stencil_size -j)];
                    }
                }
            }
            // TODO assumption that we never have only one column of blocks
            if (g_col > (cols - tile_width)){
                for (int i = 0; i < reps; i++) {
                    shared_data[offset + (((tx+(i*tile_width)+1) * new_tile_width) -1)] = 0;
                }
            } else {
                for (int i = 0; i < reps; i++) {
                    for (int j = 0; j < (stencil_size); j++) {
                        shared_data[offset + (((tx+(i*tile_width)+1) * new_tile_width) - 1 - j)] = current_data[(g_row + (i*tile_width)) * cols + global_col + (j)];
                    }
                }
            }
            int secondoffset = offset + ((reps * tile_width)*new_tile_width);
            // you are the last block of the gpu
            if ((g_row-threadIdx.x) < ((firstRowGPU + rows)-(tile_width*reps))){
                for (int i = 0; i < stencil_size; i++) {
                    shared_data[ty + (i*new_tile_width) + secondoffset] = current_data[((g_row - threadIdx.x)*cols) + ((reps + i) * cols) + global_col];
                }
                for (int i = 0; i < stencil_size; i++) {
                    shared_data[ty + (i*new_tile_width) + secondoffset + stencil_size] = current_data[((g_row - threadIdx.x)*cols) + ((reps + i) * cols) + global_col + (stencil_size)];
                    shared_data[ty + (i*new_tile_width) + secondoffset+ (stencil_size*2)] = current_data[((g_row - threadIdx.x)*cols) + ((reps + i) * cols) + global_col + (stencil_size*2)];
                }
            } else {
                for (int i = 0; i < stencil_size; i++) {
                    shared_data[ty + (i*new_tile_width) + secondoffset] = data_bottom[i + (i*cols)];
                }
                for (int i = 0; i < stencil_size; i++) {
                    shared_data[ty + (i*new_tile_width) + secondoffset + stencil_size] = data_bottom[ty + (i*cols) + (stencil_size)];
                    shared_data[ty + (i*new_tile_width) + secondoffset+ (stencil_size*2)] = data_bottom[ty + (i*cols)+ (stencil_size*2)];
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
        std::vector<T*> ptrs_data, ptrs_top, ptrs_bottom;
        typename std::vector<T*>::iterator it_data, it_top, it_bottom;
        T* current_data, *shared_data, *data_bottom, *data_top;
        int n, m, rows, cols, stencil_size, firstRow, firstRowGPU, tile_width, new_tile_width, inside_elements, total_elements, reps;
        T neutral_value;
        bool shared_mem;
        int counter;
    };

}