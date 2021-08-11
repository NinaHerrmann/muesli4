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
         */
        PLMatrix(int n, int m, int r, int c, int ss, int tw)
                : ArgumentType(), current_data(0), shared_data(0), n(n), m(m), rows(r),
                  cols(c), stencil_size(ss), firstRow(Muesli::proc_id*r), firstRowGPU(0),
                  tile_width(tw), width(2*stencil_size+tile_width)
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
        void addDevicePtr(T* d_ptr, T* d_ptr1, T* d_ptr2, T* d_ptr3, T* d_ptr4)
        {
            ptrs.push_back(d_ptr);
            if (it != ptrs.begin()) {
                it = ptrs.begin();
                data_top = *it;
            }
            ptrs1.push_back(d_ptr1);
            if (it1 != ptrs1.begin()) {
                it1 = ptrs1.begin();
                data_bottom = *it1;
            }
            ptrs2.push_back(d_ptr2);
            if (it2 != ptrs2.begin()) {
                it2 = ptrs2.begin();
                data_left = *it2;
            }
            ptrs3.push_back(d_ptr3);
            if (it3 != ptrs3.begin()) {
                it3 = ptrs3.begin();
                data_right = *it3;
            }
            ptrs4.push_back(d_ptr4);
            if (it4 != ptrs4.begin()) {
                it4 = ptrs4.begin();
                current_data = *it4;
            }
        }

        /**
         * \brief Updates the pointer to point to current data (that resides in one of
         *        the GPUs memory or in CPU memory, respectively.
         */
        void update()
        {
            if (++it == ptrs.end()) {
                it = ptrs.begin();
            }
            data_top = *it;
            if (++it1 == ptrs1.end()) {
                it1 = ptrs1.begin();
            }
            data_bottom = *it1;
            if (++it2 == ptrs2.end()) {
                it2 = ptrs2.begin();
            }
            data_left = *it2;
            if (++it3 == ptrs3.end()) {
                it3 = ptrs3.begin();
            }
            data_right = *it3;
            if (++it4 == ptrs4.end()) {
                it4 = ptrs4.begin();
            }
            current_data = *it4;
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
                if (!sminit){
                    int tx = threadIdx.x;
                    int ty = threadIdx.y;
                    // Very simple approach write middle data to sm
                    for (int i = 0; i < tile_width*tile_width; i++) {
                        int row = i / tile_width;
                        //smem[((ty) * tile_width) + tx] = current_data[(row) * cols + i%tile_width];
                    }
                    __syncthreads();
                    sminit = true;
                }
                // check if we need to catch data from other GPU
                if ((col < 0) || (col >= m) || (row < 0 && firstRowGPU == 0) || (row >= n && firstRowGPU == (n-rows))) {
                    // out of bounds -> return neutral value
                    return 0;
                } else if(row >= rows && firstRowGPU == 0 || row < 0 && firstRowGPU != 0) {
                    // get "bottom" or "top"
                    return current_data[(row-firstRow)*cols + col];
                } else { // in bounds -> return from current data or sm
                    // if data is on other gpu
                    if (row >= rows && firstRowGPU == 0 || row < 0 && firstRowGPU != 0) {
                        // get "bottom" or "top"
                        return current_data[(row - firstRow) * cols + col];
                    }
                    int r = blockIdx.y * blockDim.y + threadIdx.y;
                    int c = blockIdx.x * blockDim.x + threadIdx.x;
                    int rowIndex = (row - firstRowGPU - r) + threadIdx.y;
                    int colIndex = (col - c) + threadIdx.x;
                    int indextotake = (rowIndex) * tile_width + colIndex;
                    // if data is not in SM (outside of tile) take current_data
                    if (indextotake >= tile_width*tile_width){
                        //return current_data[(row - firstRow) * cols + col];
                    }

                    if (rowIndex == 19 && colIndex == 19 && r == 15 && c == 15) {
                        // 15 - 31
                        printf("%d-%d-%d-%d= row %d firrowgpu %d threadidxy %d;", rowIndex, colIndex, r, c, row , firstRowGPU , threadIdx.y);
                    }
                    if (rowIndex > 100){
                        printf("Got you! %d \n", rowIndex);
                    }
                    return 2;//smem[15];
                }
            } else {
                // TODO If GPU first GPU top nvf

                if ((col < 0) || (col >= m) || (row < 0 && firstRowGPU == 0) || (row >= n && firstRowGPU == (n-rows))) {
                    // out of bounds -> return neutral value
                    return 0;
                } else if(row >= rows && firstRowGPU == 0) {
                    return data_bottom[col + stencil_size];
                } else if(row < 0 && firstRowGPU != 0){
                    return data_top[col + stencil_size];
                } else { // in bounds -> return desired value
                    return current_data[(row-firstRow)*cols + col];
                }
            }
            // TODO assumes row wise distribution
        /*    int localrow = row - firstRowGPU;
            // upper right corner has wrong value

            else { return current_data[(row-firstRowGPU)*cols + col];}*/
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
        void printSM(int size) const {
#ifdef __CUDA_ARCH__
            for (int i = 1; i < size; i++) {
                if (i%tile_width==0){printf("\n");}
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
        void readToSM(int r, int c) {
            sminit = false;
            shared_mem = true;
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
        void printcurrentData(int row){
            for (int i = 0; i < row * cols; i++){
                if(i%cols == 0){
                    printf("\n");
                }
                printf("%d;", current_data[i]);
            }
        }
    private:
        std::vector<T*> ptrs;
        typename std::vector<T*>::iterator it;
        std::vector<T*> ptrs1;
        typename std::vector<T*>::iterator it1;
        std::vector<T*> ptrs2;
        typename std::vector<T*>::iterator it2;
        std::vector<T*> ptrs3;
        typename std::vector<T*>::iterator it3;
        std::vector<T*> ptrs4;
        typename std::vector<T*>::iterator it4;
        T* current_data, *shared_data, *data_bottom, *data_top, *data_left, *data_right;
        T *smem = SharedMemory<T>();
        int n, m, rows, cols, stencil_size, firstRow, firstRowGPU, tile_width, width;
        T neutral_value;
        bool shared_mem, sminit;
    };

}