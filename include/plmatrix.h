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
         * \brief Returns the element at the given global indices (\em row, \em col).
         *
         * @param row The global row index.
         * @param col The global col index.
         */
        MSL_USERFUNC
        T get(int row, int col) const
        {
#ifdef __CUDA_ARCH__
// GPU version: read from shared memory.
            int localrow = row - firstRowGPU;
            if (localrow >= rows) {return data_bottom[col+stencil_size];}
            if (localrow < 0) {return data_top[col+stencil_size];}
            if (col < 0) { return data_left[row];}
            if (col >= m) {return data_right[row];}
            return 0.0;
            int r = blockIdx.y * blockDim.y + threadIdx.y;
            int c = blockIdx.x * blockDim.x + threadIdx.x;
            int rowIndex = (row-firstRowGPU-r+stencil_size)+threadIdx.y;
            int colIndex = (col-c+stencil_size)+threadIdx.x;
            //printf("get  %d % d\n", row, col);
            //printf("access %d\n", rowIndex*width + colIndex);
            return shared_data[rowIndex*width + colIndex];
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
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            for (int i = 0; i < size; i++) {
             //   printf("%.2f;", shared_data[i]);
            }
#endif
        }
#ifdef __CUDACC__
        /**
         * \brief Load (tile_width+stencil)*(tile_height+stencil)
         */
        __device__
        void readToSM(int r, int c, int tile_width) {
            T *smem = SharedMemory<T>();
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int row = r-firstRow;
            //printf("%d;", ty + tx+stencil_size);

            smem[(ty+stencil_size)*width + tx+stencil_size] = current_data[(row+stencil_size)*cols + c];
            smem[(ty+stencil_size)*width + tx+stencil_size +1] =current_data[(row+stencil_size)*cols + c +1];
            smem[(ty+stencil_size)*width + tx+stencil_size -1] =current_data[(row+stencil_size)*cols + c -1];
            smem[(ty+stencil_size+1)*width + tx+stencil_size] =current_data[(row+stencil_size+1)*cols + c];
            smem[(ty+stencil_size-1)*width + tx+stencil_size] =current_data[(row+stencil_size-1)*cols + c];
            printf("%d--", (ty+stencil_size)*width + tx+stencil_size);
            // copy top
            __syncthreads();

            shared_data = smem;
        }
#endif
#ifdef __CUDACC__
        /**
         * \brief Each thread block (on the GPU) reads elements from the padded local
         *        matrix to shared memory. Called by the mapStencil skeleton kernel.
         */
        __device__
        void readToSharedMem(int r, int c, int tile_width)
        {
            int tx = threadIdx.x; int ty = threadIdx.y;
            int row = r-firstRow;
            T *smem = SharedMemory<T>();

            // read assigned value into shared memory
            smem[(ty+stencil_size)*width + tx+stencil_size] = current_data[(row+stencil_size)*cols + c];

            // read halo values
            // first row of tile needs to read upper stencil_size rows of halo values
            if (ty == 0) {
                for (int i = 0; i < stencil_size; i++) {
                    smem[i*width + stencil_size+tx] = current_data[(row+i)*cols + c];
                }
            }

            // last row of tile needs to read lower stencil_size rows of halo values
            if (ty == tile_width-1) {
                for (int i = 0; i < stencil_size; i++) {
                    smem[(i+stencil_size+tile_width)*width + stencil_size+tx] =
                            current_data[(row+stencil_size+i+1)*cols + c];
                }
            }

            // first column of tile needs to read left hand side stencil_size columns of halo values
            if (tx == 0) {
                for (int i = 0; i < stencil_size; i++) {
                    if (c+i-stencil_size < 0) {
                        smem[(ty+stencil_size)*width + i] = neutral_value;
                    }
                    else
                        smem[(ty+stencil_size)*width + i] =
                                current_data[(row+stencil_size)*cols + c+i-stencil_size];
                }
            }

            // last column of tile needs to read right hand side stencil_size columns of halo values
            if (tx == tile_width-1) {
                for (int i = 0; i < stencil_size; i++) {
                    if (c+i+1 > m-1)
                        smem[(ty+stencil_size)*width + i+tile_width+stencil_size] = neutral_value;
                    else
                        smem[(ty+stencil_size)*width + i+tile_width+stencil_size] =
                                current_data[(row+stencil_size)*cols + c+i+1];
                }
            }

            // upper left corner
            if (tx == 0 && ty == 0) {
                for (int i = 0; i < stencil_size; i++) {
                    for (int j = 0; j < stencil_size; j++) {
                        if (c+j-stencil_size < 0)
                            smem[i*width + j] = neutral_value;
                        else
                            smem[i*width + j] = current_data[(row+i)*cols + c+j-stencil_size];
                    }
                }
            }

            // upper right corner
            if (tx == tile_width-1 && ty == 0) {
                for (int i = 0; i < stencil_size; i++) {
                    for (int j = 0; j < stencil_size; j++) {
                        if (c+j+1 > m-1)
                            smem[i*width + j+stencil_size+tile_width] = neutral_value;
                        else
                            smem[i*width + j+stencil_size+tile_width] = current_data[(row+i)*cols + c+j+1];
                    }
                }
            }

            // lower left corner
            if (tx == 0 && ty == tile_width-1) {
                for (int i = 0; i < stencil_size; i++) {
                    for (int j = 0; j < stencil_size; j++) {
                        if (c+j-stencil_size < 0)
                            smem[(i+stencil_size+tile_width)*width + j] = neutral_value;
                        else
                            smem[(i+stencil_size+tile_width)*width + j] =
                                    current_data[(row+i+stencil_size+1)*cols + c+j-stencil_size];
                    }
                }
            }

            // lower right corner
            if (tx == tile_width-1 && ty == tile_width-1) {
                for (int i = 0; i < stencil_size; i++) {
                    for (int j = 0; j < stencil_size; j++) {
                        if (c+j+1 > m-1)
                            smem[(i+stencil_size+tile_width)*width + j+stencil_size+tile_width] = neutral_value;
                        else
                            smem[(i+stencil_size+tile_width)*width + j+stencil_size+tile_width] =
                                    current_data[(row+i+stencil_size+1)*cols + c+j+1];
                    }
                }
            }

            __syncthreads();

            shared_data = smem;
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
        int n, m, rows, cols, stencil_size, firstRow, firstRowGPU, tile_width, width;
        T neutral_value;
    };

}