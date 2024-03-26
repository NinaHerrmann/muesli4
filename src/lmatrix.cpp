/*
 * lmatrix.cpp
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

template <typename T>
msl::LMatrix<T>::LMatrix(msl::DMatrix<T>& dm, Distribution gpu_dist)
  : current_device(-1), colOffset(dm.getFirstCol()), dist(gpu_dist), dmatrix(&dm), smem_ptr(0), valid_smem(0)
{
}

template <typename T>
msl::LMatrix<T>::~LMatrix()
{
}

template <typename T>
void msl::LMatrix<T>::update()
{
#ifdef __CUDACC__
  current_device = (current_device+1)%Muesli::num_gpus;
  dmatrix->setGpuDistribution(dist);
  current_plan = dmatrix->getExecPlan(current_device);//plans[current_device];
#endif
}

template <typename T>
MSL_USERFUNC
int msl::LMatrix<T>::getRows() const
{
#ifdef __CUDA_ARCH__
  return current_plan.nLocal;
#else
  return dmatrix->getLocalRows();
#endif
}

template <typename T>
MSL_USERFUNC
int msl::LMatrix<T>::getCols() const
{
#ifdef __CUDA_ARCH__
  return current_plan.mLocal;
#else
  return dmatrix->getLocalCols();
#endif
}

template <typename T>
MSL_USERFUNC
T* msl::LMatrix<T>::operator[](int rowIndex) const
{
#ifdef __CUDA_ARCH__
  return &(current_plan.d_Data[rowIndex * current_plan.mLocal]);
#else
  return &(dmatrix->getLocalPartition()[rowIndex * dmatrix->getLocalCols()]);
#endif
}

template <typename T>
MSL_USERFUNC
T msl::LMatrix<T>::get(int row, int col) const
{
#ifdef __CUDA_ARCH__
  return current_plan.d_Data[(row-current_plan.firstRow)*current_plan.mLocal + col - colOffset];
#else
  return dmatrix->getLocalPartition()[(row-dmatrix->getFirstRow())*dmatrix->getLocalCols() + col - colOffset];
#endif
}

