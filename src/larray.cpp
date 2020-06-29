/*
 * larray.cpp
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
msl::LArray<T>::LArray(msl::DArray<T>& da, Distribution gpu_dist)
  : current_device(-1), dist(gpu_dist), darray(&da), smem_ptr(0), valid_smem(0)
{
}

template <typename T>
msl::LArray<T>::LArray(msl::DArray<T>& da, detail::FunctorBase* f, Distribution gpu_dist)
  : current_device(-1), dist(gpu_dist), darray(&da), smem_ptr(0), valid_smem(0)
{
  f->addArgument(this);
}

template <typename T>
msl::LArray<T>::~LArray()
{
}

template <typename T>
void msl::LArray<T>::update()
{
#ifdef __CUDACC__
  current_device = (current_device+1)%Muesli::num_gpus;
  darray->setGpuDistribution(dist);
  current_plan = darray->getExecPlan(current_device);
#endif
}

template <typename T>
MSL_USERFUNC
int msl::LArray<T>::getSize() const
{
#ifdef __CUDA_ARCH__
  return current_plan.size;
#else
  return darray->getSize();
#endif
}

template <typename T>
MSL_USERFUNC
T msl::LArray<T>::operator[](int index) const
{
#ifdef __CUDA_ARCH__
  return current_plan.d_Data[index];
#else
  return darray->getLocalPartition()[index];
#endif
}

template <typename T>
MSL_USERFUNC
T msl::LArray<T>::get(int index) const
{
#ifdef __CUDA_ARCH__
  return current_plan.d_Data[index-current_plan.first];
#else
  return darray->getLocalPartition()[index-darray->getFirstIndex()];
#endif
}





