/*
 * properties.h
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

#include "muesli.h"

namespace msl {

namespace detail {

int getMaxBlocksPerRow(int device)
{
  cudaSetDevice(device);
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, device);
  return devProp.maxGridSize[0];
}

int getMaxBlocksPerCol(int device)
{
  cudaSetDevice(device);
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, device);
  return devProp.maxGridSize[1];
}

int getMaxThreadsPerBlock(int device)
{
  cudaSetDevice(device);
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, device);
  return devProp.maxThreadsPerBlock;
}

int getThreadsPerBlock(int device)
{
  cudaSetDevice(device);
  if (msl::Muesli::threads_per_block == 0) {
    int tpb = getMaxThreadsPerBlock(device);
    return tpb;
  } else {
    return msl::Muesli::threads_per_block;
  }
}

}
}

