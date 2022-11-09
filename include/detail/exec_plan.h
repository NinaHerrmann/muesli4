/*
 * exec_plan.h
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

template <typename T> struct GPUExecutionPlan {
  // Number of elements per GPU
  long size;

  // Number of byte stored on the GPU
  size_t bytes;

  // Number of rows on the GPU
  // On the new DM data structure this has been repurposed to hold the number of
  // elements.
  int nLocal;

  // Number of columns on the GPU. Obsolete on the new data structure
  int mLocal;

  // Number of rows on the GPU (relevant for the new Data structures only)
  int gpuRows;

  // Number of columns on the GPU (relevant on the new data structures only)
  int gpuCols;

  // Opt: for 3D Number of Depth on the GPU (relevant on the new data structures only)
  int gpuDepth;

  // Index of the first element processed in the GPU. If data structure is a
  // matrix, then this is the row major index.
  int first;

  // First row where the GPU processing starts (global)
  int firstRow;

  // First column where the GPU processing starts (global)
  int firstCol;

  // First depth where the GPU processing starts (global)
  int firstDepth;

  // Last row where the GPU processing ends (global)
  int lastRow;

  // Last column where the GPU processing ends. (global)
  int lastCol;

  // Last Depth where the GPU processing ends. (global)
  int lastDepth;

  // Host copy of the data stored in the GPU
  T *h_Data;

  // Data stored in the GPU
  T *d_Data;
};
