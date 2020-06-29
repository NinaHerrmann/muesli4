/*
 * matrix_test.cpp
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

#include <iostream>
#include <cmath>
#include <cstdlib>

#include "muesli.h"
#include "dmatrix.h"
#include "darray.h"
#include "curry.h"
#include "functors.h"

template<typename T>
T initWithIndexProduct(int a, int b) {
  return static_cast<T>(a * b + a);
}

template<typename T>
T initWithIndexPos(int a, int b) {
  return static_cast<T>(a * 4 + b);
}

template<typename T>
T add(int a, int b) {
  return (T) (a + b);
}

int add1(int a, int row, int col, int b) {
  return a + b;
}

int add2(int row, int col, int a) {
  return a + row + col;
}

template<typename T>
struct myfunctor : public msl::MMapIndexFunctor<T, T> {
  MSL_USERFUNC
  T operator()(int i, int j, T Cij) const {
    return Cij + i + j;
  }
};

struct myMapFunctor : public msl::MMapFunctor<int, int> {
  MSL_USERFUNC
  int operator()(int value) const {
    return value + 1;
  }
};

template<typename T>
struct Add : public msl::MFoldFunctor<T, T> {
  MSL_USERFUNC
  int operator()(T val1, T val2) const {
    return val1 + val2;
  }
};

template<typename T>
struct Add1 : public msl::MZipIndexFunctor<T, T, T> {
  MSL_USERFUNC
  int operator()(int row, int col, T val1, T val2) const {
    return val1 + val2 + row + col;
  }
};

struct Stencil : public msl::MMapStencilFunctor<int, int> {
  Stencil() {
    this->setStencilSize(1);
    this->setTileWidth(2);
  }

  MSL_USERFUNC
  int operator()(int row, int col, const msl::PLMatrix<int>& input) const {
    return (input.get(row, col) + input.get(row - 1, col)
        + input.get(row, col + 1) + input.get(row + 1, col)
        + input.get(row, col - 1)) / 5;
  }
};

int main(int argc, char** argv) {
  msl::initSkeletons(argc, argv);

  msl::printv("Distributed Matrix Test\n");

  int sqrtp = (int) (sqrt((double) msl::Muesli::num_local_procs) + 0.1);
  msl::printv("sqrtp %i\n", sqrtp);
  msl::DMatrix<int> A(16, 16, sqrtp, sqrtp, &initWithIndexPos<int>);
  msl::DMatrix<int> B(16, 16, 1, msl::Muesli::num_local_procs,
                      &initWithIndexPos<int>);
  msl::DMatrix<int> C(16, 16, msl::Muesli::num_local_procs, 1,
                      &initWithIndexPos<int>);

  A.show("A");
  B.show("B");
  C.show("C");

  Add<int> myaddFoldCols;
  msl::DArray<int> A_result = A.foldCols(myaddFoldCols);
  msl::DArray<int> B_result = B.foldCols(myaddFoldCols);
  msl::DArray<int> C_result = C.foldCols(myaddFoldCols);

  A_result.show("A_Result");
  B_result.show("B_Result");
  C_result.show("C_Result");

#ifndef __CUDACC__
  msl::DArray<int> D_result(16, 0);
  msl::DArray<int> E_result(16, 0);
  msl::DArray<int> F_result(16, 0);

  D_result = A.foldCols(&add<int>);
  E_result = B.foldCols(&add<int>);
  F_result = C.foldCols(&add<int>);

  D_result.show("D_Result");
  E_result.show("E_Result");
  F_result.show("F_Result");
#endif

  msl::terminateSkeletons();
  return 0;
}

