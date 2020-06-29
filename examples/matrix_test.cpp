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
  msl::DMatrix<int> A(4, 4, sqrtp, sqrtp, &add<int>);
  A.show("A");
  msl::DMatrix<int> B(A);
  B.show("B");

  int a = A.get(2, 2);
  int b = B.get(3, 3);
  msl::printv("A[2][2] = %d\nB[3][3] = %d\n\n", a, b);

  auto mapIndexLambda =
      [] MSL_GPUFUNC (int row, int col, int a) {return a+row+col;};
  A.mapIndexInPlace(mapIndexLambda);
  A.show("A");

  auto mapLambda = [] MSL_GPUFUNC (int a) {return (float)a;};
  auto AF = A.map<float>(mapLambda);
  AF.show("AF");

  auto mapIndexLambdaDouble =
      [] MSL_GPUFUNC (int row, int col, int a) {return (double)a;};
  auto AD = AF.mapIndex<double>(mapIndexLambdaDouble);
  AD.show("AD");

  auto zipLambda = [] MSL_GPUFUNC (float a, double b) {return (float)a+b;};
  AF.zipInPlace(AD, zipLambda);
  AF.show("AF");

  auto zipIndexLambda =
      [] MSL_GPUFUNC (int row, int col, int a, double b) {return (int)row+col+a+b;};
  A.zipIndexInPlace(AD, zipIndexLambda);
  A.show("A");

  auto zipLambda1 = [] MSL_GPUFUNC (float a, double b) {return (int)a+b;};
  auto AI = AF.zip<int>(AD, zipLambda1);
  AI.show("AI");

  auto AI1 = A.zipIndex<int>(AD, zipIndexLambda);
  AI1.show("AI1");

  myfunctor<int> mf;
  A.mapIndexInPlace(mf);
  A.show("A");

  myMapFunctor mmf;
  msl::DMatrix<int> D = A.map<int, myMapFunctor>(mmf);
  D.show("D");

  Add<int> myadd;
  int foldResult = D.fold(myadd);
  msl::printv("D.fold(add) = %d\n\n", foldResult);

  Add<int> myaddFoldRows;
  msl::DArray<int> foldRowsResult = D.foldRows(myaddFoldRows);
  foldRowsResult.show("foldRowsResult");

  Add<int> myaddFoldCols;
  msl::DArray<int> foldColsResult = D.foldCols(myaddFoldCols);
  foldColsResult.show("foldColsResult");

  //Add<int> myadd;
  A.zipInPlace(B, myadd);
  A.show("A");

  Add1<int> myadd1;
  A.zipIndexInPlace(B, myadd1);
  A.show("A");
  B.show("B");

  msl::DMatrix<int> F = A.zipIndex<int>(B, myadd1);
  F.show("F");

  msl::DMatrix<int> G = A.zip<int>(B, myadd);
  G.show("G");

  msl::DMatrix<int> H(4, 4, sqrtp, sqrtp, &add<int>);
  H.show("H");
  H.broadcastPartition(0, 0);
  H.show("H after broadcastPartition(0, 0)");

  msl::DMatrix<int> S(16, 16, msl::Muesli::num_total_procs, 1, &add<int>);
  S.show();
  Stencil s;
  msl::DMatrix<int> S1 = S.mapStencil<int>(s, 0);
  S1.show();
  S1.mapStencilInPlace(s, 0);
  S1.show();

  int matrix1[4] = { 0, 1, 2, 3 };
  msl::DMatrix<int> DM1(2, 2, matrix1);
  msl::DMatrix<int> DM2(2, 2, 1);
  DM1.show("DM1");
  DM2.show("DM2");

  msl::terminateSkeletons();
  return 0;
}

