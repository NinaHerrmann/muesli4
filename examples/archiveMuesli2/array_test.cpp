/*
 * array_test.cpp
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
#include <functional>

#include "muesli.h"
#include "darray.h"
#include "curry.h"
#include "functors.h"

template <typename T>
T add(int a)
{
  return (T) (a + 1);
}

int add1(int a, int index, int b)
{
  return a+b;
}

int add2(int index, int a)
{
  return a+index;
}

template <typename T>
struct myfunctor : public msl::AMapIndexFunctor<T, T>
{
  MSL_USERFUNC
  T operator()(int i, T Ai) const
  {
    return Ai + i;
  }
};

struct myMapFunctor : public msl::AMapFunctor<int, int>
{
  MSL_USERFUNC
  int operator()(int value) const
  {
    return value + 1;
  }
};

template <typename T>
struct Add : public msl::AFoldFunctor<T, T>
{
  MSL_USERFUNC
  int operator()(T val1, T val2) const
  { 
    return val1 + val2;
  }
};

template <typename T>
struct Add1 : public msl::AZipIndexFunctor<T, T, T>
{
  MSL_USERFUNC
  int operator()(int index, T val1, T val2) const
  {
    return val1 + val2 + index;
  }
};

struct Stencil : public msl::AMapStencilFunctor<int, int>
{
  Stencil()
  {
    this->setStencilSize(2);
  }

  MSL_USERFUNC
  int operator() (int index, const msl::PLArray<int>& input) const
  {
    return (input.get(index-1) + input.get(index) + input.get(index+1)) / 3;
  }
};

int main(int argc, char** argv)
{
  msl::initSkeletons(argc, argv);
  msl::setNumGpus(1);

  msl::DArray<int> A(8, &add<int>);
  A.setGpuDistribution(msl::Distribution::DIST);
  A.show("A");
//  A.upload();
  msl::DArray<int> B(A);
  B.show("B");

  msl::DArray<int> a(8, 1);
  msl::DArray<int> b(8, 2);
  auto addLambda = [] MSL_GPUFUNC (int a, int b) {return a+b;};
  a.zipInPlace(b, addLambda);
  a.show("a");

  auto lambdaIndex = [] MSL_GPUFUNC (int index, int x) {return x+index;};
  A.mapIndexInPlace(lambdaIndex);
  A.show("A");

  auto lambdaTest = [] MSL_GPUFUNC (int x) {return x+1;};
  auto T = A.map<int>(lambdaTest);
  T.show("T");

  auto T1 = T.mapIndex<int>(lambdaIndex);
  T1.show("T1");

  auto zipIndex = [] MSL_GPUFUNC (int index, int a, int b) {return a+b+index*2;};
  a.zipIndexInPlace(b, zipIndex);
  a.show("a");

  auto a1 = a.zip<int>(b, lambdaIndex);
  a1.show("a1");

  auto a2 = a.zipIndex<int>(a1, zipIndex);
  a2.show("a2");

  auto fold = [] MSL_GPUFUNC (int a, int b) {return a+b;};
  msl::printv("A.fold(+) = %d\n\n", A.fold(fold));

//#ifndef __CUDACC__
//  typedef int (*addIndex_t) (int, int);
//  A.mapInPlace(msl::curry((addIndex_t) lambdaIndex)(2));
//  A.show("A");
//#endif

  int val1 = A.get(2);
  int val2 = B.get(3);
  if (msl::isRootProcess()) {
    std::cout << "A[2] = " << val1 << std::endl;
    std::cout << "B[3] = " << val2 << std::endl<<std::endl;
  }

  myfunctor<int> mf;
  A.mapIndexInPlace(mf);
  A.show("A");

  myMapFunctor mmf;
  msl::DArray<int> D = A.map<int, myMapFunctor>(mmf);
  D.show("D");

  Add<int> myadd;
  int foldResult = D.fold(myadd);
  msl::printv("D.fold(add) = %d\n", foldResult);

//#ifndef __CUDACC__
//  typedef int (*add1_t)(int, int, int);
//  A.mapIndexInPlace(msl::curry((add1_t) add1) (2));
//  A.mapIndexInPlace(add2);
//  A.show("A");
//#endif

  A.zipInPlace(B, myadd);
  A.show("A");

  Add1<int> myadd1;
  A.zipIndexInPlace(B, myadd1);
  A.show("A");
  B.show("B");

  msl::DArray<int> F = A.zipIndex<int>(B, myadd1);
  F.show("F");

  msl::DArray<int> G = A.zip<int>(B, myadd);
  G.show("G");

  msl::DArray<int> S(64, &add<int>);
  S.show("S");
  Stencil s;
  msl::DArray<int> S1 = S.mapStencil<int>(s, 1);
  S1.show("S1");
  S1.mapStencilInPlace(s, 1);
  S1.show("S1");

  msl::terminateSkeletons();
  return 0;
}





