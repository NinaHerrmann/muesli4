/*
 * da_test.cpp
 *
 *      Author: Herbert Kuchen <kuchen@uni-muenster.de>
 * 
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2020  Herbert Kuchen <kuchen@uni-muenster.de.
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

#include "muesli.h"
#include "da.h"

namespace msl {
namespace test {

class Square : public Functor<int, int> {
public: MSL_USERFUNC int operator() (int x) const {return x*x;}
};

class Mult : public Functor<int, int> {
private: int y;
public: 
Mult(int factor):
  y(factor){}

MSL_USERFUNC int operator() (int x) const {return x*y;}
};

//class Project1 : public Functor2<int, int, int>{  // class funktioniert nicht; nur struct !?
//  MSL_USERFUNC int operator()(int i, int Ai) const {return i;}
//};

struct consti : public Functor2<int, int, int>{ // public msl::AMapIndexFunctor<int, int>{
  MSL_USERFUNC int operator()(int i, int Ai) const {return i;}
};

class Sum : public Functor2<int, int, int>{
public: MSL_USERFUNC int operator() (int x, int y) const {return x+y;}
};

class Sum3 : public Functor3<int, int, int, int>{
public: MSL_USERFUNC int operator() (int x, int y, int z) const {return x+y+z;}
};

void da_test(int dim) {
  DA<int> a(dim, 2);
  a.show("a1");

  Square sq;
  a.mapInPlace(sq);
  a.show("a2");

//  auto inc = [] MSL_USERFUNC (int a) {return a+1;}; // type error due to MSL_USERFUNC
//  a.mapInPlace(inc);
//  a.show("a2b");

  Mult mult(3);
  a.mapInPlace(mult);
  a.show("a3");

  consti pr;
  a.mapIndexInPlace(pr);
  a.show("a4");

  Sum sum;
  int result = a.fold(sum,true);
  printf("result: %i\n",result);
  a.show("a5");

  DA<int> b = a.map(sq); //simplified! type error with non-simplified map (see da.cpp)
  b.show("b1");

  a.zipInPlace(b,sum);
  a.show("a6");

  auto lambda = [] (int i) {return 1-i;};
  a.permutePartition(lambda);
  a.show("a7");

  a.broadcastPartition(1);
  a.show("a8");

  DA<int> c = a.mapIndex(pr); //simplified! type error with non-simplified version (see da.cpp)
  c.show("c1");

  Sum3 sum3;
  a.zipIndexInPlace(c,sum3);
  a.show("a9");

  DA<int> d = a.zip(c,sum); //simplified! type error with non-simplified version (see da.cpp)
  d.show("d1");

  DA<int> e = a.zipIndex(c,sum3); //simplified! type error with non-simplified version (see da.cpp)
  e.show("e1");

  return;
}
}} // close namespaces

int main(int argc, char** argv){
  msl::initSkeletons(argc, argv);
  msl::setNumRuns(1);
  msl::setNumGpus(atoi(argv[2]));
  msl::test::da_test(atoi(argv[1]));
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}

