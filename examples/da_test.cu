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

//class Project1 : public Functor2<int, int, int>{
//  MSL_USERFUNC int operator()(int i, int Ai) const {return i;}
//};

struct consti : public Functor2<int, int, int>{ // public msl::AMapIndexFunctor<int, int>{
  MSL_USERFUNC int operator()(int i, int Ai) const {return i;}
};

class Sum : public Functor2<int, int, int>{
public: MSL_USERFUNC int operator() (int x, int y) const {return x+y;}
};

void da_test(int dim) {
  msl::DA<int> a(dim, 2);
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

  msl::DA<int> b(dim, 3);
//    b = a.map(sq));  //syntax error?!
  b.show("b1");

  a.zipInPlace(b,sum);
  a.show("a6");

  auto lambda = [] (int i) {return 1-i;};
  a.permutePartition(lambda);
  a.show("a7");

  a.broadcastPartition(1);
  a.show("a8");
  }
}
} // close namespaces

int main(int argc, char** argv){
  msl::initSkeletons(argc, argv);
  msl::setNumRuns(1);
  msl::setNumGpus(atoi(argv[2]));
  msl::test::da_test(atoi(argv[1]));
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}

