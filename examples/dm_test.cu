/*
 * dm_test.cpp
 *
 *      Author: Nina Hermann,
 *  		Herbert Kuchen <kuchen@uni-muenster.de>
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
#include "dm.h"

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

        struct consti : public Functor3<int, int, int, int>{ // before: public msl::AMapIndexFunctor<int, int>{
            MSL_USERFUNC int operator()(int i, int j, int Ai) const {return i*100 + j;}
        };

        class Sum : public Functor2<int, int, int>{
        public: MSL_USERFUNC int operator() (int x, int y) const {return x+y;}
        };

        class Sum4 : public Functor4<int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int i, int j, int x, int y) const {return i+j+x+y;}
        };


        void dm_test(int dim) {
          //printf("Starting dm_test...\n");
          DM<int> a(10,10, 2);
          a.show("a1");

          Square sq;
          a.mapInPlace(sq);
          a.show("a2");

          Mult mult(3);
          a.mapInPlace(mult);
          a.show("a3");

          DM<int> b(10,10, 1);
          b.show("b1");

          Sum sum;
          b.zipInPlace(a,sum);
          b.show("b2");

          DM<int> c = a.zip(b,sum); 
          c.show("c1");

          Sum4 sum4;
          c.zipIndexInPlace(b,sum4); 
          c.show("c2");

          consti pr;
          a.mapIndexInPlace(pr);
          a.show("a4");

          int result = a.fold(sum,true);
          printf("result: %i\n",result);
          a.show("a5");

          return;
        }
    }} // close namespaces

int main(int argc, char** argv){
  //printf("Starting Main...\n");
  msl::initSkeletons(argc, argv);
  msl::setNumRuns(1);
  msl::setNumGpus(atoi(argv[2]));
  msl::test::dm_test(16);
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}
