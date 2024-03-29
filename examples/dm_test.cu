/*
 * dm_test.cpp
 *
 *      Author: Nina Hermann,
 *  	        Herbert Kuchen <kuchen@uni-muenster.de>
 * 
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2020  Herbert Kuchen <kuchen@uni-muenster.de>,
 *                 Nina Hermann
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
#include "dm.h"

namespace msl {
    namespace test {

        class Square : public Functor<int, int> {     
          public: MSL_USERFUNC int operator() (int y) const {return y*y;}
        };

        class Mult : public Functor<int, int> {
        private: int y;
        public:
            Mult(int factor):
                y(factor){}

            MSL_USERFUNC int operator() (int x) const {return x*y;}
        };

        struct Produkt : public Functor3<int, int, int, int>{ 
            MSL_USERFUNC int operator()(int i, int j, int Ai) const {return (i * 10) + j;}
        };

        class Sum : public Functor2<int, int, int>{
        public: MSL_USERFUNC int operator() (int x, int y) const {return x+y;}
        };

        class Sum3 : public Functor3<int, int, int, int>{
        public: MSL_USERFUNC int operator() (int i, int j, int x) const {return i+j+x;}
        };


        class Sum4 : public Functor4<int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int i, int j, int x, int y) const {return i+j+x+y;}
        };

        class CopyCond : public Functor4<int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int x, int v1, int v2, int y) const 
                   {if ((v1 * v2) % 2 == 0) return x; else return y;}
        };

 
        class Proj1 : public Functor4<int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int x, int v1, int v2, int y) const {return x;}
        };


        class Proj2 : public Functor4<int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int x, int v1, int v2, int y) const {return v1;}
        };

        class Proj4 : public Functor4<int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int x, int v1, int v2, int y) const {return y;}
        };

        void dm_test(int dim) {
          printf("Starting dm_test...\n");
          DM<int> a(10,10, 2);
          a.show("a1");
          
          DM<int> b(10,10, 1);
          b.show("b1");
 
          Produkt pr;
          a.mapIndexInPlace(pr);
          a.show("a2");

          Sum sum;
          int result = a.fold(sum,true);
          printf("result: %i\n",result);

          b.zipInPlace(a,sum);
          b.show("b2");

          DM<int> c = a.zip(b,sum);
          c.show("c1");

          Sum4 sum4;
          c.zipIndexInPlace(b,sum4);
          c.show("c2");
          
          DM<int> d = a.zipIndex(b,sum4);
          d.show("d1");

          Mult mult(3);
          a.mapInPlace(mult);
          a.show("a3");
          
          Sum3 sum3;
          a.zipInPlace3(b,c,sum3);
          a.show("a4");
          
          DA<int> ar1(10,0);
          ar1.mapIndexInPlace(sum);
          ar1.show("ar1");
          
          Square sqr;
          DA<int>  ar2 = ar1.map(sqr);
          ar2.show("ar2");
          
          Proj1 pr1;
          a.zipInPlaceAAM(ar1,ar2,b,pr1);
          a.show("a5");

          Proj2 pr2;
          a.zipInPlaceAAM(ar1,ar2,b,pr2);
          a.show("a6");

          Proj4 pr4;
          a.zipInPlaceAAM(ar1,ar2,b,pr4);
          a.show("a7");

          CopyCond copyCond;
          a.zipInPlaceAAM(ar1,ar2,c,copyCond);
          a.show("a8");

          return;
        }
    }} // close namespaces

int main(int argc, char** argv){
  //printf("Starting Main...\n");
  msl::setNumRuns(1);
  msl::setNumGpus(2);
  msl::initSkeletons(argc, argv);
  printf("Starting Program %c with %d nodes %d cpus and %d gpus\n", msl::Muesli::program_name, msl::Muesli::num_total_procs,
  msl::Muesli::num_local_procs, msl::Muesli::num_gpus);
  msl::test::dm_test(16);
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}
