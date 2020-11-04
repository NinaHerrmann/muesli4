/*
 *      piDA.cpp  -- Monte Carlo simulation for numerical integration delivering pi
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

class Count: public Functor2<int,int,int>{
private: 
  const int multiplier= 1103515245;
  const int maxRand = 1<<31;
  const int increment = 12345;

public:
   MSL_USERFUNC
   int operator()(int idx, int throws) const {
     int i, insideCircle = 0;
     double randX, randY;
     int state = idx;
     for (i = 0; i < throws; ++i) {
       randX = (state =  myRand(state)) / (double) maxRand;
       randY = (state = myRand(state)) / (double) maxRand;
       if (randX * randX + randY * randY < 1) ++insideCircle;
     }
     return insideCircle; 
   }

   MSL_USERFUNC
   int myRand(int state) const {
     return (multiplier * state + increment) % maxRand;
   }
};

class Sum : public Functor2<int, int, int>{
public: MSL_USERFUNC int operator() (int x, int y) const {return x+y;}
};

void compute_pi(int n, int throws){
  DA<int> process(n,throws);
  Count counter;
  DA<int> result = process.mapIndex(counter);
  Sum add;
  double pi = 4.0 * result.fold(add,true) / (n*throws); 
  printf("pi: %lf\n",pi);
  return;
};
}} // close namespaces
  
int main(int argc, char** argv){  // #processes, #throws, #mpi_nodes
  msl::initSkeletons(argc, argv);
  msl::setNumRuns(1);
  msl::setNumGpus(atoi(argv[3]));
  msl::test::compute_pi(atoi(argv[1]), atoi(argv[2]));
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}


