/*
 * froebenius.cpp
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

#include "muesli.h"
#include "dmatrix.h"

namespace msl {

namespace test {

namespace frobenius {

template <class T>
T init(int i, int j)
{
  return i+j+1;
}

template <typename T>
class Square : public MMapFunctor<T, T>
{
public:
  MSL_USERFUNC
  T operator() (T x) const
  {
    return x*x;
  }
};

template <typename T>
class Sum : public MFoldFunctor<T, T>
{
public:
  MSL_USERFUNC
  T operator() (T x, T y) const
  {
    return x+y;
  }
};

template <class T>
void frobenius(int dim, bool output) {
  msl::DMatrix<T> dm(dim, dim, 1, msl::Muesli::num_total_procs, &init<T>, Distribution::DIST);

  Square<T> sq;
  Sum<T> sum;
  //auto sq = [] MSL_GPUFUNC (T a) {return a*a;};
  //auto sum = [] MSL_GPUFUNC (T a, T b) {return a+b;};
  dm.mapInPlace(sq);
  T f_norm = dm.fold(sum, 1);

  if (output) {
    msl::printv("||A||_F = %f\n", sqrt(f_norm));
  }
}

}
}
}

int main(int argc, char** argv)
{
  using namespace msl::test::frobenius;
  msl::initSkeletons(argc, argv);

  auto dimension = 32;
  auto num_gpus = 2;
  auto runs = 1;

  bool output = true;
  bool warmup = false;

  if (argc < 4) {
    if (msl::isRootProcess()) {
      std::cout << std::endl << std::endl;
      std::cout << "Usage: " << argv[0]
          << " #nDimension #nRuns #nGPUs" << std::endl;
      std::cout << "Default values: nDimension = " << dimension << ", nRuns = "
          << runs << ", nGPUs = " << num_gpus
          << std::endl;
      std::cout << std::endl << std::endl;
    }
    output = true;
    warmup = false;
  } else {
    dimension = atoi(argv[1]);
    runs = atoi(argv[2]);
    num_gpus = atoi(argv[3]);
    output = false;
#ifdef __CUDACC__
    warmup = true;
#endif
  }

  msl::setNumRuns(runs);
  msl::setNumGpus(num_gpus);

  // warmup
  if (warmup) {
    msl::Timer tw("Warmup");
    frobenius<float>(dimension, false);
    msl::printv("Warmup: %fs\n", tw.stop());
  }

  msl::startTiming();
  for (int run = 0; run < msl::Muesli::num_runs; ++run) {
    frobenius<float>(dimension, output);
    msl::splitTime(run);
  }
  msl::stopTiming();

  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}
