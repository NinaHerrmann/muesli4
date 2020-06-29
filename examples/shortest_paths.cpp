/*
 * shortest_paths.cpp
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

namespace msl {

namespace examples {

namespace shortest {

// Divide INT_MAX by 2 to prevent arithmetic overflow
// when calculating infinity+infinity.
// See struct min_plus_matmult function call operator
int infinity = getPositiveInfinity<int>()/2;
//int infinity = 99999;

int log2(int x)
{
  int result = 0;
  while (x >>= 1)
    result++;
  return result;
}

inline int negate(const int a)
{
  return -a;
}

int identity(int a)
{
  return a;
}

inline int init(const int a, const int b)
{
  return abs(a - b) <= 1 ? abs(a - b) : infinity;
}

struct MinPlus: public MMapIndexFunctor<int, int>
{
  int tw;
  LMatrix<int> A, B;

  MinPlus(DMatrix<int>& _A, DMatrix<int>& _B)
    : tw(16), A(_A), B(_B, Distribution::COPY)
  {
    this->addArgument(&A);
    this->addArgument(&B);
    this->setLocalIndices(true);
    this->setTileWidth(tw);
  }

  MSL_USERFUNC
  int min(int a, int b) const
  {
    return a <= b ? a : b;
  }

  MSL_USERFUNC
  int operator()(int i, int j, int Cij) const
  {
    return tiling(i, j, Cij);
  }

  MSL_USERFUNC
  int simple(int i, int j, int Cij) const
  {
    int minimum = Cij;

    for (int k = 0; k < this->mLocal; k++) {
      minimum = min(minimum, A[i][k] + B[j][k]); // B transposed
    }

    return minimum;
  }

  MSL_USERFUNC
  int tiling(int i, int j, int Cij) const
  {
    int minimum = Cij;

    for (int p = 0; p < this->mLocal/tw; p++) {
      auto t_A = A.getRowTile(p, i, this);
      auto t_B = B.getColTile(p, j, this);

      for (int k = 0; k < tw; k++) {
        minimum = min(minimum, t_A.get(k) + t_B.get(k));
      }
    }

    return minimum;
  }
};

DMatrix<int>& min_plus_matmult(DMatrix<int>& A, DMatrix<int>& B, DMatrix<int>* R)
{
  A.rotateRows(negate);
  B.rotateCols(negate);

  MinPlus mp(A, B);
  for (int i = 0; i < A.getBlocksInRow(); i++) {
    R->mapIndexInPlace(mp);
    A.rotateRows(-1);
    B.rotateCols(-1);
  }

  A.rotateRows(&identity);
  B.rotateCols(&identity);

  return *R;
}

void shortest(int n, bool output)
{
  // calculate square root of num processes
  int sqrtp = (int) (sqrt((double) Muesli::num_total_procs) + 0.1);
  // create matrix A
  msl::DMatrix<int> A(n, n, sqrtp, sqrtp, init);
  // create result matrix R
  msl::DMatrix<int> R(n, n, sqrtp, sqrtp, infinity);

  // output matrix A
  if (output) {
    A.show("A");
  }

  for (int i = 0; i < log2(n); i++) {
    msl::DMatrix<int> Acopy(A);
    // A = A*A
    A = min_plus_matmult(A, Acopy, &R);
  }

  // output shortest paths matrix
  if (output) {
    R.show("Shortest_paths(A)");
  }
}

} // namespace shortest
} // namespace examples
} // namespace msl

int main(int argc, char** argv)
{
  msl::initSkeletons(argc, argv);

  int size = 32; int nRuns = 1; int nGPUs = 2;
  bool output = 0; bool noWarmup = 1;
  if (argc < 4) {
    if (msl::isRootProcess()) {
      std::cout << std::endl << std::endl;
      std::cout << "Usage: " << argv[0] << " #size #numRuns" << std::endl;
      std::cout << "Default values: size = " << size
                      << ", nRuns = " << nRuns
                      << ", nGPUs = " << nGPUs
                      << std::endl << std::endl << std::endl;
    }
    output = 1;
  } else {
    size = atoi(argv[1]);
    nRuns = atoi(argv[2]);
    nGPUs = atoi(argv[3]);
    // warmup only for GPUs
#ifdef __CUDACC__
    noWarmup = 0;
#endif
  }

  msl::setNumRuns(nRuns);
  msl::setNumGpus(nGPUs);
//  msl::setThreadsPerBlock(TW*TW);

  // warmup
  if (!noWarmup) {
    msl::Timer tw("Warmup");
    msl::examples::shortest::shortest(size, 0);
    msl::printv("Warmup: %fs\n", tw.stop());
  }

  msl::startTiming();
  for (int run = 0; run < msl::Muesli::num_runs; run++) {
    msl::examples::shortest::shortest(size, output);
    msl::splitTime(run);
   }
   msl::stopTiming();


  msl::terminateSkeletons();
  return 0;
}





