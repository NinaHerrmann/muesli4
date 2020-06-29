/*
 * matmult.cpp
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

namespace test {

namespace matmult {

double valA = 1.0;
double valB = 0.01;

template <typename T>
void checkResults(DMatrix<T>& dmC, int n)
{
  T** C = new T*[n];
  for (int i = 0; i < n; i++) {
    C[i] = new T[n];
  }
  dmC.gather(C);
  if (isRootProcess()) {
    bool correct = true;
    double eps = 1.e-6;
#pragma omp parallel for
	for (int i = 0; i < n; i++) {
	  for (int j = 0; j < n; j++) {
      double abs_err = fabs(C[i][j] - (n * valB));
      double dot_length = n;
      double abs_val = fabs(C[i][j]);
      double rel_err = abs_err/abs_val/dot_length ;

      if (rel_err > eps) {
        //printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, C[i][j], n*valB, eps);
        correct = false;
      }
	  }
	}
	printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
  }

  for (int i = 0; i < n; i++) {
	  delete[] C[i];
  }
  delete[] C;
}

template <typename T>
T initA(int a, int b)
{
  return a * 4 + b;
}

template <typename T>
T initB(int a, int b)
{
  return 16 + a * 4 + b;
}

template <typename T>
struct dotproduct : public MMapIndexFunctor<T, T>
{
  LMatrix<T> A, B;

  dotproduct(DMatrix<T>& _A, DMatrix<T>& _B)
    : A(_A), B(_B, Distribution::COPY)
  {
    this->addArgument(&A);
    this->addArgument(&B);
    this->setLocalIndices(true);
  }

  MSL_USERFUNC
  T operator()(int i, int j, T Cij) const
  {
    return simple(i, j, Cij);
  }

  MSL_USERFUNC
  T simple(int i, int j, T Cij) const
  {
    T sum = Cij;

    for (int k = 0; k < this->mLocal; k++) {
      sum += A[i][k] * B[k][j]; // B transposed
    }

    return sum;
  }
};

template <typename T>
DMatrix<T>& matmult(DMatrix<T>& A, DMatrix<T>& B, DMatrix<T>* C)
{
  // Initial shifting
  auto negate = [] (int a) {return -a;};
  A.rotateRows(negate);
  B.rotateCols(negate);

  // Submatrix multiplication + stepwise shifting
  dotproduct<T> dp(A, B);
  for (int i = 0; i < A.getBlocksInRow(); ++i) {
    C->mapIndexInPlace(dp);
    A.rotateRows(-1);
    B.rotateCols(-1);
  }

  // Final shifting
  auto identity = [] (int a) {return a;};
  A.rotateRows(identity);
  B.rotateCols(identity);

  return *C;
}

template <typename T>
void testMatMult(int n, bool output, bool check)
{
  int sqrtp = (int) (sqrt((double) Muesli::num_local_procs) + 0.1);
  DMatrix<T> A(n, n, sqrtp, sqrtp, &initA<T>);
  DMatrix<T> B(n, n, sqrtp, sqrtp, &initB<T>);

  // output matrices A and B
  if (output) {
    A.show("A");
    B.show("B");
  }

  // matrix multiplication C=A*B
  DMatrix<T> C(n, n, sqrtp, sqrtp, (T)0.0);
  C = matmult(A, B, &C);

  // output matrix C
  if (output) {
    C.show("C = A x B");
  }

  // check results
  if (check) {
    checkResults(C, n);
  }
}

} // namespace matmult
} // namespace test
} // namespace msl

int main(int argc, char** argv)
{
  msl::initSkeletons(argc, argv);

  int size = 128; int nRuns = 1; int nGPUs = 1;
  bool output = true; bool checkResult = 0; bool noWarmup = 1;
  if (argc < 4) {
    if (msl::isRootProcess()) {
      std::cout << std::endl << std::endl;
      std::cout << "Usage: " << argv[0] << " #size #nRuns #nGPUs" << std::endl;
      std::cout << "Default values: size = " << size
                << ", nRuns = " << nRuns
                << ", nGPUs = " << nGPUs
                << std::endl << std::endl << std::endl;
    }
    checkResult = 1;
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

  // warmup
  if (!noWarmup) {
    msl::Timer tw("Warmup");
    msl::test::matmult::testMatMult<double>(size, 0, 0);
    msl::printv("Warmup: %fs\n", tw.stop());
  }

  msl::startTiming();
  for (int run = 0; run < msl::Muesli::num_runs; run++) {
    msl::test::matmult::testMatMult<double>(size, output, checkResult);
    msl::splitTime(run);
  }
  msl::stopTiming();

  msl::terminateSkeletons();
  return 0;
}











