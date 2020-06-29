/*
 * jacobi.cpp
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

#include "muesli.h"
#include "dmatrix.h"

namespace msl {

namespace jacobi {

template <typename T>
T add(int a, int b)
{
  return (T) (a + b);
}

class Jacobi : public MMapStencilFunctor<int, int>
{
public:
  Jacobi(int ss, int tw)
	: stencil_size(ss)
  {
    this->setTileWidth(tw);
    this->setStencilSize(stencil_size);
  }

  MSL_USERFUNC
  int operator() (int row, int col, const msl::PLMatrix<int>& input) const
  {
    int sum = 0;
    for (int i = -stencil_size; i <= stencil_size; i++) {
      for (int j = -stencil_size; j <= stencil_size; j++) {
        //if (!(i == 0 && j == 0))
          sum += input.get(row+i, col+j);
      }
    }
    return sum/(4*stencil_size+1);
  }
private:
  int stencil_size;
};

} // namespace jacobi
} // namspace msl

int main(int argc, char** argv)
{
  msl::initSkeletons(argc, argv);

  int n = 32; int m = 32;
  int tile_width = -1; int ss = 1;
  int nGPUs = 2; int nRuns = 1;
  bool warmup = false;
  if (argc == 7) {
    n = atoi(argv[1]);
    m = atoi(argv[2]);
    tile_width = atoi(argv[3]);
    ss = atoi(argv[4]);
    nGPUs = atoi(argv[5]);
    nRuns = atoi(argv[6]);
#ifdef __CUDACC__
    warmup = true;
#endif
  }

  msl::setNumGpus(nGPUs);
  msl::setNumRuns(nRuns);

  // warmup
  if (warmup) {
    msl::Timer tw("Warmup");
    // Create matrix.
    msl::DMatrix<int> mat(n, m, msl::Muesli::num_total_procs, 1, &msl::jacobi::add<int>);

    // mapStencil
    msl::jacobi::Jacobi jacobi(ss, tile_width);
    int neutral_value = 1;
    mat.mapStencilInPlace(jacobi, neutral_value);
    msl::printv("Warmup: %fs\n", tw.stop());
  }

  msl::startTiming();
  for (int run = 0; run < msl::Muesli::num_runs; ++run) {
    // Create matrix.
    msl::DMatrix<int> mat(n, m, msl::Muesli::num_total_procs, 1, &msl::jacobi::add<int>);

    // mapStencil
    msl::jacobi::Jacobi jacobi(ss, tile_width);
    int neutral_value = 1;
    mat.mapStencilInPlace(jacobi, neutral_value);
    //mat.show();
    msl::splitTime(run);
  }
  msl::stopTiming();

  msl::terminateSkeletons();
}





