/*
 * jacobi.cpp
 *
 *      Author: Steffen Ernsting <endizhupani@uni-muenster.de>
 *
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2020 Endi Zhupani <endizhupani@uni-muenster.de>
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

#include "dm.h"
#include "muesli.h"
#define EPSILON 0.01
#define MAX_ITER 10000
namespace msl {

namespace jacobi {

// template <typename T> T add(int a, int b) { return (T)(a + b); }

class Jacobi : public StencilFunctor<float, float> {
public:
  Jacobi() : StencilFunctor(1) {}

  MSL_USERFUNC
  float operator()(int tile_local_row, int tile_local_col, int tile_cols,
                   int tile_rows, float *tile_elements) const {

    if (tile_local_row == 0 ||
        tile_local_row == tile_rows - 1 - stencil_radius_) {
      return tile_elements[tile_local_row * tile_cols + tile_local_col];
    }

    if (tile_local_col == 0 ||
        ctile_local_colol == tile_cols - 1 - stencil_radius) {
      return tile_elements[tile_local_row * tile_cols + tile_local_col];
    }

    float sum = 0;
    for (int i = -stencil_radius_; i <= stencil_radius_; i++) {
      if (i == 0)
        continue;
      sum += tile_elements[(tile_local_row + i) * tile_cols + tile_local_col];
    }

    for (int i = -stencil_radius_; i <= stencil_radius_; i++) {
      if (i == 0)
        continue;
      sum += tile_elements[tile_local_row * tile_cols + tile_local_col + i];
    }
    return sum / (4 * stencil_radius_);
  }
};

class JacobiSweepFunctor : public MMapStencilFunctor<float, float> {
  MSL_USERFUNC
  float operator()(int rowIndex, int colIndex,
                   const msl::PLMatrix<float> &input) const {
    float sum = 0;
    for (int i = -stencil_radius_; i <= stencil_radius_; i++) {
      if (i == 0)
        continue;
      sum += input.get(row + i, col);
    }

    for (int i = -stencil_radius_; i <= stencil_radius_; i++) {
      if (i == 0)
        continue;
      sum += input.get(row, col + i);
    }
    return sum / (4 * stencil_radius_);
  }
}

class AbsoluteDifference : public Functor2<float, float, float> {
  MSL_USERFUNC
  float operator()(float x, float y) {
    auto diff = x - y;
    if (diff < 0) {
      diff *= (-1);
    }
    return diff;
  }
}

class JacobiNeutralValueFunctor : public Functor2<int, int, float> {
public:
  // Global number of rows
  int glob_rows_;

  // Global number of columns
  int glob_cols_;

  JacobiNeutralValueFunctor(int glob_rows, int glob_cols)
      : glob_cols_(glob_cols), glob_rows_(glob_rows) {}
  MSL_USERFUNC
  float operator()(int x, int y) { // here, x represents rows
    // left and right column must be 100;
    if (y == 0 || y == (glob_cols_ - 1)) {
      return 100;
    }

    // top broder must be 100;
    if (x == 0) {
      return 100;
    }

    // bottom border must be 0
    if (x == glob_rows_ - 1) {
      return 0;
    }

    // this should never be called if indexes don't represent border points
    // inner values are 75
    return 75;
  }
}

class Max : public Functor2<float, float, float> {
  MSL_USERFUNC
  float operator()(float x, float y) {
    if (x > y)
      return x;

    return y;
  }
}

} // namespace jacobi
} // namespace msl

int run(int n, int m, int stencil_radius) {
  msl::DM<float> mat(m, n, 75);

  msl::jacobi::AbsoluteDifference difference_functor();
  msl::jacobi::Max max_functor();
  float global_diff = 10;

  // mapStencil
  msl::jacobi::Jacobi jacobi();

  // Neutral value provider
  msl::jacobi::JacobiNeutralValueFunctor neutral_value_functor(n, m);

  int num_iter;
  while (global_diff > EPSILON && num_iter < MAX_ITER) {
    msl::DM<float> new_m = mat.mapStencil(jacobi, neutral_value_functor);

    if (num_iter % 4 == 0) {
      msl::DM<float> differences = new_m.zip(mat, difference_functor);
      global_diff = differneces.fold(max_functor, true);
    }
    num_iter++;
  }

  return 0;
}

int main(int argc, char **argv) {
  msl::initSkeletons(argc, argv);

  int n = 32;
  int m = 32;
  int stencil_radius = 1;
  int nGPUs = 2;
  int nRuns = 1;
  bool warmup = false;
  if (argc == 5) {
    n = atoi(argv[1]);
    m = atoi(argv[2]);
    nGPUs = atoi(argv[3]);
    nRuns = atoi(argv[4]);
#ifdef __CUDACC__
    warmup = true;
#endif
  }

  msl::setNumGpus(nGPUs);
  msl::setNumRuns(nRuns);

  msl::startTiming();
  for (int run = 0; run < msl::Muesli::num_runs; ++run) {
    // Create matrix.
    run(n, m, stencil_radius);
    // mat.show();
    msl::splitTime(run);
  }
  msl::stopTiming();

  msl::terminateSkeletons();
}
