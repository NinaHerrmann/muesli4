
/**
 * Copyright (c) 2020 Endi Zhupani
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */

#include "dm.h"
#include "muesli.h"
#include <algorithm>
#define EPSILON 0.01
#define MAX_ITER 10000
namespace msl {

namespace jacobi {

/**
 * @brief Averages the top, bottom, left and right neighbours of a specific
 * element
 *
 */
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
  JacobiNeutralValueFunctor(int glob_rows, int glob_cols, float default_neutral)
      : glob_cols_(glob_cols), glob_rows_(glob_rows),
        default_neutral(default_neutral) {}
  MSL_USERFUNC
  float operator()(int x, int y) { // here, x represents rows
    // left and right column must be 100;
    if (y < 0 || y >= glob_cols_) {
      return 100;
    }

    // top broder must be 100;
    if (x < 0) {
      return 100;
    }

    // bottom border must be 0
    if (x >= glob_rows_) {
      return 0;
    }

    // this should never be called if indexes don't represent border points
    // inner values are 75
    return default_neutral;
  }

private:
  // Global number of rows
  int glob_rows_;

  // Global number of columns
  int glob_cols_;

  int default_neutral;
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
  msl::DM<float> mat(m, n, 75, true);

  msl::jacobi::AbsoluteDifference difference_functor();
  msl::jacobi::Max max_functor();
  float global_diff = 10;

  // mapStencil
  msl::jacobi::Jacobi jacobi();

  // Neutral value provider
  msl::jacobi::JacobiNeutralValueFunctor neutral_value_functor(n, m, 75);

  int num_iter;
  while (global_diff > EPSILON && num_iter < MAX_ITER) {
    msl::DM<float> new_m = mat.mapStencil(jacobi, neutral_value_functor);

    if (num_iter % 4 == 0) {
      msl::DM<float> differences = new_m.zip(mat, difference_functor);
      global_diff = differneces.fold(max_functor, true);
    }
    std::swap(new_m, mat);
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
