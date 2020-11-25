
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
#define MAX_ITER 1
namespace msl {

namespace jacobi {

class JacobiNeutralValueFunctor : public Functor2<int, int, float> {
public:
  JacobiNeutralValueFunctor(int glob_rows, int glob_cols, float default_neutral)
      : glob_cols_(glob_cols), glob_rows_(glob_rows),
        default_neutral(default_neutral) {}
  MSL_USERFUNC
  float operator()(int x, int y) const { // here, x represents rows
    // left and right column must be 100;
    if (y < 0 || y > (glob_cols_ - 1)) {
      return 100;
    }

    // top broder must be 100;
    if (x < 0) {
      return 100;
    }

    // bottom border must be 0
    if (x > (glob_rows_ - 1)) {
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
};
/**
 * @brief Averages the top, bottom, left and right neighbours of a specific
 * element
 *
 */
class JacobiSweepFunctor
    : public MMapStencilFunctor<float, float, JacobiNeutralValueFunctor> {
public:
  JacobiSweepFunctor() : MMapStencilFunctor() {}
  MSL_USERFUNC
  float operator()(
      int rowIndex, int colIndex,
      const msl::PLMatrix<float, JacobiNeutralValueFunctor> &input) const {
    float sum = 0;
    for (int i = -stencil_size; i <= stencil_size; i++) {
      if (i == 0)
        continue;
      // printf("%d, %d is: %f\n", rowIndex + i, colIndex,
      // input.get(rowIndex + i, colIndex));
      sum += input.get(rowIndex + i, colIndex);
    }

    for (int i = -stencil_size; i <= stencil_size; i++) {
      if (i == 0)
        continue;
      // printf("%d, %d is: %f\n", rowIndex, colIndex + i,
      // input.get(rowIndex, colIndex + i));
      sum += input.get(rowIndex, colIndex + i);
    }
    return sum / (4 * stencil_size);
  }
};

class AbsoluteDifference : public Functor2<float, float, float> {
public:
  MSL_USERFUNC
  float operator()(float x, float y) const {
    auto diff = x - y;
    if (diff < 0) {
      diff *= (-1);
    }
    return diff;
  }
};

class Max : public Functor2<float, float, float> {
public:
  MSL_USERFUNC
  float operator()(float x, float y) const {
    if (x > y)
      return x;

    return y;
  }
};

int run(int n, int m, int stencil_radius) {
  DM<float> mat(m, n, 75, true);

  AbsoluteDifference difference_functor;
  Max max_functor;
  float global_diff = 10;

  // mapStencil
  JacobiSweepFunctor jacobi;
  jacobi.setStencilSize(1);

  // Neutral value provider
  JacobiNeutralValueFunctor neutral_value_functor(n, m, 75);

  int num_iter = 0;
  while (global_diff > EPSILON && num_iter < MAX_ITER) {
    DM<float> new_m = mat.mapStencil(jacobi, neutral_value_functor);

    // if (num_iter % 4 == 0) {
    //   DM<float> differences = new_m.zip(mat, difference_functor);
    //   global_diff = differences.fold(max_functor, true);
    // }
    std::swap(new_m, mat);
    num_iter++;
  }
  mat.show();
  return 0;
}
} // namespace jacobi
} // namespace msl
int main(int argc, char **argv) {
  msl::initSkeletons(argc, argv);

  int n = 10;
  int m = 10;
  int stencil_radius = 1;
  int nGPUs = 1;
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
  msl::setNumThreads(1);
  msl::Muesli::cpu_fraction = 0.25;

  msl::startTiming();
  for (int r = 0; r < msl::Muesli::num_runs; ++r) {
    msl::jacobi::run(n, m, stencil_radius);
    msl::splitTime(r);
  }
  msl::stopTiming();

  msl::terminateSkeletons();

  return 0;
}
