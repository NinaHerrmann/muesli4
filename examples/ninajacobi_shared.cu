
/**
 * Copyright (c) 2020 Nina Herrmann, Endi Zhupani
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */
#include <algorithm>
#include "dm.h"
#include "muesli.h"

#define EPSILON 0.03
#define MAX_ITER 5000
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

            int default_neutral = 75;
        };

/**
 * @brief Averages the top, bottom, left and right neighbours of a specific
 * element
 *
 */
        class JacobiSweepFunctor
                : public MMapStencilFunctor<float, float, JacobiNeutralValueFunctor> {
        public:
            JacobiSweepFunctor() : MMapStencilFunctor(){}

            MSL_USERFUNC
            float operator()(
                    int rowIndex, int colIndex,
                    const msl::PLMatrix<float> &input) const {
                float sum = 0;
                // Add top and bottom values.
                for (int i = -stencil_size; i <= stencil_size; i++) {
                    if (i == 0)
                        continue;
                    sum += input.get(rowIndex + i, colIndex);
                }

                // Add left and right values.
                for (int i = -stencil_size; i <= stencil_size; i++) {
                    if (i == 0)
                        continue;
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

        class zipIdentity : public Functor2<float, float, float> {
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
        class CopyFunctor : public Functor<float, float>{
        public:
            MSL_USERFUNC
            float operator()(float y) const {
                return y;
            }
        };

        int run(int n, int m, int stencil_radius, int tile_width) {
            JacobiNeutralValueFunctor neutral_value_functor(n, m, 75);
            DM<float> mat(n, m, 75, true);

            AbsoluteDifference difference_functor;
            Max max_functor;
            float global_diff = 10;

            // mapStencil
            JacobiSweepFunctor jacobi;
            jacobi.setStencilSize(1);
            jacobi.setTileWidth(tile_width);
            //jacobi.setNVF(neutral_value_functor);

            // Neutral value provider
            DM<float> new_m(n, m, 75, true);
            int num_iter = 0;
            float milliseconds,maps , diffs, difffolds, move = 0;
            while (global_diff > EPSILON && num_iter < MAX_ITER) {
                if (num_iter % 4 == 0) {
                    new_m = mat.mapStencil(jacobi, neutral_value_functor);
                    DM<float> differences = new_m.zip(mat, difference_functor);
                    global_diff = differences.fold(max_functor, true);
                    mat = std::move(new_m);
                } else {
                    mat.mapStencilInPlace(jacobi, neutral_value_functor);
                }
                num_iter++;
            }

            if (msl::isRootProcess()) {
                printf("R:%d;", num_iter);
                //mat.download();
                //mat.show("matrix");
                //printf("\n mapstencil %.3fs;\n", maps / 1000);
                //printf("differences zip %.3fs;\n", diffs / 1000);
                //printf("differences fold %.3fs;\n", difffolds / 1000);
                //printf("Move %.3fs;\n", move / 1000);
                //printf("It %d;\n", num_iter);
            }
            return 0;
        }

    } // namespace jacobi
} // namespace msl
int main(int argc, char **argv) {
    msl::initSkeletons(argc, argv);
    int n = 500;
    int m = 500;
    int stencil_radius = 1;
    int nGPUs = 1;
    int nRuns = 1;
    int tile_width = msl::DEFAULT_TILE_WIDTH;
    msl::Muesli::cpu_fraction = 0.25;
    //bool warmup = false;

    char *file = nullptr;

    if (argc >= 6) {
        n = atoi(argv[1]);
        m = atoi(argv[2]);
        nGPUs = atoi(argv[3]);
        nRuns = atoi(argv[4]);
        msl::Muesli::cpu_fraction = atof(argv[5]);
        //printf("pass %.2f", atof(argv[5]));
        if (msl::Muesli::cpu_fraction > 1) {
            msl::Muesli::cpu_fraction = 1;
        }
        tile_width = atoi(argv[6]);

    }
    if (argc == 8) {
        file = argv[7];
    }

    msl::setNumGpus(nGPUs);
    msl::setNumRuns(nRuns);

    if (msl::isRootProcess()) {
//        printf("%d; %d; %.2f; %d", n, nGPUs, msl::Muesli::cpu_fraction, msl::Muesli::num_runs);
        //printf("Config:\tSize:%d; #GPU:%d; CPU perc:%.2f;", n, nGPUs, msl::Muesli::cpu_fraction);
    }
    msl::startTiming();
    for (int r = 0; r < msl::Muesli::num_runs; ++r) {
        msl::jacobi::run(n, m, stencil_radius, tile_width);
        msl::splitTime(r);
    }

    if (file) {
        std::string id = "" + std::to_string(n) + ";" + std::to_string(nGPUs) +
                         ";" + std::to_string(msl::Muesli::cpu_fraction * 100) + ";";
        msl::printTimeToFile(id.c_str(), file);
    } else {
        msl::stopTiming();
    }
    msl::terminateSkeletons();
    return 0;
}
