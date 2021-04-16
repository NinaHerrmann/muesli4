
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

        class NeutralValueFunctor : public Functor2<int, int, float> {
        public:
            NeutralValueFunctor(int glob_rows, int glob_cols, float default_neutral)
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
        class SweepFunctor
                : public DMMapStencilFunctor<float, float, NeutralValueFunctor> {
        public:
            SweepFunctor() : DMMapStencilFunctor(){}
//
            MSL_USERFUNC
            float operator()(
                    int rowIndex, int colIndex, float *input, int ncol, int nrow, float *paddingborder) const {
                float sum = 0;
                // Add top and bottom values.

                for (int i = -stencil_size; i <= stencil_size; i++) {
                    if (i == 0)
                        continue;
                    float value = 0;
                    // top broder must be 100;
                    if (rowIndex + i < 0) {
                        value = paddingborder[colIndex];
                    } else {
                        if (rowIndex + i > (nrow - 1)) {
                            int where = ncol + colIndex;
                            value = paddingborder[where];
                        } else {
                            int index = (rowIndex + i) * ncol + colIndex;
                            value = input[index];
                        }
                    }

                    sum += value;
                }

                // Add left and right values.
                for (int i = -stencil_size; i <= stencil_size; i++) {
                    if (i == 0)
                        continue;
                    float value = 0;
                    if (colIndex + i < 0 || colIndex + i > (ncol - 1)) {
                        value = 100;
                    } else {
                        int index = rowIndex * ncol + colIndex + i;
                        value = input[index];
                    }
                    sum += value;
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
            NeutralValueFunctor neutral_value_functor(n, m, 75);
            DM<float> mat(n, m, 75, true);

            AbsoluteDifference difference_functor;
            Max max_functor;
            float global_diff = 10;

            // mapStencil
            SweepFunctor jacobi;
            jacobi.setStencilSize(1);
            jacobi.setTileWidth(tile_width);
            //jacobi.setNVF(neutral_value_functor);

            // Neutral value provider
            DM<float> differences(n, m, 0, true);
            DM<float> test_m(n, m, 75, true);
            DM<float> test2_m(n, m, 75, true);

            int num_iter = 0;
         /*   float  milliseconds, t6= 0, t7= 0, t8= 0, t9=0, t10=0;
            auto start = std::chrono::high_resolution_clock::now();
            auto stop = std::chrono::high_resolution_clock::now();
            double tstencil = 0, tinplace = 0, tzip= 0, tfold= 0, tmove= 0;
            std::chrono::duration<double> tstencil1 = stop-start, tinplace1 = stop-start, tzip1= stop-start, tfold1= stop-start, tmove1= stop-start;

*/
            //float milliseconds,maps , diffs, difffolds, move = 0;
            while (global_diff > EPSILON && num_iter < MAX_ITER) {
                if (num_iter % 50 == 0) {
                    //printf("calc with test_m\n");
                    //start = std::chrono::high_resolution_clock::now();
                    test_m.mapStencilMM(test2_m, jacobi, neutral_value_functor);
                 /*   stop = std::chrono::high_resolution_clock::now();
                    tinplace1 = stop - start;
                    tinplace += tstencil1.count();
                    start = std::chrono::high_resolution_clock::now();*/
                    differences = test_m.zip(test2_m, difference_functor);
                    /*stop = std::chrono::high_resolution_clock::now();
                    tzip1 = stop - start;
                    tzip += tstencil1.count();
                    start = std::chrono::high_resolution_clock::now();*/
                    global_diff = differences.fold(max_functor, true);
                    /*stop = std::chrono::high_resolution_clock::now();
                    tfold1 = stop - start;
                    tfold += tstencil1.count();*/
                } else {
                    if (num_iter % 2 == 0) {
                        //start = std::chrono::high_resolution_clock::now();
                        test_m.mapStencilMM(test2_m, jacobi, neutral_value_functor);
                        /*stop = std::chrono::high_resolution_clock::now();
                        tinplace1 = stop - start;
                        tinplace += tinplace1.count();*/

                    } else {
                        //start = std::chrono::high_resolution_clock::now();
                        test2_m.mapStencilMM(test_m, jacobi, neutral_value_functor);
//                        stop = std::chrono::high_resolution_clock::now();
//                        tinplace1 = stop - start;
//                        tinplace += tinplace1.count();
                    }
                }
                num_iter++;
            }
            //printf("\nStencil %.3fs; InPlace %.3f; Zip %.3f; Fold %.3f; Move %.3f; \n", tstencil * 1000, tinplace* 1000, tzip* 1000, tfold* 1000, tmove* 1000);

            /*test_m.download();
            test_m.show("test_m");
            test2_m.download();
            test2_m.show("test2_m");
            differences.download();
            differences.show("othermatrix");*/
            if (msl::isRootProcess()) {
                printf("R:%d;%.2f;", num_iter, global_diff);

                //test2_m.download();
                //test2_m.show("transfer");
                //test_m.printTime();
                //test2_m.printTime();
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
