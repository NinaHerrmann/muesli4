/**
 * Copyright (c) 2020 Nina Herrmann
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */
#include <algorithm>
#include "dm.h"
#include "muesli.h"

#define EPSILON 0.03
#define MAX_ITER 1000
namespace msl {

    namespace jacobi {

        class GoLNeutralValueFunctor : public Functor2<int, int, int> {
        public:
            GoLNeutralValueFunctor(int default_neutral)
                    : default_neutral(default_neutral) {}

            MSL_USERFUNC
            int operator()(int x, int y) const {
                // All Border are not populated.
                return default_neutral;
            }

        private:
            int default_neutral = 0;
        };

/**
 * @brief Averages the top, bottom, left and right neighbours of a specific
 * element
 *
 */
        class GoLFunctor
                : public DMMapStencilFunctor<int, int, GoLNeutralValueFunctor> {
        public:
            GoLFunctor() : DMMapStencilFunctor(){}
//
            MSL_USERFUNC
            int operator()(
                    int rowIndex, int colIndex, int *input, int ncol, int nrow, int *paddingborder) const {
                int value = 0;
                int sum = 0;
                // top broder must be 100;
                int index = (rowIndex) * ncol + colIndex;
                int live_status = input[index];
                for (int rowoffset = -stencil_size; rowoffset <= stencil_size; rowoffset++) {
                    for (int coloffset = -stencil_size; coloffset <= stencil_size; coloffset++) {
                        if (rowoffset == 0 && coloffset == 0)
                            continue;
                        if (rowIndex + rowoffset < 0 || colIndex + coloffset < 0 || colIndex + coloffset > (ncol - 1) ||
                            rowIndex + rowoffset > (nrow - 1)) {
                            // TODO: nvf
                            value = 0;
                        } else {
                            int indexoffset = (rowIndex + rowoffset) * ncol + colIndex + coloffset;
                            value = input[indexoffset];
                        }
                        sum += value;
                    }
                }
                int future_live_status = 0;
                // If the cell is alive and has 2-3 neighbours it survives
                if (live_status == 1 && (sum == 2 || sum == 3 )) {
                    future_live_status = 1;
                }
                // If the cell is dead and has 3 neighbours it gets alive
                if (live_status == 0 && sum == 3) {
                    future_live_status = 1;
                }

                return future_live_status;
            }
        };

        // Check if the population replicates itself
        class SelfReplication : public Functor2<float, float, float> {
        public:
            MSL_USERFUNC
            float operator()(float x, float y) const {
                if (x == y) {
                    return 0;
                }
                return 1;
            }
        };

        // Check if at least one thing is alive
        class Max : public Functor2<float, float, float> {
        public:
            MSL_USERFUNC
            float operator()(float x, float y) const {
                if (x > y)
                    return x;

                return y;
            }
        };


        int run(int n, int m, int stencil_radius, int tile_width, int iterations) {
            GoLNeutralValueFunctor dead_nvf(0);
            Max max_functor;
            int global_diff = 10;

            // mapStencil
            GoLFunctor GoL;
            GoL.setStencilSize(1);
            GoL.setTileWidth(tile_width);

            // Neutral value provider
            DM<int> differences(n, m, 1, true);
            DM<int> data1(n, m, 0, true);
            DM<int> data2(n, m, 0, true);

            SelfReplication difference_functor;

            int num_iter = 0;
            int maxnumberalive = 1;
            for (int i = 0; i < n * m; i++) {
                data1.set(i, rand() % 2);
            }
            data1.download();
            //data1.show("start");
            while (global_diff > 0 && num_iter < iterations && maxnumberalive > 0) {
                if (num_iter % 50 == 0) {
                    data1.mapStencilMM(data2, GoL, dead_nvf);
                    differences = data1.zip(data2, difference_functor);
                    global_diff = data2.fold(max_functor, true);
                    maxnumberalive = differences.fold(max_functor, true);
                } else {
                    if (num_iter % 2 == 0) {
                        data1.mapStencilMM(data2, GoL, dead_nvf);
                    } else {
                        data2.mapStencilMM(data1, GoL, dead_nvf);
                    }
                }
                num_iter++;
            }
            if (msl::isRootProcess()) {
                if (maxnumberalive == 0 ){
                    printf("no more living;");
                }
                if (num_iter < MAX_ITER){
                    printf("iteration reached %d %d;", global_diff, maxnumberalive);
                }
                if (global_diff == 0){
                    printf("no difference any more;");
                }
                printf("R:%d;", num_iter);
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
    int iterations = MAX_ITER;
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
        iterations = atoi(argv[7]);

    }
    if (argc == 9) {
        file = argv[8];
    }

    msl::setNumGpus(nGPUs);
    msl::setNumRuns(nRuns);

    if (msl::isRootProcess()) {
//        printf("%d; %d; %.2f; %d", n, nGPUs, msl::Muesli::cpu_fraction, msl::Muesli::num_runs);
        //printf("Config:\tSize:%d; #GPU:%d; CPU perc:%.2f;", n, nGPUs, msl::Muesli::cpu_fraction);
    }
    msl::startTiming();
    for (int r = 0; r < msl::Muesli::num_runs; ++r) {
        msl::jacobi::run(n, m, stencil_radius, tile_width, iterations);
        msl::splitTime(r);
    }

    if (file) {
        std::string id = "" + std::to_string(n) + ";" + std::to_string(nGPUs) + ";" + std::to_string(tile_width) +
                         ";" + std::to_string(msl::Muesli::cpu_fraction * 100) + ";";
        msl::printTimeToFile(id.c_str(), file);
    } else {
        msl::stopTiming();
    }
    msl::terminateSkeletons();
    return 0;
}
