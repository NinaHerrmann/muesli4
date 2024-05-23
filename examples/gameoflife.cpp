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
namespace msl::gameoflife {

        /**
         * @brief https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
         */
        class GoLFunctor
                : public DMMapStencilFunctor<int, int> {
        public:
            GoLFunctor() : DMMapStencilFunctor(){}
            MSL_USERFUNC
            int operator()(
                    int rowIndex, int colIndex, PLMatrix<int> *input, int ncol, int nrow) const {
                int sum = 0;

                sum += input->get(rowIndex-1, colIndex) + input->get(rowIndex-1, colIndex-1)+ input->get(rowIndex-1, colIndex+1)
                        + input->get(rowIndex+1, colIndex)+ input->get(rowIndex+1, colIndex-1)+ input->get(rowIndex+1, colIndex+1)
                        + input->get(rowIndex, colIndex-1)+ input->get(rowIndex, colIndex+1);
                int live_status = input->get(rowIndex, colIndex);

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


        int run(int n, int m, int stencil_radius, int tile_width, int iterations, int iterations_used, char *file) {
            double start = MPI_Wtime();
            Max max_functor;

            // mapStencil
            GoLFunctor GoL;
            GoL.setStencilSize(1);
            GoL.setTileWidth(tile_width);

            // Neutral value provider
            DM<int> differences(n, m, 1, true);
            DM<int> data1(n, m, 0, true);
            DM<int> data2(n, m, 0, true);

            SelfReplication difference_functor;

            //int num_iter = 0;
            for (int i = 0; i < n * m; i++) {
                data1.set(i, rand() % 2);//data1.set(i, rand() % 2);
            }
            data1.updateHost();
            double end = MPI_Wtime();
            if (msl::isRootProcess()) {
                if (file) {
                    std::ofstream outputFile;
                    outputFile.open(file, std::ios_base::app);
                    outputFile << "" << (end-start) << ";";
                    outputFile.close();
                }
            }
            start = MPI_Wtime();
            while (iterations_used < iterations) {
                if (iterations_used % 2 == 0) {
                    data1.mapStencilMM(data2, GoL, 0);
                } else {
                    data2.mapStencilMM(data1, GoL, 0);
                }
                iterations_used++;
            }
            data1.updateHost();
            data2.updateHost();

            end = MPI_Wtime();
            if (msl::isRootProcess()) {
                if (file) {
                    std::ofstream outputFile;
                    outputFile.open(file, std::ios_base::app);
                    outputFile << "" << (end-start) << ";" << std::to_string(iterations_used) + ";" ;
                    outputFile.close();
                }
            }
            return 0;
        }

    } // namespace msl
int main(int argc, char **argv) {
    msl::initSkeletons(argc, argv);
    int n = 100;
    int m = 100;
    int stencil_radius = 1;
    int nGPUs = 1;
    int nRuns = 1;
    int iterations = MAX_ITER;
    int tile_width = msl::DEFAULT_TILE_WIDTH;
    msl::Muesli::cpu_fraction = 0.25;
    //bool warmup = false;
    msl::setDebug(true);

    char *file = nullptr;

    if (argc >= 6) {
        n = atoi(argv[1]);
        m = atoi(argv[2]);
        nGPUs = atoi(argv[3]);
        nRuns = atoi(argv[4]);
        msl::Muesli::cpu_fraction = atof(argv[5]);
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

    int iterations_used=0;
    for (int r = 0; r < msl::Muesli::num_runs; ++r) {
        msl::gameoflife::run(n, m, stencil_radius, tile_width, iterations, iterations_used, file);
    }

    if (file) {
        std::ofstream outputFile;
        outputFile.open(file, std::ios_base::app);
        outputFile << "" + std::to_string(n) + ";" + std::to_string(nGPUs) + ";" + std::to_string(tile_width) +";" +
        std::to_string(iterations) << ";" << "\n";
        outputFile.close();
    } else {
    }
    msl::terminateSkeletons();
    return 0;
}
