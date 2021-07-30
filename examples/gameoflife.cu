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
                    int rowIndex, int colIndex, PLMatrix<int> *input, int ncol, int nrow) const {
                int sum = 0;
                // top broder must be 100;

                sum += input->get(rowIndex-1, colIndex) + input->get(rowIndex-1, colIndex-1)+ input->get(rowIndex-1, colIndex+1)
                        + input->get(rowIndex+1, colIndex)+ input->get(rowIndex+1, colIndex-1)+ input->get(rowIndex+1, colIndex+1)
                        + input->get(rowIndex, colIndex-1)+ input->get(rowIndex, colIndex+1);

                //printf("%d;%d;%d\n", rowIndex, colIndex, sum);
                /*int live_status = input->get(rowIndex, colIndex);

                int future_live_status = 0;
                // If the cell is alive and has 2-3 neighbours it survives
                if (live_status == 1 && (sum == 2 || sum == 3 )) {
                    future_live_status = 1;
                }
                // If the cell is dead and has 3 neighbours it gets alive
                if (live_status == 0 && sum == 3) {
                    future_live_status = 1;
                }
*/

                return sum;
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
            GoLNeutralValueFunctor dead_nvf(0);
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
                data1.set(i, 1);//data1.set(i, rand() % 2);
            }
            data1.download();
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
            //data1.show("start");
            while (iterations_used < 1) {
                if (iterations_used % 50 == 0) {
                    data1.mapStencilMM(data2, GoL, dead_nvf);
                    data2.download();
                    data2.show("data2");
                } else {
                    if (iterations_used % 2 == 0) {
                        data1.mapStencilMM(data2, GoL, dead_nvf);
                        data2.download();
                        data2.show("data2");
                    } else {
                        data2.mapStencilMM(data1, GoL, dead_nvf);
                        data1.download();
                        data1.show("data1");
                    }
                }
                iterations_used++;
            }

            data1.download();
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
    int iterations_used=0;
    for (int r = 0; r < msl::Muesli::num_runs; ++r) {
        msl::jacobi::run(n, m, stencil_radius, tile_width, iterations, iterations_used, file);
        //msl::splitTime(r);
    }

    if (file) {
       // std::string id = "" + std::to_string(n) + ";" + std::to_string(nGPUs) + ";" + std::to_string(tile_width) +";" + std::to_string(iterations) + ";" + std::to_string(iterations_used) +
                         ";" + std::to_string(msl::Muesli::cpu_fraction * 100) + ";";
        std::ofstream outputFile;
        outputFile.open(file, std::ios_base::app);
        outputFile << "" + std::to_string(n) + ";" + std::to_string(nGPUs) + ";" + std::to_string(tile_width) +";" +
        std::to_string(iterations) << ";" << "\n";
        outputFile.close();
       // msl::printTimeToFile(id.c_str(), file);
    } else {
        //msl::stopTiming();
    }
    msl::terminateSkeletons();
    return 0;
}
