#include <muesli.h>
#include "dm.h"
#include "dc.h"

#define ENABLE_2D_EXAMPLE 1
#define ENABLE_3D_EXAMPLE 1

namespace msl::heatdiffusion {
    class Initialize_2D : public Functor3<int, int, float, float> {
    public:
        Initialize_2D(int rows, int cols) : Functor3() {
            this->rows = rows;
            this->cols = cols;
        }

        MSL_USERFUNC float operator()(int x, int y, float c) const override {
            float value = 0;

            if ((y == 0 || y == cols - 1) && x != 0) {
                value = 2;
            }
            if (x == rows - 1) {
                value = 5;
            }
            return value;
        }

    private:
        int rows, cols;
    };

    class Initialize_3D : public Functor4<int, int, int, float, float> {
    public:
        Initialize_3D(int cols, int depth) : Functor4() {
            this->cols = cols;
            this->depth = depth;
        }

        MSL_USERFUNC float operator()(int x, int y, int z, float c) const override {
            float value = 0;
            if (z == 0 && y != 0) {
                value = 1;
            } else if (y == 0 || y == cols - 1) {
                value = 2;
            } else if (z == depth - 1) {
                value = 5;
            }
            return value;
        }

    private:
        int cols, depth;
    };

    MSL_USERFUNC float heat3D(const PLCube<float> &plCube, int x, int y, int z) {
        float newval = 0;
        newval += plCube(x - 1, y, z);
        newval += plCube(x + 1, y, z);
        newval += plCube(x, y - 1, z);
        newval += plCube(x, y + 1, z);
        newval += plCube(x, y, z - 1);
        newval += plCube(x, y, z + 1);
        newval /= 6;
        return newval;
    }

    MSL_USERFUNC float heat2D(const NPLMatrix<float> &nplMatrix, int x, int y) {
        float newval = 0;
        newval += nplMatrix(x - 1, y);
        newval += nplMatrix(x + 1, y);
        newval += nplMatrix(x, y - 1);
        newval += nplMatrix(x, y + 1);

        newval /= 4;
        return newval;
    }


    void heat_2D(int size, int iterations, int output, int argc, char *argv[], int gpu) {
        msl::initSkeletons(argc, argv);
        startTiming();
        msl::Muesli::num_gpus = gpu;

        DM<float> dm(size, size, 0, true);
        DM<float> dm_copy(size, size, 0, true);

        double timeinit = splitTime(0);
        Initialize_2D init2d = Initialize_2D(size, size);
        dm.mapIndexInPlace(init2d);
        double timefill = splitTime(0);
        DM<float> *dmp1 = &dm;
        DM<float> *dmp2 = &dm_copy;
        for (size_t i = 0; i < iterations; ++i) {
            dmp1->mapStencil<heat2D>(dm_copy, 1, 0, false);
            dmp2->mapStencil<heat2D>(dm, 1, 0, false);
        }
        double timecalc = splitTime(0);

        if (msl::isRootProcess() && output) {
            dmp2->updateHost();
            float *gather = dmp2->gather();
            std::string fileName = "d2-s" + std::to_string(size) + "-i" + std::to_string(iterations) + ".out";
            std::ofstream outputFile(fileName, std::ios::app |
                                               std::ios::binary); // append file or create a file if it does not exist
            for (int x = 0; x < size; x++) {
                for (int y = 0; y < size; y++) {
                    int index = y + (size * x);
                    float zelle = gather[index];
                    outputFile << zelle << ";"; // write
                }
                outputFile << "\n"; // write
            }
            outputFile.close();
        }
        double totaltime = splitTime(0);

        if (msl::isRootProcess()) {
            std::string fileName = "init-runtime-d2-s" + std::to_string(size) + "-i" + std::to_string(iterations) + "-g" + std::to_string(msl::Muesli::num_gpus) + ".out";
            std::ofstream outputFile(fileName, std::ios::app); // append file or create a file if it does not exist
            outputFile << "2" << ";" <<  size << ";" << iterations << ";" << timeinit << ";"<< timefill << ";" << timecalc << ";"
                << totaltime << ";" << msl::Muesli::num_gpus << ";" << msl::Muesli::num_total_procs << "\n"; // write
            outputFile.close();
            std::cout << "2;" << size << ";" << iterations << ";" << timeinit << ";" << timefill << ";" << timecalc
                      << ";" << totaltime << "\n";
        }
        msl::terminateSkeletons();

        exit(0);
    }


    void heat_3D(int size, int iterations, int output, int argc, char *argv[], int gpu) {
        msl::initSkeletons(argc, argv);
        msl::Muesli::num_gpus = gpu;

        startTiming();

        DC<float> dc(size, size, size, 0, false);
        DC<float> dc_copy(size, size, size, 0, false);

        double timeinit = splitTime(0);

        Initialize_3D init3d = Initialize_3D(size, size);
        dc.mapIndexInPlace(init3d);
        double timefill = splitTime(0);

        DC<float> *dcp1 = &dc;
        DC<float> *dcp2 = &dc_copy;

        for (size_t i = 0; i < iterations; ++i) {
            dcp1->mapStencil<heat3D>(*dcp2, 1, 0);
            dcp2->mapStencil<heat3D>(*dcp1, 1, 0);
        }
        double timecalc = splitTime(0);

        if (msl::isRootProcess() && output == 1) {
            dcp2->updateHost();
            float *gather = dcp2->gather();
            std::string fileName = "d3-s" + std::to_string(size) + "-i" + std::to_string(iterations) + ".out";
            std::ofstream outputFile(fileName, std::ios::app); // append file or create a file if it does not exist
            for (int x = 0; x < size; x++) {
                for (int y = 0; y < size; y++) {
                    for (int z = 0; z < size; z++) {
                        int index = x + (size * y) + (size * size * z);
                        float zelle = gather[index];
                        outputFile << zelle << ";"; // write
                    }
                    outputFile << "\n"; // write
                }
            }
            outputFile.close();
        }
        double totaltime = stopTiming();

        if (msl::isRootProcess()) {
            std::string fileName = "runtime-d3-s" + std::to_string(size) + "-i" + std::to_string(iterations) + "-g" + std::to_string(msl::Muesli::num_gpus) + ".out";
            std::ofstream outputFile(fileName, std::ios::app); // append file or create a file if it does not exist
            outputFile << "3" << ";" <<  size << ";" << iterations << ";" << timeinit << ";"<< timefill << ";" << timecalc
                << ";" << totaltime << ";" << msl::Muesli::num_gpus << ";" << msl::Muesli::num_total_procs << "\n"; // write

            outputFile.close();
            std::cout << "3;" << size << ";" << iterations << ";" << timeinit << ";" << timefill << ";" << timecalc
                      << ";" << totaltime << "\n";
        }
        msl::terminateSkeletons();

        exit(0);
    }
}

void exitWithUsage() {
    std::cerr
            << "Usage: ./heat_diffusion [-d <dim> ] [-s <size> ] [-g <nGPUs>] [-n <iterations>] [-e <exportFile>]"
            << std::endl;
    exit(-1);
}

int getIntArg(char *s, bool allowZero = true) {
    int i = std::atoi(s);
    if (i < 0 || (i == 0 && !allowZero)) {
        exitWithUsage();
    }
    return i;
}

int main(int argc, char *argv[]) {
    int gpus = 1;
    int iterations = 1;
    int runs = 1;
    int output = 0;
    int size, dim;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            exitWithUsage();
        }
        switch (argv[i++][1]) {
            case 'd':
                dim = getIntArg(argv[i]);
                break;
            case 's':
                size = getIntArg(argv[i]);
                break;
            case 'g':
                gpus = getIntArg(argv[i]);
                break;
            case 'n':
                iterations = getIntArg(argv[i], true);
                break;
            case 'o':
                output = getIntArg(argv[i]);
                break;
            case 'u':
                runs = getIntArg(argv[i]);
                break;
            case 't':
                msl::setNumThreads(getIntArg(argv[i]));
                break;
            default:
                exitWithUsage();
        }
    }
    msl::Muesli::cpu_fraction = 0;
    msl::Muesli::num_gpus = gpus;
    for (int i = 0; i < runs; i++) {
#if ENABLE_2D_EXAMPLE
        if (dim == 2) {
            msl::heatdiffusion::heat_2D(size, iterations, output, argc, argv, gpus);
        }
#endif
#if ENABLE_3D_EXAMPLE
        if (dim == 3) {
            msl::heatdiffusion::heat_3D(size, iterations, output, argc, argv, gpus);
        }
#endif
    }
    return 0;
}
