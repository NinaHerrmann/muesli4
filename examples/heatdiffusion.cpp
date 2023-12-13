#include <muesli.h>
#include "dm.h"
#include "dc.h"

#define ENABLE_1D_EXAMPLE 1
#define ENABLE_2D_EXAMPLE 1
#define ENABLE_3D_EXAMPLE 1

namespace msl::heatdiffusion {
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

    void heat_2D(int size, int iterations, int output, int argc, char *argv[]) {
        double timeStart = MPI_Wtime();
        msl::initSkeletons(argc, argv);

        DM<float> dm(size, size, 0, true);
        DM<float> dm_copy(size, size, 0, true);

        double init = MPI_Wtime();
        double timeinit = init-timeStart;
#pragma omp parallel for default(none) shared(dm,size)
        for (size_t i = 0; i < size; ++i) {
            dm.set2D(i, 0, 2);
            dm.set2D(i, size-1, 2);
        }

        for (size_t i = 0; i < size; ++i) {
            dm.set2D(0, i, 0);
            dm.set2D(size-1, i, 5);
        }
        dm.cpuMemoryInSync = true;

        dm.updateDevice(1);

        double fill = MPI_Wtime();
        double timefill = fill-init;
        DM<float> *dmp1 = &dm;
        DM<float> *dmp2 = &dm_copy;
        for (size_t i = 0; i < iterations; ++i) {
            dmp1->mapStencil<heat2D>(dm_copy, 1, 0);
            dmp2->mapStencil<heat2D>(dm, 1, 0);
        }
        double calc = MPI_Wtime();
        double timecalc = calc-init;

        if (msl::isRootProcess() && output == 1) {
            dmp2->updateHost();
            float *gather = dmp2->gather();
            std::string fileName = "d2-s" + std::to_string(size) + "-i" + std::to_string(iterations) + ".out";
            std::ofstream outputFile(fileName, std::ios::app | std::ios::binary); // append file or create a file if it does not exist
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
        double endTime = MPI_Wtime();
        double totaltime = endTime - timeStart;

        if (msl::isRootProcess()) {
            std::string fileName = "init-runtime-d2-s" + std::to_string(size) + "-i" + std::to_string(iterations) + ".out";
            std::ofstream outputFile(fileName, std::ios::app); // append file or create a file if it does not exist
            outputFile << "2" << ";" <<  size << ";" << iterations << ";" << timeinit << ";"<< timefill << ";" << timecalc << ";"
                << totaltime << "\n"; // write
            outputFile.close();
        }
        exit(0);
    }

    void heat_3D(int size, int iterations, int output, int argc, char *argv[]) {
       double timeStart = MPI_Wtime();
       msl::initSkeletons(argc, argv);
       DC<float> dc(size, size, size, 0, false);
       DC<float> dc_copy(size, size, size, 0, false);

       double init = MPI_Wtime();
       double timeinit = init-timeStart;
#pragma omp parallel for default(none) shared(dc,size)
       for (size_t i = 0; i < size; ++i) {
           for (size_t j = 0; j < size; ++j) {
               dc.set(0, i, j, 1);
               dc.set(size-1, i, j, 5);
               dc.set(i, 0, j, 2);
               dc.set(i, size-1, j, 2);
           }
       }
       dc.updateDevice(1);

       double fill = MPI_Wtime();
        double timefill = fill-init;

        DC<float> *dcp1 = &dc;
        DC<float> *dcp2 = &dc_copy;

        for (size_t i = 0; i < iterations; ++i) {
            dcp1->mapStencil<heat3D>(*dcp2, 1, 0);
            dcp2->mapStencil<heat3D>(*dcp1, 1, 0);
        }
        double calc = MPI_Wtime();
        double timecalc = calc-init;


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
        double endTime = MPI_Wtime();
        double totaltime = endTime - timeStart;

        if (msl::isRootProcess()) {
            std::string fileName = "runtime-d3-s" + std::to_string(size) + "-i" + std::to_string(iterations) + ".out";
            std::ofstream outputFile(fileName, std::ios::app); // append file or create a file if it does not exist
            outputFile << "3" << ";" <<  size << ";" << iterations << ";" << timeinit << ";"<< timefill << ";" << timecalc
                << ";" << totaltime << "\n"; // write
            outputFile.close();
        }

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
    msl::setDebug(false);
    msl::Muesli::cpu_fraction = 0;
    msl::Muesli::num_gpus = gpus;
    for (int i = 0; i < runs; i++) {
#if ENABLE_2D_EXAMPLE
        if (dim == 2) {
            msl::heatdiffusion::heat_2D(size, iterations, output, argc, argv);
        }
#endif
#if ENABLE_3D_EXAMPLE
        if (dim == 3) {
            msl::heatdiffusion::heat_3D(size, iterations, output, argc, argv);
        }
#endif
    }
    return 0;
}
