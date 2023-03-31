/*
 * meanblur_test.cpp
 *
 *      Author: Justus Dieckmann
 *
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2022  Justus Dieckmann
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

#include <iostream>
#include <cmath>
#include <vector>
#include <csignal>
#include <climits>
#include "muesli.h"
#include "dc.h"
#include "functors.h"
#include "vec3.h"

namespace msl::dc_map_stencil_blur {

    const size_t stencilradius = 20;

    template <size_t radius>
    MSL_USERFUNC float update(const PLCube<float> &plCube, int x, int y, int z) {
        float res = 0;
        for (int mx = x - radius; mx <= x + radius; mx++) {
            for (int my = y - radius; my <= y + radius; my++) {
                for (int mz = z - radius; mz <= z + radius; mz++) {
                    res += plCube(mx, my, mz);
                }
            }
        }
        return res;
    }

    class Initialize : public Functor4<int, int, int, float, float> {
    public:
        MSL_USERFUNC float operator()(int x, int y, int z, float n) const override {
            return std::rand() / (float) INT_MAX;
        }
    };

    void blur_test(vec3<int> size, int iterations, size_t paramstencilradius) {
        DC<float> dc(size.x, size.y, size.z);

        Initialize initialize;
        dc.mapIndexInPlace(initialize);

        DC<float> dc2(size.x, size.y, size.z);

        // Pointers for swapping.
        DC<float> *dcp1 = &dc;
        DC<float> *dcp2 = &dc2;
        double time = MPI_Wtime();
        double onlyKernelTime = 0.0;

        for (int i = 0; i < iterations; i++) {
            dcp1->mapStencil<update<stencilradius>>(*dcp2, stencilradius, 0);
            double endTimex = MPI_Wtime();

            onlyKernelTime += endTimex - Muesli::start_time;

            std::swap(dcp1, dcp2);
        }
        double endTime = MPI_Wtime();
        double totalTime = endTime - time;

        std::cout << size.x << ";" << iterations << ";" << msl::Muesli::num_total_procs << ";" << msl::Muesli::num_threads << ";" << msl::Muesli::num_gpus << ";" << totalTime << " ; " << onlyKernelTime << std::endl;
    }
}

void exitWithUsage() {
    std::cerr << "Usage: ./dc_mapstencil_blur [-d <xdim> <ydim> <zdim>] [-g <nGPUs>] [-n <iterations>]" << std::endl;
    exit(-1);
}

int getIntArg(char* s, bool allowZero = false) {
    int i = std::atoi(s);
    if (i < 0 || (i == 0 && !allowZero)) {
        exitWithUsage();
    }
    return i;
}

int main(int argc, char** argv){
    vec3<int> size {100, 100, 100};
    int gpus = 1;
    int iterations = 1;
    size_t stencilradius;
    std::string importFile, exportFile;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            exitWithUsage();
        }
        switch(argv[i++][1]) {
            case 'd':
                if (argc < i + 3) {
                    exitWithUsage();
                }
                size.x = getIntArg(argv[i++]);
                size.y = getIntArg(argv[i++]);
                size.z = getIntArg(argv[i]);
                break;
            case 'g':
                gpus = getIntArg(argv[i]);
                break;
            case 'n':
                iterations = getIntArg(argv[i], true);
                break;
            case 'r':
                stencilradius = getIntArg(argv[i], true);
                break;
            default:
                exitWithUsage();
        }
    }

    msl::setNumRuns(1);
    msl::initSkeletons(argc, argv);
    msl::Muesli::cpu_fraction = 0;
    msl::Muesli::num_gpus = gpus;
    msl::dc_map_stencil_blur::blur_test(size, iterations, stencilradius);
    msl::terminateSkeletons();
    return EXIT_SUCCESS;
}