/*
 * gassimulation_test.cpp
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

#include "muesli.h"
#include "dc.h"
#include "da.h"
#include "functors.h"
#include "array.h"
#include "vec3.h"

#ifdef __CUDACC__
#define MSL_MANAGED __managed__
#define MSL_CONSTANT __constant__
#define POW(a, b)      powf(a, b)
#define EXP(a)      exp(a)
#else
#define MSL_MANAGED
#define MSL_CONSTANT
#define POW(a, b)      std::pow(a, b)
#define EXP(a)      std::exp(a)
#endif


int CHECK = 0;
int OUTPUT = 1;
namespace msl::minnested {

    MSL_USERFUNC int innerfunction(int x, int z) {
        return x+z;
    }


    struct Outerfunction : public Functor2<int, int, int> {
        MSL_USERFUNC int operator()(int x, int z) const {

            return innerfunction(x, z);
        }
    };

    void nested_test(int dimension, int iterations, const std::string &importFile, const std::string &exportFile) {
        DA<int> exampleds(dimension, 5);
        DA<int> *exampledspointer = &exampleds;

        Outerfunction of;
        for (int i = 0; i < iterations; ++i) {
            exampledspointer->mapIndexInPlace(of);
        }
        int * gather = exampledspointer->gather();

        if (msl::isRootProcess()) {
            if (!exportFile.empty()) {
                FILE *file = fopen(exportFile.c_str(), "w"); // append file or create a file if it does not exist
                for (int x = 0; x < dimension; x++) {
                    int zelle = gather[x];
                    fprintf(file, "%d;", zelle); // write
                    fprintf(file, "\n"); // write
                }
                fprintf(file, "\n"); // write
                fclose(file);        // close file
                printf("File created. Located in the project folder.\n");
            }
        }
    }
}

void exitWithUsage() {
    std::cerr << "Usage: ./minnested [-d <x>] [-g <nGPUs>] [-n <iterations>] [-i <importFile>] [-e <exportFile>]" << std::endl;
    exit(-1);
}

int getIntArg(char* s, bool allowZero = false) {
    int i = std::atoi(s);
    if (i < 0 || (i == 0 && !allowZero)) {
        printf("allow 0");

        exitWithUsage();
    }
    return i;
}

int main(int argc, char** argv){
    int size = 0;
    int gpus = 1;
    int iterations = 1;
    std::string importFile, exportFile;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            printf("missing - ");

            exitWithUsage();
        }
        switch(argv[i++][1]) {
            case 'd':
                size = getIntArg(argv[i]);
                break;
            case 'g':
                gpus = getIntArg(argv[i]);
                break;
            case 'n':
                iterations = getIntArg(argv[i], true);
                break;
            case 'i':
                importFile = std::string(argv[i]);
                break;
            case 'e':
                exportFile = std::string(argv[i]);
                break;
            case 't':
                msl::setNumThreads(getIntArg(argv[i]));
                break;
            default:
                printf("default switch");
                exitWithUsage();
        }
    }

    msl::setNumRuns(1);
    msl::setDebug(false);
    msl::initSkeletons(argc, argv);
    msl::Muesli::cpu_fraction = 0;
    msl::Muesli::num_gpus = gpus;
    msl::minnested::nested_test(size, iterations, importFile, exportFile);
    msl::terminateSkeletons();
    return EXIT_SUCCESS;
}
