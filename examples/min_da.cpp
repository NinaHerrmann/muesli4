/*
 * DA_test.cpp
 *
 *      Author: Nina Hermann,
 *  	        Herbert Kuchen <kuchen@uni-muenster.de>
 * 
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2020  Herbert Kuchen <kuchen@uni-muenster.de>,
 *                 Nina Hermann
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
#include <test_hepers/basic.cpp>

#include "muesli.h"
#include "da.h"

bool CHECK = false;
bool OUTPUT = false;
namespace msl::test {

    class Mult : public Functor<int, int> {
    private:
        int y;
    public:
        explicit Mult(int factor) :
                y(factor) {}

        MSL_USERFUNC int operator()(int x) const override {
            y * 3;
            return x * y; }
    };

    class Add : public Functor<int, int> {
    private:
        int y;
    public:
        explicit Add(int factor) :
                y(factor) {}

        MSL_USERFUNC int operator()(int x) const override { return x + y; }
    };
    class SubOne : public Functor<int, int> {
        MSL_USERFUNC int operator()(int x) const override {
            int j = 0;
            for (int i = 0; i < x; i++) {
                j += 1;
            }
            return x - j; }
    };
    void da_test(int dim, const std::string &nextfile, int reps, const char *skeletons) {

        // ************* Init *********************** //
        double runtimes[11] = {0.0};
        double t = 0.0;
        DA<int> a(10, 3);
        DA<int> b(10, 3);

        int *muesliResult = new int[dim];
        Mult mult(3);

        int *manResults = new int[dim];
        if (check_str_inside(skeletons, "map,")) {
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                b.map(mult, a);
            }
            b.gather(muesliResult);
            if (CHECK && msl::isRootProcess()) {
                check_array_value_equal("Map", dim, muesliResult, 9);
            }
            runtimes[2] += MPI_Wtime() - t;
        }

        if (msl::isRootProcess()) {
            int arraysize = sizeof(runtimes) / (sizeof(double));
            print_and_doc_runtimes(OUTPUT, nextfile, runtimes, arraysize);
        }
    }
}

int main(int argc, char **argv) {
    //printf("Starting Main...\n");
    int dim = 4, nGPUs = 1, reps = 1;
    const char * skeletons;
    arg_helper(argc, argv, dim, nGPUs, reps, CHECK, const_cast<char *&>(skeletons));

    std::string nextfile = "Data/da_" + std::to_string(msl::Muesli::num_total_procs) + "_" +
                           std::to_string(msl::Muesli::num_gpus) + "_" + std::to_string(dim) + "_" +
                           std::to_string(msl::Muesli::cpu_fraction);    
    if (msl::isRootProcess()) {
        printf("Starting Program %s with %d nodes %d cpus and %d gpus\n", msl::Muesli::program_name,
               msl::Muesli::num_total_procs,
               msl::Muesli::num_local_procs, msl::Muesli::num_gpus);
    }
    msl::test::da_test(dim, nextfile, reps, skeletons);
    msl::terminateSkeletons();
    return EXIT_SUCCESS;
}
