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

        MSL_USERFUNC int operator()(int x) const override { return x * y; }
    };

    struct Produkt : public Functor3<int, int, int, int> {
        MSL_USERFUNC int operator()(int i, int j, int Ai) const override { return (i * j * Ai); }
    };

    class Mult4 : public Functor4<int, int, int, int, int> {
    private:
        int y;
    public:
        explicit Mult4(int factor) :
                y(factor) {}

        MSL_USERFUNC int operator()(int i, int j, int Ai, int Bi) const override { return (i * j * Ai * Bi * y); }
    };

    class Mult5 : public Functor5<int, int, int, int, int, int> {
    private:
        int y;
    public:
        explicit Mult5(int factor) :
                y(factor) {}

        MSL_USERFUNC int operator()(int i, int j, int l, int Ai, int Bi) const override {
            return (i * j * l * Ai * Bi * y);
        }
    };

    class Sum : public Functor2<int, int, int> {
    public:
        MSL_USERFUNC int operator()(int x, int y) const override { return x + y; }
    };

    class Sum3 : public Functor3<int, int, int, int> {
    public:
        MSL_USERFUNC int operator()(int i, int j, int x) const override { return i + j + x; }
    };


    class Sum4 : public Functor4<int, int, int, int, int> {
    public:
        MSL_USERFUNC int operator()(int i, int j, int x, int y) const override { return i + j + x + y; }
    };

    class Sum5 : public Functor5<int, int, int, int, int, int> {
    public:
        MSL_USERFUNC int operator()(int i, int j, int x, int y, int l) const override { return i + j + x + y + l; }
    };

    void da_test(int dim, const std::string &nextfile, int reps, const char *skeletons) {

        // ************* Init *********************** //
        double runtimes[11] = {0.0};
        double t = 0.0;
        DA<int> a(dim);
        
        int *muesliResult = new int[dim];

        if (check_str_inside(skeletons, "Fill,")) {
            t = MPI_Wtime();
            a.fill(2);
            a.gather(muesliResult);
            runtimes[0] += MPI_Wtime() - t;

            if (CHECK && msl::isRootProcess()) {
                check_array_value_equal("Fill", dim, muesliResult, 2);
            }
        }

        t = MPI_Wtime();
        DA<int> b(dim, 3);
        runtimes[1] += MPI_Wtime() - t;
        if (check_str_inside(skeletons, "Initfill,")) {

            if (CHECK) {
                b.gather(muesliResult);
                if (msl::isRootProcess()) {
                    check_array_value_equal("Initfill", dim, muesliResult, 3);
                }
            }
        }

        Produkt pr;
        Mult4 mul4(3);
        Mult5 mul5(3);
        Mult mult(3);
        Sum sum;
        Sum3 sum3;
        Sum4 sum4;
        Sum5 sum5;
        int *manResults = new int[dim];
        DA<int> map_dest(dim, 3);
        if (check_str_inside(skeletons, "map,")) {
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                b.map(mult, map_dest);
            }
            b.gather(muesliResult);
            if (CHECK && msl::isRootProcess()) {
                check_array_value_equal("Map", dim, muesliResult, 9);
            }
            runtimes[2] += MPI_Wtime() - t;
        }

        if (check_str_inside(skeletons, "mapInPlace,")) {

            b.fill(2);
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                b.mapInPlace(mult);
            }
            b.gather(muesliResult);
            runtimes[3] += MPI_Wtime() - t;
            if (CHECK && msl::isRootProcess()) {
                for (int i = 0; i < reps; i++) {
                    for (int j = 0; j < dim; j++) {
                        manResults[j] = 3 * 2;
                    }
                }
                check_array_array_equal("mapInPlace", dim, muesliResult, manResults);


            }
        }

        DA<int> mapIndex(dim, 6);
        if (check_str_inside(skeletons, "mapIndex,")) {
            b.fill(6);
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                a.mapIndex(sum, b);
            }
            a.gather(muesliResult);
            runtimes[4] += MPI_Wtime() - t;
            if (CHECK) {
                for (int j = 0; j < dim; j++)
                    manResults[j] = j + 6;

                if (msl::isRootProcess()) {
                    check_array_array_equal("mapIndex", dim, muesliResult, manResults);
                }
            }
        }

        if (check_str_inside(skeletons, "mapIndexInPlace,")) {

            b.fill(3);
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                b.mapIndexInPlace(sum);
            }
            b.gather(muesliResult);
            runtimes[5] += MPI_Wtime() - t;

            if (CHECK) {
                for (int i = 0; i < reps; i++) {
                    for (int j = 0; j < dim; j++) {
                        manResults[j] = 3 + j;
                    }
                }
                if (msl::isRootProcess()) {
                    check_array_array_equal("MapIndexInPlace", dim, muesliResult, manResults);
                }
            }
        }
        // ************* Fold *********************** //
        if (check_str_inside(skeletons, "fold,")) {

            b.fill(3);
            int foldresult = 0;
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                foldresult = b.fold(sum, true);
            }
            runtimes[10] += MPI_Wtime() - t;

            if (CHECK) {
                if (msl::isRootProcess()) {
                    check_value_value_equal("Fold", foldresult, 3 * dim);
                }
            }
        }
        // ************* Zip *********************** //
        DA<int> c(dim, 3);
        c.fill(20);
        if (check_str_inside(skeletons, "zip,")) {
            b.fill(10);
            t = MPI_Wtime();

            for (int i = 0; i < reps; i++) {
                a.zip(b, c, sum);
            }
            a.gather(muesliResult);
            runtimes[6] += MPI_Wtime() - t;
            if (CHECK && msl::isRootProcess()) {
                check_array_value_equal("Zip", dim, muesliResult, 30);
            }

        }

        if (check_str_inside(skeletons, "zipInPlace,")) {

            b.fill(20);
            c.fill(10);
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                b.zipInPlace(c, sum);
            }
            b.gather(muesliResult);
            runtimes[7] += MPI_Wtime() - t;

            if (CHECK) {
                if (msl::isRootProcess()) {
                    check_array_value_equal("zipInPlace", dim, muesliResult, 30);
                }
            }
        }
        if (check_str_inside(skeletons, "zipIndex,")) {

            a.fill(7);
            c.fill(5);
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                b.zipIndex(c, a, sum3);
            }
            b.gather(muesliResult);
            runtimes[8] += MPI_Wtime() - t;

            if (CHECK) {
                // independent from reps.
                for (int j = 0; j < dim; j++) {
                    manResults[j] = j + 7 + 5;
                }

                if (msl::isRootProcess()) {
                    check_array_array_equal("zipIndex", dim, muesliResult, manResults);

                }
            }
        }
        if (check_str_inside(skeletons, "zipIndexInPlace,")) {

            b.fill(3);
            c.fill(2);
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                b.zipIndexInPlace(c, pr);
            }
            b.gather(muesliResult);
            runtimes[9] += MPI_Wtime() - t;

            if (CHECK) {
                for (int i = 0; i < reps; i++) {
                    for (int j = 0; j < dim; j++) {
                        manResults[j] = j * 3 * 2;
                    }
                }
                
                if (msl::isRootProcess()) {
                    check_array_array_equal("ZipIndexInPlace", dim, muesliResult, manResults);
                }
            }
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
