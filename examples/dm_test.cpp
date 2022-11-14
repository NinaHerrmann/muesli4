/*
 * dm_test.cpp
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
#include <cstring>
#include <test_hepers/basic.cpp>

#include "muesli.h"
#include "dm.h"

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

    class Sum : public Functor2<int, int, int> {
    public:
        MSL_USERFUNC int operator()(int x, int y) const override { return x + y; }
    };

    class Sum3 : public Functor3<int, int, int, int> {
    public:
        MSL_USERFUNC int operator()(int i, int j, int x) const override { return i + j + x; }
    };

    class Index : public Functor3<int, int, int, int> {
    private:
        int y;
    public:
        explicit Index(int cols) :
                y(cols) {}

        MSL_USERFUNC int operator()(int i, int j, int x) const override { return (i * y) + j; }
    };


    class Sum4 : public Functor4<int, int, int, int, int> {
    public:
        MSL_USERFUNC int operator()(int i, int j, int x, int y) const override { return i + j + x + y; }
    };

    void dm_test(int dim, const std::string& nextfile, int reps, const char *skeletons) {
        // ************* Init *********************** //
        double runtimes[11] = {0.0};
        double t;

        DM<int> a(dim, dim, 2);
        int elements = dim * dim;

        if (check_str_inside(skeletons, "Fill,")) {

            t = MPI_Wtime();
            a.fill(3);
            int *fillResult = a.gather();
            runtimes[0] += MPI_Wtime() - t;

            if (CHECK && msl::isRootProcess()) {
                check_array_value_equal("Fill", elements, fillResult, 3);
            }
        }
        DM<int> b(dim, dim, 5);

        if (check_str_inside(skeletons, "Initfill,")) {

            t = MPI_Wtime();
            runtimes[1] += MPI_Wtime() - t;
            int *constResult = b.gather();

            if (CHECK && msl::isRootProcess()) {
                check_array_value_equal("Initfill", elements, constResult, 5);
            }
        }
        Produkt pr;
        Mult4 mul4(3);
        Mult mult(3);
        Sum sum;
        Sum3 sum3;
        Sum4 sum4;
        Index index(dim);
        /*DM<int> rotate(dim, dim, 3);
        rotate.mapIndexInPlace(index);
        //if (CHECK) { rotate.show(); }
           rotate.rotateRows(1);*/
        /*
           if (CHECK) { printf("Rotate 1 \n"); rotate.show(); }
           rotate.rotateRows(-1);
           if (CHECK) { printf("Rotate -1 \n");rotate.show(); }
           rotate.rotateRows(2);
           if (CHECK) { printf("Rotate 2 \n");rotate.show(); }
           rotate.rotateRows(-2);
           if (CHECK) { printf("Rotate -2 \n"); rotate.show(); }
           //rotate.rotateRows(dim+1);
           rotate.rotateCols(-2);
           rotate.rotateCols(2);
           rotate.rotateCols(5);
           rotate.rotateCols(-5);*/
        int *muesliResults;
        int *manResults = new int[elements];

        if (check_str_inside(skeletons, "map,")) {

            a.fill(3);
            b.fill(3);
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                a.map(mult, b);
            }
            muesliResults = a.gather();
            runtimes[2] += MPI_Wtime() - t;
            if (CHECK && msl::isRootProcess()) {
                check_array_value_equal("Map", elements, muesliResults, 9);
            }
        }
        if (check_str_inside(skeletons, "mapInPlace,")) {

            a.fill(2);
            t = MPI_Wtime();

            for (int i = 0; i < reps; i++) {
                a.mapInPlace(mult);
            }
            muesliResults = a.gather();
            runtimes[3] += MPI_Wtime() - t;
            for (int j = 0; j < elements; j++) {
                manResults[j] = 2;
            }
            for (int i = 0; i < reps; i++) {
                for (int j = 0; j < elements; j++) {
                    manResults[j] = manResults[j] * 3;
                }
            }

            if (CHECK && msl::isRootProcess()) {
                check_array_array_equal("mapInPlace", elements, muesliResults, manResults);
            }
        }
        if (check_str_inside(skeletons, "mapIndex,")) {

            a.fill(6);
            b.fill(6);
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                b.mapIndex(sum3, a);
            }
            muesliResults = b.gather();
            runtimes[4] += MPI_Wtime() - t;

            for (int j = 0; j < elements; j++) {
                manResults[j] = 6;
            }
            for (int j = 0; j < elements; j++) {
                manResults[j] = int(j / dim) + (j % dim) + manResults[j];
            }

            if (CHECK && msl::isRootProcess()) {
                check_array_array_equal("mapIndex", elements, muesliResults, manResults);
            }
        }
        if (check_str_inside(skeletons, "mapIndexInPlace,")) {
            a.fill(2);
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                a.mapIndexInPlace(pr);
            }
            muesliResults = a.gather();
            runtimes[5] += MPI_Wtime() - t;

            for (int i = 0; i < reps; i++) {
                for (int j = 0; j < elements; j++) {
                    manResults[j] = 2 * int(j / dim) * (j % dim);
                }
            }
            if (CHECK && msl::isRootProcess()) {
                check_array_array_equal("MapIndexInPlace", elements, muesliResults, manResults);
            }
        }
        // ************* Fold *********************** //
        if (check_str_inside(skeletons, "fold,")) {
            a.fill(2);
            int foldresult;
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                foldresult = a.fold(sum, true);
            }
            runtimes[10] += MPI_Wtime() - t;

            if (CHECK && msl::isRootProcess()) {
                check_value_value_equal("Fold", foldresult, 2 * elements);
            }
        }

        // ************* Zip *********************** //

        DM<int> c(dim, dim, 2);
        a.fill(10);
        b.fill(20);
        if (check_str_inside(skeletons, "zip,")) {

            for (int j = 0; j < elements; j++) {
                manResults[j] = 20;
            }
            for (int j = 0; j < elements; j++) {
                manResults[j] = 10 + manResults[j];
            }

            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                c.zip(b, a, sum);
            }
            muesliResults = c.gather();
            runtimes[6] += MPI_Wtime() - t;

            if (CHECK && msl::isRootProcess()) {
                check_array_value_equal("Zip", elements, muesliResults, 30);
            }
        }

        if (check_str_inside(skeletons, "zipIndex,")) {
            a.fill( 7);
            b.fill( 2);
            c.fill( 8);
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                b.zipIndex(c, a, sum4);
            }
            muesliResults = b.gather();
            runtimes[7] += MPI_Wtime() - t;

            int *manResults = new int[elements];
            // Independent from reps?
            for (int j = 0; j < elements; j++) {
                manResults[j] = int(j / dim) + (j % dim) + 7 + 8;
            }

            if (CHECK && msl::isRootProcess()) {
                check_array_array_equal("zipIndex", elements, muesliResults, manResults);
            }
        }
        if (check_str_inside(skeletons, "zipInPlace,")) {

            a.fill( 10);
            b.fill(20);
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                b.zipInPlace(a, sum);
            }
            muesliResults = b.gather();
            runtimes[8] = MPI_Wtime() - t;

            for (int j = 0; j < elements; j++) {
                manResults[j] = 20;
            }
            for (int i = 0; i < reps; i++) {
                for (int j = 0; j < elements; j++) {
                    manResults[j] = 10 + manResults[j];
                }
            }
            if (CHECK && msl::isRootProcess()) {
                check_array_array_equal("zipInPlace", elements, muesliResults, manResults);
            }
        }
        if (check_str_inside(skeletons, "zipIndexInPlace,")) {

            a.fill(4);
            b.fill( 2);
            for (int j = 0; j < elements; j++) {
                manResults[j] = 4;
            }
            for (int i = 0; i < reps; i++) {
                for (int j = 0; j < elements; j++) {
                    manResults[j] = manResults[j] * int(j / dim) * int(j % dim) * 3 * 2;
                }
            }
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                a.zipIndexInPlace(b, mul4);
            }
            muesliResults = a.gather();
            runtimes[9] += MPI_Wtime() - t;

            if (CHECK && msl::isRootProcess()) {
                check_array_array_equal("ZipIndexInPlace", elements, muesliResults, manResults);
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


    std::string nextfile = "Data/dm_" + std::to_string(msl::Muesli::num_total_procs) + "_" +
                           std::to_string(msl::Muesli::num_gpus) + "_" + std::to_string(dim) + "_" +
                           std::to_string(msl::Muesli::cpu_fraction);
    if (msl::isRootProcess()) {
        printf("Starting Program %s with %d nodes %d cpus and %d gpus\n", msl::Muesli::program_name,
               msl::Muesli::num_total_procs,
               msl::Muesli::num_local_procs, msl::Muesli::num_gpus);
    }
    msl::test::dm_test(dim, nextfile, reps, skeletons);
    msl::terminateSkeletons();
    return EXIT_SUCCESS;
}
