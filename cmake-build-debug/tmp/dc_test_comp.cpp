/*
 * dc_test.cpp
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
#include "test_hepers/basic.cpp"
#include "muesli.h"
#include "dc.h"

bool CHECK = false;
int OUTPUT = true;
namespace msl::test {

    class Mult3 : public Functor<int, int> {
    private:
        int y;
    public:
        explicit Mult3(int factor) :
                y(factor) {}

        MSL_USERFUNC int operator()(int x) const override {
            double result = 0.0;
            for (int a = 0; a < 10; ++a) {
                for (int b = 0; b < (50); ++b) {
                    for (int c = 0; c < 10; ++c) {
                        result += ( y * a * b * c);
                        result = result / 50;
                    }
                }
            }
            return result;
        }
    };

    class LiMult3 : public Functor<double, double> {
    private:
        int y;
    public:
        explicit LiMult3(int factor) :
                y(factor) {}

        MSL_USERFUNC double operator()(double x) const override {
            double result = 0.0;
            for (int a = 0; a < 10; ++a) {
                for (int b = 0; b < (50); ++b) {
                    for (int c = 0; c < 10; ++c) {
                        result += ( y * a * b * c);
                        result = result / 50;
                    }
                }
            }
            return result;
        }
    };

    class Mult5 : public Functor5<int, int, int, int, int, int> {
    private:
        int y;
    public:
        explicit Mult5(int factor) :
                y(factor) {}

        MSL_USERFUNC int operator()(int i, int j, int l, int Ai, int Bi) const override {
            double result = 0.0;

            for (int a = 0; a < i; ++a) {
                for (int b = 0; b < (50); ++b) {
                    for (int c = 0; c < 10; ++c) {
                        result += (i * j * l * Ai * Bi * y * a * b * c);
                        result = result / 50;
                    }
                }
            }

            return result;
        }
    };

    class LiMult5 : public Functor5<int, int, double, double, double, double> {
    private:
        int y;
    public:
        explicit LiMult5(int factor) :
                y(factor) {}

        MSL_USERFUNC double operator()(int i, int j, double l, double Ai, double Bi) const override {
            double result = 0.0;

            for (int a = 0; a < i; ++a) {
                for (int b = 0; b < (50); ++b) {
                    for (int c = 0; c < 10; ++c) {
                        result += (i * j * l + Ai * Bi + y * a + b * c);
                        result = result / 50;
                    }
                }
            }

            return result;
        }
    };

    class Sum : public Functor2<double, double, double> {
    public:
        MSL_USERFUNC double operator()(double x, double y) const override {
            double result = 0.0;
            for (int a = 0; a < 10; ++a) {
                for (int b = 0; b < (50); ++b) {
                    for (int c = 0; c < 10; ++c) {
                        result += ( y + a + b + c);
                        result = result / 50;
                    }
                }
            }
            return result;
        }
    };

    class Sum4 : public Functor4<int, int, int, double, double> {
    public:
        MSL_USERFUNC double operator()(int i, int j, int x, double y) const override {
            double result = 0.0;

            for (int a = 0; a < 10; ++a) {
                for (int b = 0; b < (50); ++b) {
                    for (int c = 0; c < 10; ++c) {
                        result += (i + j + y * a + b * c);
                        result = result / 50;
                    }
                }
            }

            return result;
        }
    };

    class Sum5 : public Functor5<int, int, int, double, double, double> {
    public:
        MSL_USERFUNC double operator()(int i, int j, int x, double y, double l) const override {
            double result = 0.0;

            for (int a = 0; a < 10; ++a) {
                for (int b = 0; b < (50); ++b) {
                    for (int c = 0; c < 10; ++c) {
                        result += (i * j * l * y * a * b * c);
                        result = result / 50;
                    }
                }
            }

            return result;
        }
    };


    void dc_test(int dim, const std::string &nextfile, int reps, const char *skeletons) {

        // ************* Init *********************** //
        int elements = dim * dim * dim;
        double runtimes[9] = {0.0};
        double t;
        Mult5 mul5(3);
        Mult3 mul3(3);
        // Done.
        LiMult3 limul3(3);
        // Check
        LiMult5 limul5(3);
        Sum sum;
        Sum4 sum4;
        Sum5 sum5;
        DC<double> a(dim, dim, dim, 2.0);
        DC<double> b(dim, dim, dim, 3.0);
        DC<double> map_dest(dim, dim, dim, 5.0);
        double *pResults;
        auto *manpResults = new double[dim * dim * dim];

        if (check_str_inside(skeletons, "map,")) {
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                map_dest.map(mul3, b);
            }
            pResults = map_dest.gather();
            runtimes[0] += MPI_Wtime() - t;
            if (CHECK && msl::isRootProcess()) {
                manpResults = map_dest.gather();
                check_array_array_equal("Map", elements, pResults, manpResults);
            }
        }
        b.fill(2.0);
        if (check_str_inside(skeletons, "mapInPlace,")) {
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) { b.mapInPlace(limul3); }
            pResults = b.gather();
            runtimes[1] += MPI_Wtime() - t;
            if (CHECK && msl::isRootProcess()) {
                for (int j = 0; j < elements; j++) {
                    manpResults[j] = 2;
                }
                for (int i = 0; i < reps; i++) {
                    for (int j = 0; j < elements; j++) {
                        manpResults[j] = manpResults[j] * 3 / 50;
                    }
                }
                check_array_array_equal("mapInPlace", elements, pResults, manpResults);
            }
        }

        b.fill(2.0);
        a.fill(6.0);

        if (check_str_inside(skeletons, "mapIndex,")) {
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                b.mapIndex(sum4, a);
            }
            pResults = b.gather();
            runtimes[2] += MPI_Wtime() - t;
            if (CHECK && msl::isRootProcess()) {
                for (int j = 0; j < elements; j++) {
                    int depth = int(j / (dim * dim));
                    manpResults[j] = depth + int(j - (depth * dim * dim)) / dim + (j % dim) + 6;
                }
                check_array_array_equal("mapIndex", elements, pResults, manpResults);
            }
        }

        b.fill(3.0);
        if (check_str_inside(skeletons, "mapIndexInPlace,")) {
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                b.mapIndexInPlace(sum4);
            }
            pResults = b.gather();
            runtimes[3] += MPI_Wtime() - t;
            if (CHECK && msl::isRootProcess()) {
                for (int j = 0; j < elements; j++) {
                    manpResults[j] = 3;
                }
                for (int i = 0; i < reps; i++) {
                    for (int j = 0; j < elements; j++) {
                        int depth = int(j / (dim * dim));
                        manpResults[j] =
                                depth + int(j - (depth * dim * dim)) / dim + (j % dim) + manpResults[j];
                    }
                }
                check_array_array_equal("MapIndexInPlace", elements, pResults, manpResults);
            }
        }
        // ************* Fold *********************** //
        if (check_str_inside(skeletons, "fold,")) {
            b.fill(3.0);
            double result = 0;
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                result = b.fold(sum, true);
            }
            runtimes[8] += MPI_Wtime() - t;

            if (CHECK && msl::isRootProcess()) {
                check_value_value_equal("Fold", result, elements * 3.0);
            }
        }
        // ************* Zip *********************** //
        DC<double> c(dim, dim, dim, 3.3);
        DS<double> datastructure(dim);
        DC<double> d(dim, dim, dim, 2.0);
        if (check_str_inside(skeletons, "zip,")) {
            b.fill(10.0);
            c.fill(20.0);
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                d.zip(b, c, sum);
            }
            pResults = d.gather();
            runtimes[4] += MPI_Wtime() - t;

            if (CHECK && msl::isRootProcess()) {
                check_array_value_equal("Zip", elements, pResults, 30.0);
            }
        }
        if (check_str_inside(skeletons, "zipInPlace,")) {
            b.fill(10.0);
            c.fill(10.0);
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                b.zipInPlace(c, sum);
            }
            pResults = b.gather();
            runtimes[5] += MPI_Wtime() - t;
            for (int j = 0; j < elements; j++) {
                manpResults[j] = 10;
            }
            for (int i = 0; i < reps; i++) {
                for (int j = 0; j < elements; j++) {
                    manpResults[j] = 10 + manpResults[j];
                }
            }
            if (CHECK && msl::isRootProcess()) {
                check_array_array_equal("ZipInPlace", elements, pResults, manpResults);
            }
        }
        if (check_str_inside(skeletons, "zipIndex,")) {
            b.fill(7.0);
            c.fill(5.0);
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                d.zipIndex(b, c, sum5);
            }
            pResults = d.gather();
            runtimes[6] += MPI_Wtime() - t;

            if (CHECK && msl::isRootProcess()) {
                for (int j = 0; j < elements; j++) {
                    int depth = int(j / (dim * dim));
                    manpResults[j] = depth + int(j - (depth * dim * dim)) / dim + (j % dim) + 7 + 5;
                }
                check_array_array_equal("zipIndex", elements, pResults, manpResults);
            }
        }


        b.fill(3.0);
        c.fill(2.0);
        if (check_str_inside(skeletons, "zipIndexInPlace,")) {
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                b.zipIndexInPlace(c, sum5);
            }
            pResults = b.gather();
            runtimes[7] += MPI_Wtime() - t;

            if (CHECK && msl::isRootProcess()) {
                for (int j = 0; j < elements; j++) {
                    int depth = int(j / (dim * dim));
                    manpResults[j] = depth + int(j - (depth * dim * dim)) / dim + (j % dim) + 3 + 2;
                }
                for (int i = 0; i < reps - 1; i++) {
                    for (int j = 0; j < elements; j++) {
                        int depth = int(j / (dim * dim));
                        manpResults[j] = depth + int(j - (depth * dim * dim)) / dim + (j % dim) + manpResults[j] + 2;
                    }
                }
                check_array_array_equal("ZipIndexInPlace", elements, pResults, manpResults);
            }
        }

        if (msl::isRootProcess()) {
            int arraysize = sizeof(runtimes) / (sizeof(double));
            print_and_doc_runtimes(OUTPUT, nextfile, runtimes, arraysize, dim, reps);
        }
    }
}

int main(int argc, char **argv) {
    //printf("Starting Main...\n");

    int dim = 4, nGPUs = 1, reps = 1;
    const char *skeletons;
    arg_helper(argc, argv, dim, nGPUs, reps, CHECK, const_cast<char *&>(skeletons));

    std::string nextfile = "/home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/Data/dc_Processes-" +
            std::to_string(msl::Muesli::num_total_procs) + "_GPUs-" + std::to_string(msl::Muesli::num_gpus) ;
    if (msl::isRootProcess() && OUTPUT) {
        printf("%s; %d; %d; %d; %d; %.2f;", "complex", dim, msl::Muesli::num_total_procs,
               msl::Muesli::num_local_procs, msl::Muesli::num_threads, msl::Muesli::cpu_fraction);
    }
    msl::test::dc_test(dim, nextfile, reps, skeletons);
    msl::terminateSkeletons();
    return EXIT_SUCCESS;
}
