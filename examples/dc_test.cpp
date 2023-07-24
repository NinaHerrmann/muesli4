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
int OUTPUT = 1;
namespace msl::test {

    class Mult : public Functor<int, int> {
    private:
        int y;
    public:
        explicit Mult(int factor) :
                y(factor) {}

        MSL_USERFUNC int operator()(int x) const override {
            return y * x;
        }
    };

    class Mult5 : public Functor5<int, int, int, int, int, int> {
    private:
        int y;
    public:
        explicit Mult5(int factor) :
                y(factor) {}

        MSL_USERFUNC int operator()(int i, int j, int l, int Ai, int Bi) const override {
            return i * j * l * Ai * Bi * y;
        }
    };

    class Sum : public Functor2<int, int, int> {
    public:
        MSL_USERFUNC int operator()(int x, int y) const override {
            return y + x;
        }
    };


    class Sum4 : public Functor4<int, int, int, int, int> {
    public:
        MSL_USERFUNC int operator()(int i, int j, int x, int y) const override {
            return i + j + x + y;
        }
    };

    class Sum5 : public Functor5<int, int, int, int, int, int> {
    public:
        MSL_USERFUNC int operator()(int i, int j, int x, int y, int l) const override {
            return (i + j + x + y + l);
        }//(i+j+x) % 50;}
    };


    void dc_test(int dim, const std::string &nextfile, int reps, const char *skeletons) {
        if (msl::isRootProcess()) {
            printf("Starting dc_test...\n");
        }



        // ************* Init *********************** //
        int elements = dim * dim * dim;
        double runtimes[11] = {0.0};
        double t = MPI_Wtime();


        if (check_str_inside(skeletons, "Fill")) {
            t = MPI_Wtime();
            DC<int> a(dim, dim, dim);
            a.fill(2);
            // TODO does not work for all sizes.
            int *fillResult = a.gather();

            if (CHECK && msl::isRootProcess()) {
                check_array_value_equal("Fill", elements, fillResult, 2);
            }

            runtimes[0] += MPI_Wtime() - t;
            t = MPI_Wtime();
        }


        DC<int> b(dim, dim, dim, 3);

        if (check_str_inside(skeletons, "Initfill,")) {

            int *constructorResult = b.gather();

            if (CHECK && msl::isRootProcess()) {
                check_array_value_equal("Initfill", elements, constructorResult, 3);
            }
            runtimes[1] += MPI_Wtime() - t;

        }
        Mult5 mul5(3);
        Mult mult(3);
        Sum sum;
        Sum4 sum4;
        Sum5 sum5;
        int *mapResults;
        int *manmapResults = new int[dim * dim * dim];
        DC<int> map_dest(dim, dim, dim, 5);


        if (check_str_inside(skeletons, "map,")) {

            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                map_dest.map(mult, b);
            }
            mapResults = map_dest.gather();
            runtimes[2] += MPI_Wtime() - t;
            if (CHECK && msl::isRootProcess()) {
                check_array_value_equal("Map", elements, mapResults, 9);
            }
        }
        b.fill(2);
        t = MPI_Wtime();
        if (check_str_inside(skeletons, "mapInPlace,")) {
            for (int i = 0; i < reps; i++) {
                b.mapInPlace(mult);
            }
            for (int j = 0; j < elements; j++) {
                manmapResults[j] = 2;
            }
            for (int i = 0; i < reps; i++) {
                for (int j = 0; j < elements; j++) {
                    manmapResults[j] = manmapResults[j] * 3;
                }
            }
            mapResults = b.gather();
            runtimes[3] += MPI_Wtime() - t;
            if (CHECK && msl::isRootProcess()) {
                check_array_array_equal("mapInPlace", elements, mapResults, manmapResults);
            }
        }

        b.fill(2);

        if (check_str_inside(skeletons, "mapIndex,")) {

            DC<int> mapIndex(dim, dim, dim, 6);
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                b.mapIndex(sum4, mapIndex);

            }
            mapResults = b.gather();
            runtimes[4] += MPI_Wtime() - t;

            if (CHECK && msl::isRootProcess()) {
                int *mapIndex_comp = new int[elements];
                for (int j = 0; j < elements; j++) {
                    int depth = int(j / (dim * dim));
                    mapIndex_comp[j] = depth + int(j - (depth * dim * dim)) / dim + (j % dim) + 6;
                }
                check_array_array_equal("mapIndex", elements, mapResults, mapIndex_comp);
            }
        }

        b.fill(3);
        t = MPI_Wtime();
        if (check_str_inside(skeletons, "mapIndexInPlace,")) {

            for (int i = 0; i < reps; i++) {
                b.mapIndexInPlace(sum4);
            }
            mapResults = b.gather();
            runtimes[5] += MPI_Wtime() - t;

            if (CHECK && msl::isRootProcess()) {
                int *mapIndexInPlace_comp = new int[elements];
                for (int j = 0; j < elements; j++) {
                    mapIndexInPlace_comp[j] = 3;
                }
                for (int i = 0; i < reps; i++) {
                    for (int j = 0; j < elements; j++) {
                        int depth = int(j / (dim * dim));
                        mapIndexInPlace_comp[j] =
                                depth + int(j - (depth * dim * dim)) / dim + (j % dim) + mapIndexInPlace_comp[j];
                    }
                }
                check_array_array_equal("MapIndexInPlace", elements, mapResults, mapIndexInPlace_comp);
            }
        }
        // ************* Fold *********************** //
        if (check_str_inside(skeletons, "fold,")) {

            b.fill(3);
            int result = 0;
            t = MPI_Wtime();
            for (int i = 0; i < reps; i++) {
                result = b.fold(sum, true);
            }
            runtimes[10] += MPI_Wtime() - t;

            if (CHECK && msl::isRootProcess()) {
                check_value_value_equal("Fold", result, elements*3);
            }
        }
        // ************* Zip *********************** //
        DC<int> c(dim, dim, dim, 3);
        //DC<int>* d = new DC<int>(dim,dim,dim); delete d;
        DC<int> d(dim, dim, dim);
        int *zipResults = new int[dim * dim * dim];
        int *manzipResults = new int[dim * dim * dim];

        if (check_str_inside(skeletons, "zip,")) {
            b.fill(10);
            c.fill(20);

            t = MPI_Wtime();

            for (int i = 0; i < reps; i++) {
                d.zip(b, c, sum);
            }
            zipResults = d.gather();
            runtimes[6] += MPI_Wtime() - t;

            if (CHECK && msl::isRootProcess()) {
                check_array_value_equal("Zip", elements, zipResults, 30);
            }
        }
        if (check_str_inside(skeletons, "zipInPlace,")) {
            b.fill(10);
            c.fill(10);
            t = MPI_Wtime();

            for (int i = 0; i < reps; i++) {
                b.zipInPlace(c, sum);
            }
            zipResults = b.gather();
            runtimes[7] += MPI_Wtime() - t;
            for (int j = 0; j < elements; j++) {
                manzipResults[j] = 10;
            }
            for (int i = 0; i < reps; i++) {
                for (int j = 0; j < elements; j++) {
                    manzipResults[j] = 10 + manzipResults[j];
                }
            }
            if (CHECK && msl::isRootProcess()) {
                check_array_array_equal("ZipInPlace", elements, zipResults, manzipResults);
            }
        }

        if (check_str_inside(skeletons, "zipIndex,")) {
            b.fill(7);
            c.fill(5);
            t = MPI_Wtime();

            for (int i = 0; i < reps; i++) {
                d.zipIndex(b, c, sum5);
            }
            zipResults = d.gather();
            runtimes[8] += MPI_Wtime() - t;

            if (CHECK && msl::isRootProcess()) {
                int *zipIndex_comp = new int[elements];
                for (int j = 0; j < elements; j++) {
                    int depth = int(j / (dim * dim));
                    zipIndex_comp[j] = depth + int(j - (depth * dim * dim)) / dim + (j % dim) + 7 + 5;
                }
                check_array_array_equal("zipIndex", elements, zipResults, zipIndex_comp);
            }
        }


        b.fill(3);
        c.fill(2);
        if (check_str_inside(skeletons, "zipIndexInPlace,")) {
            t = MPI_Wtime();

            for (int i = 0; i < reps; i++) {
                b.zipIndexInPlace(c, mul5);
            }
            zipResults = b.gather();
            runtimes[9] += MPI_Wtime() - t;

            if (CHECK && msl::isRootProcess()) {
                int *zipIndexInPlace_comp = new int[elements];
                for (int i = 0; i < elements; i++) {
                    zipIndexInPlace_comp[i] = 3;
                }
                for (int i = 0; i < reps; i++) {
                    for (int j = 0; j < elements; j++) {
                        int depth = int(j / (dim * dim));
                        zipIndexInPlace_comp[j] = zipIndexInPlace_comp[j] * depth * (int(j - (depth * dim * dim))
                                / dim) * (j % dim) * 3 * 2;
                    }
                }
                check_array_array_equal("ZipIndexInPlace", elements, zipResults, zipIndexInPlace_comp);
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

    std::string nextfile = "Data/dc_" + std::to_string(msl::Muesli::num_total_procs) + "_" +
                          std::to_string(msl::Muesli::num_gpus) + "_" + std::to_string(dim) + "_" +
                          std::to_string(msl::Muesli::cpu_fraction);
    if (msl::isRootProcess() && OUTPUT) {
        printf("%d; %d; %d; %d; %.2f\n", dim, msl::Muesli::num_total_procs,
               msl::Muesli::num_local_procs, msl::Muesli::num_gpus, msl::Muesli::cpu_fraction);
    }
    msl::test::dc_test(dim, nextfile, reps, skeletons);
    msl::terminateSkeletons();
    return EXIT_SUCCESS;
}
