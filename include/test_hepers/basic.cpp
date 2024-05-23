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
 * Copyright 2022  Nina Hermann
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
#include "muesli.h"
#include <string>
#include <algorithm>

// https://stackoverflow.com/questions/3613284/c-stdstring-to-boolean
bool to_bool(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    std::istringstream is(str);
    bool b;
    is >> std::boolalpha >> b;
    return b;
}
void arg_helper(int argc, char** argv, int &dim, int &nGPUs, int &reps, bool &CHECK,  char *  & skeletons) {

    msl::setNumRuns(1);
    msl::initSkeletons(argc, argv);

    argc >= 2 ? dim = std::stoi(argv[1]) : dim = 4;
    argc >= 3 ? nGPUs = std::stoi(argv[2]) : nGPUs = 0;
    argc >= 4 ? msl::Muesli::cpu_fraction = std::strtod(argv[3], nullptr) : msl::Muesli::cpu_fraction = 0.0;
    argc >= 5 ? CHECK = to_bool(argv[4]) : CHECK = true;
    argc >= 6 ? reps = std::stoi(argv[5]) : reps = 1;
    argc >= 7 ? msl::setNumThreads(std::stoi(argv[6])) : msl::setNumThreads(32);
    argc >= 8 ? skeletons = argv[7] : skeletons = "all";
    if (msl::isRootProcess()){
        if (argc < 2 || CHECK) {
            printf("FYI Taking cl arguments: #elements \t nGPUs \t cpu-Fraction \t compare results to non parallel results "
                   "(true/false) \t repetitions \t skeletons\n");
        }
        if (CHECK) {
            printf("Executing with\t \t  %d \t\t %d \t %f \t\t %s \t\t\t\t\t\t %d \t\t %s\n", dim, nGPUs,
                   msl::Muesli::cpu_fraction,
                   CHECK ? "true" : "false", reps, skeletons);
        }
    }
    msl::setNumGpus(nGPUs);

}
bool check_str_inside(const char* skeletons, const char* single_skeleton) {
    if (strstr(skeletons, single_skeleton) != nullptr || strstr(skeletons, "all") != nullptr) {
        return true;
    }
    return false;
}

template <typename T>
void check_array_array_equal(const char* skelet, int elements, T array[], T expected_array[]){
        for (int i = 0; i < elements; i++) {
            if (array[i] != expected_array[i]) {
                printf("%s \t\t\t\t \xE2\x9C\x97 At Index %d - Value %.2f != %.2f \nNo further checking.\n", skelet,
                       i, array[i], expected_array[i]);
                break;
            }
            if (i == (elements) - 1) {
                printf("%s \t\t\t \xE2\x9C\x93\n", skelet);
            }
        }
}
template <typename T>
void check_array_value_equal(const char* skelet, int elements, T array[], T expected_value){
        for (int i = 0; i < elements; i++) {
            if (array[i] != expected_value) {
                printf("%s \t\t\t\t \xE2\x9C\x97 At Index %d - Value %.2f != %.2f \nNo further checking.\n", skelet,
                       i, array[i], expected_value);
                break;
            }
            if (i == (elements) - 1) {
                printf("%s \t\t\t \xE2\x9C\x93\n", skelet);
            }
        }
}
template <typename T>
void check_value_value_equal(const char* skelet, T value, T expected_value){
    if (value != expected_value) {
        printf("%s \t\t\t\t \xE2\x9C\x97 Value %.2f != %.2f\nNo further checking.\n", skelet,
                value, expected_value);
    } else {
        printf("%s \t\t\t \xE2\x9C\x93\n", skelet);
    }
}
template <typename T>
void print_and_doc_runtimes(int OUTPUT, const std::string &nextfile, T runtimes[], int arraysize, int dim, int reps){
    if (true) {
        std::ofstream outputFile;
        outputFile.open(nextfile, std::ios_base::app);
        outputFile << std::to_string(msl::Muesli::cpu_fraction) + ";" + std::to_string(msl::Muesli::num_threads) + ";"
            + std::to_string(dim) + ";" + std::to_string(reps) + ";";
        for (int i = 0; i < arraysize; i++) {
            outputFile << "" + std::to_string(runtimes[i]) + ";";
        }
        outputFile << "\n";
        outputFile.close();
    }
    // printf("Map; \tMapInPlace; \tMapIndex; \tMapIndexInPlace; \tZip; \tZipInPlace; \tZipIndex; \tZipIndexInPlace; \tFold\n");
    for (int i = 0; i < arraysize; i++) {
        printf("%.4f;", runtimes[i]);
    }
}