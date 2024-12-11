/*
 * aco_flippedds.cpp
 *
 *      Author: Nina Hermann,
 *  		Herbert Kuchen <kuchen@uni-muenster.de>
 *
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2020  Herbert Kuchen <kuchen@uni-muenster.de.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#include <cmath>
#include "dm.h"
#include "da.h"
#include "muesli.h"
#include <DecisionTree.h>

#include <algorithm>
#include "array.h"
#include "Randoms.h"

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

#include <random>
namespace msl::gbdt {


    class ReduceRow : public Functor2<double, double, double> {
    public:
        MSL_USERFUNC double operator()(double first, double second) const override {
            return first + second;
        }};
    class Min : public Functor2<double, double, double> {
    public:
        MSL_USERFUNC double operator()(double x, double y) const override {
            if (x < y) { return x; }
            return y;}
    };
    void readFileProperties(const std::string &problem, bool is_palma) {
        std::string palma_path;
        if (is_palma) {
            palma_path = "/home/n/n_herr03/muesli/data/gbdt/";
        } else {
            palma_path = "/Users/ninaherrmann/Research/muesli/data/gbdt/";
        }
        std::string filename = palma_path + problem + ".train";
        std::ifstream file(filename);
        std::vector<std::string> lines; // Vector to store lines

        if (!file.is_open()) {
            std::cerr << "Error opening file!" << std::endl;
            exit(1);
        }

        // First iteration: Read lines from the file
        std::string line;
        while (std::getline(file, line)) {
            lines.push_back(line); // Store each line in the vector
        }

        // Close the file after the first read
        file.close();
        int numRows = 0;
        int numCols = 0;

        for (const auto& st_line : lines) {
            if (st_line.empty()) continue;
            numRows++;
            std::istringstream ss(st_line);
            std::string feature;
            ss >> feature;
            int count = 0;
            while (ss >> feature) {
                count = std::stoi(feature.substr(0, feature.find(':')));
            }
            if (count > numCols) {
                numCols = count;
            }
        }
        printf("ncolumn %d nrow %d", numCols, numRows);
        numCols += 1;
        DM<float> train_data = DM<float>(numRows, numCols, 0.0);
        int row = 0;
        for (const auto& st_line : lines) {
            if (st_line.empty()) continue; // Skip empty lines
            std::istringstream ss(st_line);
            float label;
            ss >> label;
            train_data.set2D(row, 0, label);
            std::string feature;
            // Read features
            while (ss >> feature) {
                // Each feature is in the form of "index:value"
                int index = std::stoi(feature.substr(0, feature.find(':')));
                float value = std::stof(feature.substr(feature.find(':') + 1));
                train_data.set2D(row, index, value); // Use index since we store label in the first column
            }
            row++;
        }
        train_data.show("Train Data");
    }
    void fit(DM<float> data) {
        int numSamples = data.getnCol();
        int numFeatures = data.getnRow();
        // Calculate initial residuals (original targets)
        int numTrees;
        std::vector<double> residuals;
        std::vector<DecisionTree::TreeNode*> stumps;
        for (const auto& row : data) {
            residuals.push_back(row);
        }

        for (int i = 0; i < numTrees; ++i) {
            DecisionTree tree;
            DecisionTree::TreeNode* stump = tree.buildStump(data, residuals);

            // Add the stump to the ensemble (in a real implementation, use boost weights, etc.)
            stumps.push_back(stump);

            // Update the residuals
            for (size_t j = 0; j < data.size(); ++j) {
                double prediction = (data[j][stump->featureIndex] <= stump->threshold) ? stump->leftValue : stump->rightValue;
                residuals[j] -= prediction; // Update residual
            }
        }
    }
    void gbdt(int iterations, const std::string &importFile, int ntrees, int isPalma = 0, int ndepth = 3) {
        startTiming();
        bool is_palma = false;
        if (isPalma == 1) {
            is_palma = true;
        }
        readFileProperties(importFile, is_palma);

        // Create one tree.
        for (int i = 0; i < ntrees; i++) {
            // Create one level.
            for (int j = 0; j < ndepth; j++) {

            }
        }
        syncStreams();
        double calctime = msl::stopTiming();
        printf("%s;%d;%s;%f;", importFile.c_str(), ntrees, "flippedds", calctime);
        syncStreams();
        stopTiming();
    }
} // close namespace

void exitWithUsage() {
    std::cerr
            << "Usage: ./gbdt_[cpu|gpu|seq] [-g <nGPUs>] [-n <iterations>] [-i <ntrees>] [-d <pathtodata>] [-t <threads>] [-l <depth>]  [-h <hpcenvironment?>]"
            << std::endl;
    exit(-1);
}

int getIntArg(char *s, bool allowZero = false) {
    int i = std::atoi(s);
    if (i < 0 || (i == 0 && !allowZero)) {
        printf("zero value!!!!!!!!!!!!!\n");
        exitWithUsage();
    }
    return i;
}

int main(int argc, char **argv) {
    msl::initSkeletons(argc, argv);
    int gpus = 1, iterations = 1, runs = 1, ntrees = 7, hpc = 0, depth = 3;
    std::string importFile;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            printf("not a - argument!!!!!!!!!!!! \n");
            exitWithUsage();
        }
        switch (argv[i++][1]) {
            case 'g':
                gpus = getIntArg(argv[i], true);
                break;
            case 'i':
                ntrees = getIntArg(argv[i]);
                break;
            case 'n':
                iterations = getIntArg(argv[i], true);
                break;
            case 'd':
                importFile = std::string(argv[i]);
                break;
            case 't':
                msl::setNumThreads(getIntArg(argv[i]));
                break;
            case 'l':
                depth = getIntArg(argv[i]);
                break;
            case 'h':
                hpc = getIntArg(argv[i], true);
                break;
            default:
                printf("entering default\n");
                exitWithUsage();
        }
    }
    msl::setNumGpus(gpus);
    msl::setNumRuns(runs);
    if (!importFile.empty()) {
        msl::gbdt::gbdt(iterations, importFile, ntrees, depth, false);
    } else {
        printf("Providing an import file is mandatory. \n");
        exit(-1);
    }
    msl::terminateSkeletons();
    return EXIT_SUCCESS;
}
