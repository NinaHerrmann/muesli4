/*
 * dc_test.cpp
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

#include "muesli.h"
#include "dc.h"
#include "functors.h"

typedef struct {
    float x;
    float y;
    float z;
    float w;
} f4;

int CHECK = 0;
int OUTPUT = 1;
namespace msl::dc_map_stencil {

    inline int index(int x, int y, int z, int w, int h, int d) {
        return (y) * w + x + (w * h) * z;
    }

    void printDC(DC<f4> &dc) {
        dc.updateHost();
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                f4 f = dc.localPartition[index(x, y, 1, dc.getCols(), dc.getRows(), dc.getDepth())];
                printf("(%f, %f, %f, %f), ", f.x, f.y, f.z, f.w);
            }
            printf("\n");
        }
    }

    void printDC(DC<float> &dc) {
        dc.updateHost();
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                printf("%f, ", dc.localPartition[index(x, y, 1, dc.getCols(), dc.getRows(), dc.getDepth())]);
            }
            printf("\n");
        }
    }

    void printDC(DC<int> &dc) {
        dc.updateHost();
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                printf("%i, ", dc.localPartition[index(x, y, 1, dc.getCols(), dc.getRows(), dc.getDepth())]);
            }
            printf("\n");
        }
    }

    MSL_USERFUNC float averageStencil(const PLCube<float> &cs, int x, int y, int z) {
        float sum = 0;
        float multiplier = 1.0f / (float) std::pow(2 * 1 + 1, 3);
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    sum += multiplier * cs(x + dx, y + dy, z + dz);
                }
            }
        }
        return sum;
    }

    MSL_USERFUNC int addStencil(const PLCube<int> &cs, int x, int y, int z) {
        int sum = 0;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    sum += cs(x + dx, y + dy, z + dz);
                }
            }
        }
        return sum;
    }

    MSL_USERFUNC f4 copy(const PLCube<f4> &cs, int x, int y, int z) {
        return cs(x, y, z);
    }

    class Op {
    public:
        MSL_USERFUNC int operator()(int x, int y, int z, int i) {
            return (int) 1;
        }
    };

    void dc_test(int dim) {
        Op op;
        DC<int> dc(100, 100, 16);
        DC<int> dc2(100, 100, 16);
        dc.fill(1);
        printDC(dc);
        printf("=== Executing stencil... ===\n\n");
        dc.mapStencil<addStencil>(dc2, 1, {});
        printDC(dc2);
        printf("\nCorners: \n");
        for(int x = 0; x < dc2.getCols(); x++) {
            for(int y = 0; y < dc2.getRows(); y++) {
                for(int z = 0; z < dc2.getDepth(); z++) {
                    int i = index(x, y, z, dc2.getCols(), dc2.getRows(), dc2.getDepth());
                    if (dc2.localPartition[i] <= 8) {
                        printf("(%i, %i, %i): %i\n", x, y, z, dc2.localPartition[i]);
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv){
  //printf("Starting Main...\n");
  msl::setNumRuns(1);
  msl::initSkeletons(argc, argv);
  msl::Muesli::cpu_fraction = 00;
  int dim = 0;
  if (msl::isRootProcess()) {
      printf("%d; %d; %d; %d; %.2f\n", dim, msl::Muesli::num_total_procs,
             msl::Muesli::num_local_procs, msl::Muesli::num_gpus, msl::Muesli::cpu_fraction);
  }
  msl::dc_map_stencil::dc_test(dim);
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}
