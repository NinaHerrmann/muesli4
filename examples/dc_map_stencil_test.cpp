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

#include "muesli.h"
#include "dc.h"
#include "functors.h"

int CHECK = 0;
int OUTPUT = 1;
namespace msl::dc_map_stencil_test {

    MSL_USERFUNC float averageStencil(PLCube<float> &cs, int x, int y, int z) {
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

    void dc_test(int dim) {
        DC<float> dc(5, 5, 5);
        DC<float> dc2(5, 5, 5);
        dc.fill(0.f);
        for (int i = 0; i < 5 * 5 * 5; i++) {
            dc.set(i, (float)(i % 2));
        }
        dc.prettyPrint();
        printf("=== Executing stencil... ===\n\n");
        dc.mapStencil<averageStencil>(dc2, 1, 0.f);
        dc2.prettyPrint();
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
  msl::dc_map_stencil_test::dc_test(dim);
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}
