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
#include <cstring>
#include <csignal>

#include "muesli.h"
#include "dc.h"
#include "functors.h"
#include "array.h"
#include "vec3.h"

typedef struct {
    unsigned int mantissa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
} floatparts;

const size_t Q = 19;
typedef array<float, Q> cell_t;
typedef vec3<float> vec3f;

std::ostream& operator<< (std::ostream& os, const cell_t f) {
    os << "(" << f[0] << ", " << f[1] << ", " << f[2] << "...)";
}

int CHECK = 0;
int OUTPUT = 1;
namespace msl::gassimulation {

    __managed__ float deltaT = 0.001f;

    __managed__ float tau = 0.0007;
    __managed__ float cellwidth = .01f;

    __constant__ const array<vec3f, Q> offsets {
    0, 0, 0,   // 0
    -1, 0, 0,  // 1
    1, 0, 0,   // 2
    0, -1, 0,  // 3
    0, 1, 0,   // 4
    0, 0, -1,  // 5
    0, 0, 1,   // 6
    -1, -1, 0, // 7
    -1, 1, 0,  // 8
    1, -1, 0,  // 9
    1, 1, 0,   // 10
    -1, 0, -1, // 11
    -1, 0, 1,  // 12
    1, 0, -1,  // 13
    1, 0, 1,   // 14
    0, -1, -1, // 15
    0, -1, 1,  // 16
    0, 1, -1,  // 17
    0, 1, 1,   // 18
};

__constant__ const array<unsigned char, Q> opposite = {
        0,
        2, 1, 4, 3, 6, 5,
        10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15
};

__constant__ const array<float, Q> wis {
1.f / 3,
1.f / 18,
1.f / 18,
1.f / 18,
1.f / 18,
1.f / 18,
1.f / 18,
1.f / 36,
1.f / 36,
1.f / 36,
1.f / 36,
1.f / 36,
1.f / 36,
1.f / 36,
1.f / 36,
1.f / 36,
1.f / 36,
1.f / 36,
1.f / 36,
};

    MSL_USERFUNC inline float feq(size_t i, float p, const vec3f& v) {
        float wi = wis[i];
        float c = cellwidth;
        float dot = offsets[i] * c * v;
        return wi * p * (1 + (1 / (c * c)) * (3 * dot + (9 / (2 * c * c)) * dot * dot - (3.f / 2) * (v * v)));
    }

    MSL_USERFUNC cell_t stream(const PLCube<cell_t> &plCube, int x, int y, int z) {
        cell_t cell = plCube(x, y, z);

        floatparts* parts = (floatparts*) &cell[0];

        if (parts->exponent == 255) {
            return plCube(x, y, z);
        }

        for (int i = 1; i < Q; i++) {
            int sx = x + (int) offsets[i].x;
            int sy = y + (int) offsets[i].y;
            int sz = z + (int) offsets[i].z;
            cell[i] = plCube(sx, sy, sz)[i];
        }
        return cell;
    }

    class Collision : public Functor<cell_t, cell_t> {
    public:
        MSL_USERFUNC cell_t operator() (cell_t cell) const override {
            float p = 0;
            float c = cellwidth;
            floatparts* parts = (floatparts*) &cell[0];
            if (parts->exponent == 255) {
                if ((parts->mantissa & 1) != 0) {
                    for (size_t i = 1; i < Q; i++) {
                        cell[i] = cell[opposite[i]];
                    }
                }
                return cell;
            }
            vec3f vp {0, 0, 0};
            for (size_t i = 0; i < Q; i++) {
                p += cell[i];
                vp += offsets[i] * c * cell[i];
            }
            vec3f v = p == 0 ? vp : vp * (1 / p);

            for (size_t i = 0; i < Q; i++) {
                cell[i] = cell[i] + deltaT / tau * (feq(i, p, v) - cell[i]);
            }
        }
    };

    void dc_test() {
        const std::string inputFile = "../Data/gassimulation0.bin";
        std::ifstream infile(inputFile, std::ios_base::binary);

        std::vector<char> buffer((std::istreambuf_iterator<char>(infile)),
                std::istreambuf_iterator<char>());

        auto* b = (cell_t*) buffer.data();

        DC<cell_t> dc(200, 200, 200);
        dc.fill({0.f, 0.f, 0.f, 0.f});
        for (int i = 0; i < dc.getSize(); i++) {
            dc.localPartition[i] = b[i];
        }
        dc.setCpuMemoryInSync(true);
        dc.updateDevice();

        DC<cell_t> dc2(200, 200, 200);

        // Pointers for swapping.
        DC<cell_t> *dcp1 = &dc;
        DC<cell_t> *dcp2 = &dc2;

        Collision collision;

        for (int i = 0; i < 10; i++) {

            double time = MPI_Wtime();

            dcp1->mapInPlace(collision);
            dcp1->mapStencil<stream>(*dcp2, 1, {});

            double totalTime = MPI_Wtime() - time;

            printf("Time: %f\n", totalTime);

            std::swap(dcp1, dcp2);
        }

        //const std::string outputFile = "output.dat";
        //std::ofstream outfile(outputFile, std::ios_base::binary);

        dcp1->updateHost();

        //outfile.write((char*) dcp1->localPartition, dcp1->getSize() * sizeof(cell_t));
    }
}

int main(int argc, char** argv){
  //printf("Starting Main...\n");
  msl::setNumRuns(1);
  msl::initSkeletons(argc, argv);
  msl::Muesli::cpu_fraction = 0;
  msl::Muesli::num_gpus = 1;
  int dim = 0;
  if (msl::isRootProcess()) {
      printf("%d; %d; %d; %d; %.2f\n", dim, msl::Muesli::num_total_procs,
             msl::Muesli::num_local_procs, msl::Muesli::num_gpus, msl::Muesli::cpu_fraction);
  }
  msl::gassimulation::dc_test();
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}
