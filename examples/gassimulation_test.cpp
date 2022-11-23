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

typedef struct {
    float x;
    float y;
    float z;
    float w;
} f4;

std::ostream& operator<< (std::ostream& os, const f4 f) {
    os << "(" << f.x << ", " << f.y << ", " << f.z << ", " << f.w << ")";
}

int CHECK = 0;
int OUTPUT = 1;
namespace msl::gassimulation {

    const float deltaT = 0.01f;
    const float viscosity = 0.005;
    const float cellwidth = 0.05;
    const float EPSILON = 0.00001;

    MSL_USERFUNC f4 updateU(const PLCube<f4> &cs, int x, int y, int z) {
        const f4 u = cs(x, y, z);
        const f4 u1n = cs(x - 1, y, z);
        const f4 u1p = cs(x + 1, y, z);
        const f4 u2n = cs(x, y - 1, z);
        const f4 u2p = cs(x, y + 1, z);
        const f4 u3n = cs(x, y, z - 1);
        const f4 u3p = cs(x, y, z + 1);

        const float f = 1.f / (2 * cellwidth) * (u1p.x - u1n.x + u2p.y - u2n.y + u3p.z - u3n.z);

        f4 res;
        res.x = u.x + deltaT * (viscosity / (cellwidth * cellwidth) *
                                (u1n.x + u1p.x + u2n.x + u2p.x + u3n.x + u3p.x - 6.f * u.x)
                                - f * u.x
                                - (u1n.w - u1p.w) / (2 * cellwidth));
        res.y = u.y + deltaT * (viscosity / (cellwidth * cellwidth) *
                                (u1n.y + u1p.y + u2n.y + u2p.y + u3n.y + u3p.y - 6.f * u.y)
                                - f * u.y
                                - (u2n.w - u2p.w) / (2 * cellwidth));
        res.z = u.z + deltaT * (viscosity / (cellwidth * cellwidth) *
                                (u1n.z + u1p.z + u2n.z + u2p.z + u3n.z + u3p.z - 6.f * u.z)
                                - f * u.z
                                - (u3n.w - u3p.w) / (2 * cellwidth));
        res.w = u.w;
        return res;
    }

    MSL_USERFUNC f4 updateUFromP(const PLCube<f4> &cs, int x, int y, int z) {
        f4 res = cs(x, y, z);
        res.x -= (cs(x + 1, y, z).w - cs(x - 1, y, z).w) / (2 * cellwidth);
        res.y -= (cs(x, y + 1, z).w - cs(x, y - 1, z).w) / (2 * cellwidth);
        res.z -= (cs(x, y, z + 1).w - cs(x, y, z - 1).w) / (2 * cellwidth);
        return res;
    }

    MSL_USERFUNC f4 updatePSingleIteration(const PLCube<f4> &cs, int x, int y, int z) {
        const f4 u1n = cs(x - 1, y, z);
        const f4 u1p = cs(x + 1, y, z);
        const f4 u2n = cs(x, y - 1, z);
        const f4 u2p = cs(x, y + 1, z);
        const f4 u3n = cs(x, y, z - 1);
        const f4 u3p = cs(x, y, z + 1);

        float ud = (cellwidth / 2.f) * (
                u1p.x - u1n.x
                + u2p.y - u2n.y
                + u3p.z - u3n.z
        );

        f4 res = cs(x, y, z);
        res.w = ((cellwidth * cellwidth) / 6.f) * (u1p.w + u1n.w + u2p.w + u2n.w + u3p.w + u3n.w - ud);
        return res;
    }

    class CalcError : public Functor2<f4, f4, bool>{
        public: MSL_USERFUNC bool operator() (f4 x, f4 y) const override {
            return abs(x.w) - abs(y.w) / (abs(x.w) + abs(y.w)) > EPSILON;
        }
    };

    class Or : public Functor2<bool, bool, bool> {
    public: MSL_USERFUNC bool operator() (bool x, bool y) const override {
            return x || y;
        }
    };

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

    void printDC(DC<bool> &dc) {
        dc.updateHost();
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                bool f = dc.localPartition[index(x, y, 1, dc.getCols(), dc.getRows(), dc.getDepth())];
                printf("(%i), ", f);
            }
            printf("\n");
        }
    }

    template <typename T>
    class Identity : public Functor<T, T> {
    public: MSL_USERFUNC T operator() (T x) const override {
            return x;
        }
    };

    MSL_USERFUNC f4 copy(const PLCube<f4> &cs, int x, int y, int z) {
        return cs(x, y, z);
    }



    void dc_test() {
        const std::string inputFile = "../Data/gassimulation.raw";
        std::ifstream infile(inputFile, std::ios_base::binary);

        std::vector<char> buffer((std::istreambuf_iterator<char>(infile)),
                std::istreambuf_iterator<char>());

        auto* b = (f4*) buffer.data();

        DC<f4> dc(100, 100, 16);
        for (int i = 0; i < dc.getSize(); i++) {
            dc.localPartition[i] = b[i];
        }
        dc.setCpuMemoryInSync(true);
        dc.updateDevice();

        DC<f4> dc2(100, 100, 16);
        DC<bool> difference(100, 100, 16);
        CalcError calcError;
        Or orFunctor;
        printf("=== Executing stencil... ===\n");
        f4 neutral{};

        // Pointers for swapping.
        DC<f4> *dcp1 = &dc;
        DC<f4> *dcp2 = &dc2;

        for (int i = 0; i < 2; i++) {
            int iterations = 0;
            dcp1->mapStencil<updateU>(*dcp2, 1, neutral);
            printDC(*dcp1);
            printf("\n");
            printDC(*dcp2);
            std::swap(dcp1, dcp2);
            printf("\nmapStencil updateU: \n");

            do {
                dcp1->mapStencil<updatePSingleIteration>(*dcp2, 1, neutral);
                printf("\nmapStencil updatePSingleIteration: \n");
                printDC(*dcp2);
                difference.zip(*dcp2, *dcp1, calcError);
                printf("\ndifference: \n");
                printDC(difference);
                std::swap(dcp1, dcp2);
                iterations++;
            } while (difference.fold(orFunctor, true));

            dcp1->mapStencil<updateUFromP>(*dcp2, 1, neutral);
            std::swap(dcp1, dcp2);

            printf("iterations: %i\n", iterations);
        }
    }
}

int main(int argc, char** argv){
  //printf("Starting Main...\n");
  msl::setNumRuns(1);
  msl::initSkeletons(argc, argv);
  msl::Muesli::cpu_fraction = 0;
  int dim = 0;
  if (msl::isRootProcess()) {
      printf("%d; %d; %d; %d; %.2f\n", dim, msl::Muesli::num_total_procs,
             msl::Muesli::num_local_procs, msl::Muesli::num_gpus, msl::Muesli::cpu_fraction);
  }
  msl::gassimulation::dc_test();
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}
