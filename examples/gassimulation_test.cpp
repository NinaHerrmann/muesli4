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
#include <csignal>

#include "muesli.h"
#include "dc.h"
#include "da.h"
#include "functors.h"
#include "array.h"
#include "vec3.h"

const int FLAG_OBSTACLE = 1 << 0;
const int FLAG_KEEP_VELOCITY = 1 << 1;

#ifdef __CUDACC__
#define MSL_MANAGED __managed__
#define MSL_CONSTANT __constant__
#else
#define MSL_MANAGED
#define MSL_CONSTANT
#endif

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
    return os;
}

int CHECK = 0;
int OUTPUT = 1;
namespace msl::gassimulation {

    MSL_MANAGED vec3<int> size;

    MSL_MANAGED float deltaT = 1.f;

    MSL_MANAGED float tau = 0.65;
    MSL_MANAGED float cellwidth = 1.0f;

    MSL_CONSTANT const array<vec3f, Q> offsets {
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

    MSL_CONSTANT const array<unsigned char, Q> opposite = {
            0,
            2, 1, 4, 3, 6, 5,
            10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15
    };

    MSL_CONSTANT const array<float, Q> wis {
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

    MSL_USERFUNC cell_t update(const PLCube<cell_t> &plCube, int x, int y, int z) {
        cell_t cell = plCube(x, y, z);

        auto* parts = (floatparts*) &cell[0];

        if (parts->exponent == 255 && parts->mantissa & FLAG_KEEP_VELOCITY) {
            return cell;
        }

        // Streaming.
        for (int i = 1; i < Q; i++) {
            int sx = x + (int) offsets[i].x;
            int sy = y + (int) offsets[i].y;
            int sz = z + (int) offsets[i].z;
            cell[i] = plCube(sx, sy, sz)[i];
        }

        // Collision.
        if (parts->exponent == 255 && parts->mantissa & FLAG_OBSTACLE) {
            if (parts->mantissa & FLAG_OBSTACLE) {
                cell_t cell2 = cell;
                for (size_t i = 1; i < Q; i++) {
                    cell[i] = cell2[opposite[i]];
                }
            }
            return cell;
        }
        float p = 0;
        vec3f vp {0, 0, 0};
        for (size_t i = 0; i < Q; i++) {
            p += cell[i];
            vp += offsets[i] * cellwidth * cell[i];
        }
        vec3f v = p == 0 ? vp : vp * (1 / p);

        for (size_t i = 0; i < Q; i++) {
            cell[i] = cell[i] + deltaT / tau * (feq(i, p, v) - cell[i]);
        }
        return cell;
    }

    class Initialize : public Functor4<int, int, int, cell_t, cell_t> {
    public:
        MSL_USERFUNC cell_t operator()(int x, int y, int z, cell_t c) const override {
            for (int i = 0; i < Q; i++) {
                c[i] = feq(i, 1.f, {.1f, 0, 0});
            }

            if (x <= 1 || y <= 1 || z <= 1 || x >= size.x - 2 || y >= size.y - 2 || z >= size.z - 2 ||
                std::pow(x - 50, 2) + std::pow(y - 50, 2) + std::pow(z - 8, 2) <= 225) {
                auto* parts = (floatparts*) &c[0];
                parts->sign = 0;
                parts->exponent = 255;
                if (x <= 1 || x >= size.x - 2 || y <= 1 || y >= size.y - 2 || z <= 1 || z >= size.z - 2) {
                    parts->mantissa = 1 << 22 | FLAG_KEEP_VELOCITY;
                } else {
                    parts->mantissa = 1 << 22 | FLAG_OBSTACLE;
                }
            }
            return c;
        }
    };

    void gassimulation_test(vec3<int> dimension, int iterations, const std::string &importFile, const std::string &exportFile) {
        size = dimension;
        DA<int> process(5,5);

        DC<cell_t> dc(size.x, size.y, size.z);

        if (importFile.empty()) {
            Initialize initialize;
            dc.mapIndexInPlace(initialize);
        } else {
            std::ifstream infile(importFile, std::ios_base::binary);

            std::vector<char> buffer((std::istreambuf_iterator<char>(infile)),
                                     std::istreambuf_iterator<char>());

            if (buffer.size() != size.x * size.y * size.z * sizeof(cell_t)) {
                std::cerr << "Inputfile is " << buffer.size() << " bytes big, but needs to be "
                          << size.x * size.y * size.z * sizeof(cell_t) << " to match the given dimensions!"
                          << std::endl;
                exit(-1);
            }

            auto *b = (cell_t *) buffer.data();

            infile.close();

            for (int i = 0; i < dc.getSize(); i++) {
                dc.localPartition[i] = b[i];
            }
            dc.setCpuMemoryInSync(true);
            dc.updateDevice();
        }

        DC<cell_t> dc2(size.x, size.y, size.z);

        // Pointers for swapping.
        DC<cell_t> *dcp1 = &dc;
        DC<cell_t> *dcp2 = &dc2;
        double totaltime = 0.0;
        double totalkerneltime = 0.0;
        double time = MPI_Wtime();

        for (int i = 0; i < iterations; i++) {
            dcp1->mapStencil<update>(*dcp2, 1, {});
            std::swap(dcp1, dcp2);
        }
        double endTime = MPI_Wtime();

        totaltime += time-endTime;

        if (msl::isRootProcess()) {
            std::cout << size.x << ";" << iterations << ";" << msl::Muesli::num_total_procs << ";" << msl::Muesli::num_threads << ";" << msl::Muesli::num_gpus << ";" << totaltime << ";" << totalkerneltime << std::endl;
        }

        if (!exportFile.empty()) {
            dcp1->updateHost();
            std::ofstream outfile(exportFile, std::ios_base::binary);
            outfile.write((char*) dcp1->localPartition, dcp1->getSize() * sizeof(cell_t));
            outfile.close();
        }
    }
}

void exitWithUsage() {
    std::cerr << "Usage: ./gassimulation_test [-d <xdim> <ydim> <zdim>] [-g <nGPUs>] [-n <iterations>] [-i <importFile>] [-e <exportFile>]" << std::endl;
    exit(-1);
}

int getIntArg(char* s, bool allowZero = false) {
    int i = std::atoi(s);
    if (i < 0 || (i == 0 && !allowZero)) {
        exitWithUsage();
    }
    return i;
}

int main(int argc, char** argv){
    vec3<int> size {100, 100, 100};
    int gpus = 1;
    int iterations = 1;
    std::string importFile, exportFile;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            exitWithUsage();
        }
        switch(argv[i++][1]) {
            case 'd':
                if (argc < i + 3) {
                    exitWithUsage();
                }
                size.x = getIntArg(argv[i++]);
                size.y = getIntArg(argv[i++]);
                size.z = getIntArg(argv[i]);
                break;
            case 'g':
                gpus = getIntArg(argv[i]);
                break;
            case 'n':
                iterations = getIntArg(argv[i], true);
                break;
            case 'i':
                importFile = std::string(argv[i]);
                break;
            case 'e':
                exportFile = std::string(argv[i]);
                break;
            case 't':
                msl::setNumThreads(getIntArg(argv[i]));
                break;
            default:
                exitWithUsage();
        }
    }

    msl::setNumRuns(1);
    msl::initSkeletons(argc, argv);
    msl::Muesli::cpu_fraction = 0;
    msl::Muesli::num_gpus = gpus;
    msl::gassimulation::gassimulation_test(size, iterations, importFile, exportFile);
    msl::terminateSkeletons();
    return EXIT_SUCCESS;
}