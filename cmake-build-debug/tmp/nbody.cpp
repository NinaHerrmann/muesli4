/*
 * nbody.cpp
 *
 *      Author: Steffen Ernsting <s.ernsting@uni-muenster.de>
 *
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2014 Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *                Herbert Kuchen <kuchen@uni-muenster.de.
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
#include <cstdlib>

#include "muesli.h"
#include "dc.h"
#include "da.h"
#include "rng.h"
#include "array.h"

/*typedef struct {
    float x, y, z, vx, vy, vz, mass, charge;
} particle;*/
typedef array<float, 8> particle;

namespace msl::examples::nbody {

#define EPSILON         1.0E-10f
#define DT              0.01f

    void show(DA<particle> &da) {
        da.updateHost();
        particle *p = da.getLocalPartition();

        for (int i = 0; i < da.getLocalSize(); i++) {
            std::cout << i << ": (" << p[i][0] << ", " << p[i][1] << ", " << p[i][2] << ")" << std::endl;
        }
        std::cout << "--------------------------------------------" << std::endl;
    }

    void initParticles(particle *particles, int nParticles) {
        Rng<float> rng;  // vorher: Rng<float, 0, 1> rng; HK 28.4.2020
        for (int i = 0; i < nParticles; i++) {
            particles[i] = particle();
            particles[i][0] = rng();
            particles[i][1] = rng();
            particles[i][2] = rng();
            particles[i][3] = 0.0f;
            particles[i][4] = 0.0f;
            particles[i][5] = 0.0f;
            particles[i][6] = 1.0f;
            particles[i][7] = 1.0f - 2.0f * (float) (i % 2);
        }
    }

    class CalcForce : public Functor2<int, particle, particle> {
    public:
        CalcForce(particle* p, int tile_width) : Functor2() {
            this->tw = tile_width;
            this->oldParticles = DA<particle>({});
            // this->setTileWidth(tw);
        }

        MSL_USERFUNC particle operator()(int curIndex, particle curParticle) const override {
            return {}; //tiling(curIndex, curParticle);
        }

        MSL_USERFUNC
        [[nodiscard]] particle simple(int curIndex, particle curParticle) const {
            float ax = 0.0f;
            float ay = 0.0f;
            float az = 0.0f;

            // calculate forces for the current particle
            for (int j = 0; j < oldParticles.getSize(); j++) {

                // do not evaluate interaction with yourself.
                if (j != curIndex) {

                    // Evaluate forces that j-particles exert on the i-particle.
                    float dx, dy, dz, r2, r, qj_by_r3;

                    // Here we absorb the minus sign by changing the order of i and j.
                    dx = curParticle[0] - oldParticles[j][0];
                    dy = curParticle[1] - oldParticles[j][1];
                    dz = curParticle[2] - oldParticles[j][2];

                    r2 = dx * dx + dy * dy + dz * dz;
                    r = sqrtf(r2);

                    // Quench the force if the particles are too close.
                    if (r < EPSILON)
                        qj_by_r3 = 0.0f;
                    else
                        qj_by_r3 = oldParticles[j].charge / (r2 * r);

                    // accumulate the contribution from particle j.
                    ax += qj_by_r3 * dx;
                    ay += qj_by_r3 * dy;
                    az += qj_by_r3 * dz;
                }
            }

            // advance current particle
            float vx0 = curParticle[3];
            float vy0 = curParticle[4];
            float vz0 = curParticle[5];

            float qidt_by_m = curParticle[6] * DT / curParticle[7];
            curParticle[3] += ax * qidt_by_m;
            curParticle[4] += ay * qidt_by_m;
            curParticle[5] += az * qidt_by_m;

            // Use average velocity in the interval to advance the particles' positions
            curParticle[0] += (vx0 + curParticle[3]) * DT * 0.5f;
            curParticle[1] += (vy0 + curParticle[4]) * DT * 0.5f;
            curParticle[2] += (vz0 + curParticle[5]) * DT * 0.5f;

            return curParticle;
        }

        MSL_USERFUNC
        [[nodiscard]] particle tiling(int curIndex, particle curParticle) const {
            float ax = 0.0f;
            float ay = 0.0f;
            float az = 0.0f;

            // calculate forces for the current particle
            for (int t = 0; t < oldParticles.getSize() / tw; t++) {
                auto t_oP = oldParticles.getTile(t, this);

                for (int i = 0; i < tw; i++) {
                    // do not evaluate interaction with yourself.
                    if (t * tw + i != curIndex) {
                        // Evaluate forces that j-particles exert on the i-particle.
                        float dx, dy, dz, r2, r, qj_by_r3;

                        // Here we absorb the minus sign by changing the order of i and j.
                        dx = curParticle[0] - t_oP.get(i)[0];
                        dy = curParticle[1] - t_oP.get(i)[1];
                        dz = curParticle[2] - t_oP.get(i)[2];

                        r2 = dx * dx + dy * dy + dz * dz;
                        r = sqrtf(r2);

                        // Quench the force if the particles are too close.
                        if (r < EPSILON)
                            qj_by_r3 = 0.0f;
                        else
                            qj_by_r3 = t_oP.get(i)[6] / (r2 * r);

                        // accumulate the contribution from particle j.
                        ax += qj_by_r3 * dx;
                        ay += qj_by_r3 * dy;
                        az += qj_by_r3 * dz;
                    }
                }
            }

            // advance current particle
            float vx0 = curParticle[3];
            float vy0 = curParticle[4];
            float vz0 = curParticle[5];

            float qidt_by_m = curParticle[6] * DT / curParticle[7];
            curParticle[3] += ax * qidt_by_m;
            curParticle[4] += ay * qidt_by_m;
            curParticle[5] += az * qidt_by_m;

            // Use average velocity in the interval to advance the current particle's position
            curParticle[0] += (vx0 + curParticle[3]) * DT * 0.5f;
            curParticle[1] += (vy0 + curParticle[4]) * DT * 0.5f;
            curParticle[2] += (vz0 + curParticle[5]) * DT * 0.5f;

            return curParticle;
        }
    private:
        int tw;
        DA<particle> oldParticles;

    };

    void testNBody(particle *const particles, int nParticles, int steps, int tile_width, bool output) {
        DA<particle> P(nParticles, {});
        DA<particle> oldP(nParticles, {}); // copy distributed

        if (output && msl::isRootProcess()) {
            std::cout << "Initial Particle System:" << std::endl;
            show(oldP);
        }

        CalcForce cf(oldP, tile_width);

        for (int i = 0; i < steps; i++) {
            P.mapIndexInPlace(cf);

            P.gather(oldP);
            if (output && msl::isRootProcess()) {
                show(oldP);
            }
        }
    }

} // namespace msl

int main(int argc, char **argv) {
    msl::initSkeletons(argc, argv);
    msl::Rng<float> rng_init{};  // vorher: msl::Rng<float, 0, 1> rng_init{};  HK 28.4.2020

    int nParticles = 64;
    int nSteps = 5;
    int nRuns = 1;
    int nGPUs = 2;
    int tile_width = 8;
    bool output = false;
    bool noWarmup = true;
    if (argc < 5) {
        if (msl::isRootProcess()) {
            std::cout << std::endl << std::endl;
            std::cout << "Usage: " << argv[0] << " #nParticles #timesteps #nRuns #nGPUs" << std::endl;
            std::cout << "Default values: nParticles = " << nParticles
                      << ", timesteps = " << nSteps
                      << ", nRuns = " << nRuns
                      << ", nGPUs = " << nGPUs
                      << std::endl;
            std::cout << std::endl << std::endl;
        }
        output = true;
        noWarmup = true;
    } else {
        nParticles = atoi(argv[1]);
        nSteps = atoi(argv[2]);
        nRuns = atoi(argv[3]);
        nGPUs = atoi(argv[4]);
        tile_width = 1024;
#ifdef __CUDACC__
        noWarmup = 0;
#endif
    }

    msl::setNumRuns(nRuns);
    msl::setNumGpus(nGPUs);

    // initialize particle system
    auto *particles = new msl::examples::nbody::Particle[nParticles];
    if (msl::isRootProcess())
        msl::examples::nbody::initParticles(particles, nParticles);

    // broadcast particle system
    msl::MSL_Broadcast(0, particles, nParticles);

    // warmup
    if (!noWarmup) {
        msl::Timer tw("Warmup");
        msl::examples::nbody::testNBody(particles, nParticles, nSteps, tile_width, 0);
        msl::printv("Warmup: %fs\n", tw.stop());
    }

    msl::startTiming();
    for (int run = 0; run < msl::Muesli::num_runs; run++) {
        msl::examples::nbody::testNBody(particles, nParticles, nSteps, tile_width, output);
        msl::splitTime(run);
    }
    msl::stopTiming();

    delete[] particles;

    msl::terminateSkeletons();

    return 0;
}







