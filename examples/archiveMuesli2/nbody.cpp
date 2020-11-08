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

#include "darray.h"
#include "rng.h"

namespace msl {

namespace examples {

namespace nbody {

#define EPSILON         1.0E-10f
#define DT              0.01f

struct Particle
{
  float x, y, z, vx, vy, vz, mass, charge;
};

void show(DArray<Particle>& da)
{
  da.download();
  Particle* p = da.getLocalPartition();

  for (int i = 0; i < da.getLocalSize(); i++) {
    std::cout << i << ": (" << p[i].x << ", " << p[i].y << ", " << p[i].z << ")" << std::endl;
  }
  std::cout << "--------------------------------------------" << std::endl;
}

void initParticles(Particle* particles, int nParticles)
{
  Rng<float> rng;  // vorher: Rng<float, 0, 1> rng; HK 28.4.2020
  for (int i = 0; i < nParticles; i++) {
    particles[i] = Particle();
    particles[i].x = rng();
    particles[i].y = rng();
    particles[i].z = rng();
    particles[i].vx = 0.0f;
    particles[i].vy = 0.0f;
    particles[i].vz = 0.0f;
    particles[i].mass = 1.0f;
    particles[i].charge = 1.0f - 2.0f * (float) (i % 2);
  }
}

struct CalcForce : public AMapIndexFunctor<Particle, Particle>
{
  int tw;
  LArray<Particle> oldParticles;

  CalcForce(DArray<Particle>& p, int tile_width)
    : tw(tile_width), oldParticles(p, this, Distribution::COPY)
  {
    this->setTileWidth(tw);
  }

  MSL_USERFUNC
  Particle operator()(int curIndex, Particle curParticle) const
  {
    return tiling(curIndex, curParticle);
  }

  MSL_USERFUNC
  Particle simple(int curIndex, Particle curParticle) const
  {
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
        dx = curParticle.x - oldParticles[j].x;
        dy = curParticle.y - oldParticles[j].y;
        dz = curParticle.z - oldParticles[j].z;

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
    float vx0 = curParticle.vx;
    float vy0 = curParticle.vy;
    float vz0 = curParticle.vz;

    float qidt_by_m = curParticle.charge * DT / curParticle.mass;
    curParticle.vx += ax * qidt_by_m;
    curParticle.vy += ay * qidt_by_m;
    curParticle.vz += az * qidt_by_m;

    // Use average velocity in the interval to advance the particles' positions
    curParticle.x += (vx0 + curParticle.vx) * DT * 0.5f;
    curParticle.y += (vy0 + curParticle.vy) * DT * 0.5f;
    curParticle.z += (vz0 + curParticle.vz) * DT * 0.5f;

    return curParticle;
  }

  MSL_USERFUNC
  Particle tiling(int curIndex, Particle curParticle) const
  {
    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    // calculate forces for the current particle
    for (int t = 0; t < oldParticles.getSize()/tw; t++) {
      auto t_oP = oldParticles.getTile(t, this);

      for (int i = 0; i < tw; i++) {
        // do not evaluate interaction with yourself.
        if (t*tw+i != curIndex) {
          // Evaluate forces that j-particles exert on the i-particle.
          float dx, dy, dz, r2, r, qj_by_r3;

          // Here we absorb the minus sign by changing the order of i and j.
          dx = curParticle.x - t_oP.get(i).x;
          dy = curParticle.y - t_oP.get(i).y;
          dz = curParticle.z - t_oP.get(i).z;

          r2 = dx * dx + dy * dy + dz * dz;
          r = sqrtf(r2);

          // Quench the force if the particles are too close.
          if (r < EPSILON)
            qj_by_r3 = 0.0f;
          else
            qj_by_r3 = t_oP.get(i).charge / (r2 * r);

          // accumulate the contribution from particle j.
          ax += qj_by_r3 * dx;
          ay += qj_by_r3 * dy;
          az += qj_by_r3 * dz;
        }
      }
    }

    // advance current particle
    float vx0 = curParticle.vx;
    float vy0 = curParticle.vy;
    float vz0 = curParticle.vz;

    float qidt_by_m = curParticle.charge * DT / curParticle.mass;
    curParticle.vx += ax * qidt_by_m;
    curParticle.vy += ay * qidt_by_m;
    curParticle.vz += az * qidt_by_m;

    // Use average velocity in the interval to advance the current particle's position
    curParticle.x += (vx0 + curParticle.vx) * DT * 0.5f;
    curParticle.y += (vy0 + curParticle.vy) * DT * 0.5f;
    curParticle.z += (vz0 + curParticle.vz) * DT * 0.5f;

    return curParticle;
  }
};

void testNBody(Particle* const particles, int nParticles, int steps, int tile_width, bool output)
{
  DArray<Particle> P(nParticles, particles);
  DArray<Particle> oldP(nParticles, particles, Distribution::COPY); // copy distributed

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

} // namespace nbody
} // namespace examples
} // namespace msl

int main(int argc, char** argv)
{
  msl::initSkeletons(argc, argv);
  msl::Rng<float> rng_init{};  // vorher: msl::Rng<float, 0, 1> rng_init{};  HK 28.4.2020

  int nParticles = 64;  int nSteps = 5; int nRuns = 1;
  int nGPUs = 2; int tile_width = 8;
  bool output = 0; bool noWarmup = 1;
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
    output = 1; noWarmup = 1;
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
  msl::examples::nbody::Particle* particles = new msl::examples::nbody::Particle[nParticles];
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







