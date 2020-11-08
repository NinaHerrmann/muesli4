/*
 * raytracer.cpp
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
#include <fstream>
#include <sstream>

#include "dmatrix.h"
#include "raytracer.h"
#include "functors.h"
#include "rng.h"

namespace msl {

namespace examples {

namespace raytracer {

const size_t kNumPixelSamplesU = 4;
const size_t kNumPixelSamplesV = 4;
const size_t kNumLightSamplesU = 4;
const size_t kNumLightSamplesV = 4;

Pixel init(int i, int j) {
  return Pixel(0.0f, 0.0f, 0.0f, i, j);
}

MSL_USERFUNC
Color trace(const Ray& ray, ShapeSet* shapeset) {
  Rng<float> rng; // vorher:  Rng<float, 0, 1> rng; HK 28.4.2020
  Color result(0.0f, 0.0f, 0.0f);
  Color lightResult;
  Vector toLight;

  Intersection intersection(ray);
  if (!shapeset->intersect(intersection)) {
    return result;
  }

  //Find out what lights the intersected point can see
  Point position = intersection.position();
  for (size_t i = 0; i < shapeset->nLights; i++) {
    // Sample the light
    lightResult = Color(0.0f, 0.0f, 0.0f);
    for (size_t lsv = 0; lsv < kNumLightSamplesV; lsv++) {
      for (size_t lsu = 0; lsu < kNumLightSamplesU; lsu++) {
        Point lightPoint;
        Vector lightNormal;
        RectangleLight* rectLight = &shapeset->lights[i];

        rectLight->sampleSurface((lsu + rng()) / float(kNumLightSamplesU),
                                 (lsv + rng()) / float(kNumLightSamplesV),
                                 position, lightPoint, lightNormal);

        // Fire a shadow ray to make sure we can actually see that light position
        toLight = lightPoint - position;
        float lightDistance = toLight.normalize();
        Ray shadowRay(position, toLight, lightDistance);
        Intersection shadowIntersection(shadowRay);
        bool intersected = shapeset->intersect(shadowIntersection);

        if (!intersected) {
          if (intersection.m_rectLight == 0) {
            if (intersection.m_Lambert == 0) {
              lightResult += rectLight->emitted() * intersection.m_colorModifier
                  * intersection.m_Phong->shade(position, intersection.m_normal,
                                                ray.m_direction, toLight);
            } else {
              lightResult += rectLight->emitted() * intersection.m_colorModifier
                  * intersection.m_Lambert->shade(position,
                                                  intersection.m_normal,
                                                  ray.m_direction, toLight);
            }
          }
        }
        if (shadowIntersection.m_rectLight == rectLight) {
          lightResult += rectLight->emitted() * intersection.m_colorModifier
              * intersection.m_Emitter->shade(position, intersection.m_normal,
                                              ray.m_direction, toLight);
        }
      }
    }
    lightResult /= kNumLightSamplesU * kNumLightSamplesV;

    result += lightResult;
  }

  return result;
}

struct CastRay : public MMapFunctor<Pixel, Pixel> {
  LScene scene;
  int height, width;

  CastRay(Scene& s, int h, int w)
      : scene(s),
        height(h),
        width(w) {
    this->addArgument(&scene);
  }

  MSL_USERFUNC
  Pixel operator()(Pixel p) const {
    // create RNG
    Rng<float> rng;  // vorher: Rng<float, 0, 1> rng; HK 28.4.2020
    // For each sample in the pixel...
    Color pixelColor(0.0f, 0.0f, 0.0f);

    for (size_t vsi = 0; vsi < kNumPixelSamplesV; vsi++) {
      for (size_t usi = 0; usi < kNumPixelSamplesU; usi++) {
        // Calculate a stratified random position within the pixel
        // to hide aliasing.
        float yu = 1.0f
            - (p.ypos + (vsi + rng()) / float(kNumPixelSamplesV))
                / float(height);
        float xu = (p.xpos + (usi + rng()) / float(kNumPixelSamplesU))
            / float(width);
        // Find where this pixel sample hits in the scene
        Ray ray = makeCameraRay(60.0f, Point(0.0f, 5.0f, 15.0f),
                                Point(0.0f, 0.0f, 0.0f),
                                Point(0.0f, 1.0f, 0.0f), xu, yu);

        pixelColor += trace(ray, scene.getShapeSet());
      }
    }
    // Divide by the number of pixel samples
    pixelColor /= kNumPixelSamplesU * kNumPixelSamplesV;

    p.setColor(pixelColor);
    return p;
  }
};

void testRaytrace(int height, int width, bool output) {
  // The 'scene'
  int nSpheres = 100;
  int nLights = 10;
  Scene scene(nSpheres, nLights);

  // create image
  msl::DMatrix<msl::examples::raytracer::Pixel> img(
      height, width, msl::Muesli::num_total_procs, 1, &init);

  // create functor
  CastRay cr(scene, height, width);

  // render the scene
  img.mapInPlace(cr);

  // print result
  if (output) {
    printRGB(img);
  }
}

}  // namespace raytracer
}  // namespace examples
}  // namespace msl

int main(int argc, char **argv) {
  msl::initSkeletons(argc, argv);

  msl::Rng<float> rng_init{};  // vorher: msl::Rng<float, 0, 1> rng_init{}; HK 28.4.2020

  int imgWidth = 256;
  int imgHeight = 256;
  int nRuns = 1;
  int nGPUs = 2;
  bool output = 0;
  bool noWarmup = 1;
  int nThreads = 256;
  if (argc < 6) {
    if (msl::isRootProcess()) {
      std::cout << std::endl << std::endl;
      std::cout << "Usage: " << argv[0]
                << " imgWidth imgHeight nRuns nThreads\n";
      std::cout << "Default values: imgWidth = " << imgWidth << ", imgHeight = "
                << imgHeight << ", nRuns = " << nRuns << ", nGPUs = " << nGPUs
                << ". nThreads = " << nThreads << std::endl;
      std::cout << std::endl << std::endl;
    }
    output = 1;
    noWarmup = 1;
  } else {
    imgHeight = atoi(argv[1]);
    imgWidth = atoi(argv[2]);
    nRuns = (atoi(argv[3]));
    nGPUs = atoi(argv[4]);
    nThreads = atoi(argv[5]);
    // warmup only for GPUs
#ifdef __CUDACC__
    noWarmup = 0;
#endif
  }

  msl::setNumRuns(nRuns);
  msl::setThreadsPerBlock(nThreads);
  msl::setNumGpus(nGPUs);

  // warmup
  if (!noWarmup) {
    msl::Timer tw("Warmup");
    msl::examples::raytracer::testRaytrace(imgHeight, imgWidth, 0);
    msl::printv("Warmup: %fs\n", tw.stop());
  }

  msl::startTiming();
  for (int run = 0; run < msl::Muesli::num_runs; run++) {
    msl::examples::raytracer::testRaytrace(imgHeight, imgWidth, output);
    msl::splitTime(run);
  }
  msl::stopTiming();

  msl::terminateSkeletons();

  return 0;
}

