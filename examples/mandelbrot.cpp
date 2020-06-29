/*
 * mandelbrot.cpp
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
#include <cstdlib>

#include "dmatrix.h"
#include "functors.h"

namespace msl {
namespace examples {
namespace mandelbrot {

struct Pixel
{
  unsigned char r, g, b;

  Pixel()
    : r(0), g(0), b(0)
  {}
};

std::ostream& operator<< (std::ostream& out, Pixel p)
{
  out << p.r << p.g << p.b;
  return out;
}

int writePPM(DMatrix<Pixel>& out_img , const std::string& filename)
{
  std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);

  // Gather full image
  int rows = out_img.getRows();
  int cols = out_img.getCols();
  Pixel** img = new Pixel*[rows];
  for (int i = 0; i < rows; ++i)
    img[i] = new Pixel[cols];
  out_img.gather(img);

  // Write image
  ofs << "P6\n" << cols << " " << rows << "\n255\n";

  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      ofs << static_cast<unsigned char>(img[y][x].r)
          << static_cast<unsigned char>(img[y][x].g)
          << static_cast<unsigned char>(img[y][x].b);
    }
  }

  if (ofs.fail()) {
    std::cout << "Cannot write file " << filename << "!" << std::endl;
    return 1;
  }

  return 0;
}

struct Iterate : public MMapIndexFunctor<Pixel, Pixel>
{
  int maxIters, zoom;
  float maxAbs = 4.0f, center_x, center_y;
  float dx, dy, l, r, t, b;

  Iterate(int mIters, float cx, float cy, int z, int rows, int cols)
    : maxIters(mIters), zoom(z), center_x(cx), center_y(cy)
  {
    l = center_x - ((float) cols / (zoom * 2));
    r = center_x + ((float) cols / (zoom * 2));
    t = center_y - ((float) rows / (zoom * 2));
    b = center_y + ((float) rows / (zoom * 2));

    dx = (r - l) / cols;
    dy = (b - t) / rows;
  }

  MSL_USERFUNC
  Pixel operator() (int row, int col, Pixel p) const
  {
    int iters = 0;

    float real = l + col*dx; float tmpReal = real;
    float imag = t + row*dy; float tmpImag = imag;
    float nextReal;

    while ((real * real) + (imag * imag) <= maxAbs && iters < maxIters) {
      nextReal = (real * real) - (imag * imag) + tmpReal;
      imag = (2 * real * imag) + tmpImag;
      real = nextReal;
      ++iters;
    }

    if (iters < maxIters) {
      p.r = ((iters & 63) << 1);
      p.g = ((iters << 3) & 255);
      p.b = ((iters >> 8) & 255);
    }
    return p;
  }
};

void testMandelbrot(int rows, int cols, int maxIters, float center_x, float center_y, int zoom, bool output)
{
  Pixel p;
  DMatrix<Pixel> mandelbrot(rows, cols, Muesli::num_total_procs, 1, p);

  Iterate iterate(maxIters, center_x, center_y, zoom, rows, cols);
  mandelbrot.mapIndexInPlace(iterate);

  if (output) {
    writePPM(mandelbrot, "mandelbrot.ppm");
  }
}

}
}
}

int main(int argc, char** argv)
{
  using namespace msl::examples::mandelbrot;
  msl::initSkeletons(argc, argv);

  int rows = 1000, cols = 1000, nRuns = 1, nGPUs = 1;
  int maxIters = 1000, zoom = 800;//27615;
  bool output = 1, warmup = 0;
  if (argc < 7) {
    if (msl::isRootProcess()) {
      std::cout << std::endl << std::endl;
      std::cout << "Usage: " << argv[0] << " #rows #cols #maxIters #zoom #nRuns #nGPUs" << std::endl;
      std::cout << "Default values: rows = " << rows
                << ", cols = " << cols
                << ", maxIters = " << maxIters
                << ", zoom = " << zoom
                << ", nRuns = " << nRuns
                << ", nGPUs = " << nGPUs
                << std::endl << std::endl << std::endl;
    }
  } else {
    rows = atoi(argv[1]);
    cols = atoi(argv[2]);
    maxIters = atoi(argv[3]);
    zoom = atoi(argv[4]);
    nRuns = atoi(argv[5]);
    nGPUs = atoi(argv[6]);
    output = false;
    // warmup only for GPUs
#ifdef __CUDACC__
    warmup = 1;
#endif
  }

  msl::setNumRuns(nRuns);
  msl::setNumGpus(nGPUs);

  const float center_x = -0.73f;
  const float center_y = 0.0f;

  // warmup
  if (warmup) {
    msl::Timer tw("Warmup");
    testMandelbrot(rows, cols, maxIters, center_x, center_y, zoom, false);
    msl::printv("Warmup: %fs\n", tw.stop());
  }

  msl::startTiming();
  for (int run = 0; run < msl::Muesli::num_runs; run++) {
    testMandelbrot(rows, cols, maxIters, center_x, center_y, zoom, output);
    msl::splitTime(run);
  }
  msl::stopTiming();

  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}

