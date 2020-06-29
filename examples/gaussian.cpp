/*
 * gaussian.cpp
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
#include <string>
#define _USE_MATH_DEFINES
#include <cmath>
#include "muesli.h"
#include "dmatrix.h"

namespace msl {
namespace examples  {
namespace gaussian {

#ifdef __CUDACC__
#define POW(a, b)      powf(a, b)
#define EXP(a)      exp(a)
#else
#define POW(a, b)      std::pow(a, b)
#define EXP(a)      std::exp(a)
#endif

#define TW 16
int rows, cols;

// Store input image.
int* input_image_int;
char* input_image_char;
bool ascii = false;

double getAvg(double t)
{
  double sum;
  MPI_Allreduce(&t, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sum/Muesli::num_total_procs;
}

int readPGM(const std::string& filename, int& rows, int& cols, int& max_color)
{
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    std::cout << "Error: Cannot open image file " << filename << "!" << std::endl;
    return 1;
  }

  // Read magic number.
  std::string magic;
  getline(ifs, magic);
  if (magic.compare("P5")) { // P5 is magic number for pgm binary format.
    if (magic.compare("P2")) { // P2 is magic number for pgm ascii format.
      std::cout << "Error: Image not in PGM format!" << std::endl;
      return 1;
    }
    ascii = true;
  }

  // Skip comments
  std::string inputLine;
  while (true) {
    getline(ifs, inputLine);
    if (inputLine[0] != '#') break;
  }

  // Read image size and max color.
  std::stringstream(inputLine) >> cols >> rows;
  getline(ifs, inputLine);
  std::stringstream(inputLine) >> max_color;

  // Read image.
  if (ascii) {
    input_image_int = new int[rows*cols];
    int i = 0;
    while (getline(ifs, inputLine)) {
      std::stringstream(inputLine) >> input_image_int[i++];
    }
  } else {
    input_image_char = new char[rows*cols];
    ifs.read(input_image_char, rows*cols);
  }

  return 0;
}

int writePGM(const std::string& filename, DMatrix<int>& out_image, int rows, int cols, int max_color)
{
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    std::cout << "Error: Cannot open image file " << filename << "!" << std::endl;
    return 1;
  }

  // Gather full image
  int** img = new int*[rows];
  for (int i = 0; i < rows; i++)
    img[i] = new int[cols];
  out_image.gather(img);

  // Write image header
  ofs << "P5\n" << cols << " " << rows << " " << std::endl << max_color << std::endl;

  // Write image
  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < cols; y++) {
      unsigned char intensity = static_cast<unsigned char> (img[x][y]);
      ofs << intensity;
    }
  }

  if (ofs.fail()) {
    std::cout << "Cannot write file " << filename << "!" << std::endl;
    return 1;
  }

  return 0;
}

// Gaussian blur functor.
class Gaussian : public MMapStencilFunctor<int, int>
{
  int kw, offset;
public:

  Gaussian(int k)
  : kw(k), offset(kw/2)
  {
    this->setTileWidth(TW);
    this->setStencilSize(offset);
  }

  MSL_USERFUNC
  int operator() (int row, int col, const PLMatrix<int>& input) const
  {
    float weight = 1.0f;
    float sigma = 1;
    float mean = (float)kw/2;

    // Convolution
    int sum = 0;
    for (int r = 0; r < kw; ++r) {
      for (int c = 0; c < kw; ++c) {
        sum += input.get(row+r-offset, col+c-offset) *
               EXP(-0.5 * (POW((r-mean)/sigma, 2.0) + POW((c-mean)/sigma,2.0))) / (2 * M_PI * sigma * sigma);
      }
    }

    return (float)sum/weight;
  }
};

int init(int row, int col)
{
  if (ascii) return input_image_int[row*cols+col];
  else return input_image_char[row*cols+col];
}

void testGaussian(std::string in_file, std::string out_file, int kw, bool output)
{
  int max_color;
  double gauss_time = 0.0, t_upload = 0.0, t_padding =  0.0, t_kernel = 0.0;

  // Read image
  readPGM(in_file, rows, cols, max_color);
  msl::startTiming();
  for (int run = 0; run < Muesli::num_runs; ++run) {
    // Create distributed matrix to store the grey scale image.
    DMatrix<int> gs_image(rows, cols, Muesli::num_local_procs, 1, &init);
    //writePGM("original.pgm", gs_image, rows, cols, max_color);

    double t = MPI_Wtime();
    // Gaussian blur
    Gaussian g(kw);
    gs_image.mapStencilInPlace(g, 0);
    //writePGM("afterGaussian.pgm", gs_image, rows, cols, max_color);

    // timing
    gauss_time += MPI_Wtime() - t;
    gauss_time = getAvg(gauss_time);
    t_upload += gs_image.getStencilTimes()[0];
    t_padding +=  gs_image.getStencilTimes()[1];
    t_kernel += gs_image.getStencilTimes()[2];

    if (output && msl::isRootProcess())
      writePGM(out_file, gs_image, rows, cols, max_color);

    msl::splitTime(run);
  }
  msl::stopTiming();
  if (msl::isRootProcess()) {
    std::cout << "Gaussian time: " << gauss_time/Muesli::num_runs << std::endl
              << "Upload time: " << t_upload/Muesli::num_runs << std::endl
              << "Kernel time: " << t_kernel/Muesli::num_runs << std::endl
              << "Padding time " << t_padding/Muesli::num_runs << std::endl;
  }
}

}
}
}

int main(int argc, char** argv)
{
  msl::initSkeletons(argc, argv);

  std::string in_file, out_file; int kw = 10;
  bool output = false;
  if (argc == 5) {
    in_file = argv[1];
    size_t pos = in_file.find(".");
    out_file = in_file;
    out_file.insert(pos, "_gaussian");
    kw = atoi(argv[2]);
    msl::setNumGpus(atoi(argv[3]));
    msl::setNumRuns(atoi(argv[4]));
  } else {
    in_file = "lena.pgm";
    out_file = "lena_gaussian.pgm";
    output = true;
  }

  msl::examples::gaussian::testGaussian(in_file, out_file, kw, output);

  if (msl::examples::gaussian::ascii)
    delete[] msl::examples::gaussian::input_image_int;
  else
    delete[] msl::examples::gaussian::input_image_char;

  msl::terminateSkeletons();
  return 0;
}
