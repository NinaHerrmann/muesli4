/*
 * canny.cpp
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
#include <cmath>
#include "muesli.h"
#include "dmatrix.h"

namespace msl {
namespace examples  {
namespace canny {

#ifdef __CUDACC__
#define SQRT(a)     sqrtf(a)
#define FMOD(a, b)  fmodf(a, b)
#define ATAN2(a, b) atan2f(a, b)
#define ROUND(a)    lroundf(a)
#define ABS(a)      fabsf(a)
#else
#define SQRT(a)     std::sqrt(a)
#define FMOD(a, b)  std::fmod(a, b)
#define ATAN2(a, b) std::atan2(a, b)
#define ROUND(a)    std::round(a)
#define ABS(a)      std::abs(a)
#endif

#define TW 16
//int rows, cols;

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

// Grayscale pixel class.
struct GS_Pixel {
  int intensity;
  float direction;

  MSL_USERFUNC
  GS_Pixel()
    : intensity(0), direction(0.0f)
  {}

  MSL_USERFUNC
  GS_Pixel(int intens, float dir)
    : intensity(intens), direction(dir)
  {}

  MSL_USERFUNC
  bool operator>(const GS_Pixel& p)
  {
    return intensity > p.intensity;
  }

  MSL_USERFUNC
  bool operator<(const GS_Pixel& p)
  {
    return intensity < p.intensity;
  }
};

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

int readPGMA(const std::string& filename, int& rows, int& cols, int& max_color)
{
  std::ifstream ifs(filename);
  if (!ifs) {
    std::cout << "Error: Cannot open image file " << filename << "!" << std::endl;
    return 1;
  }

  // Read magic number.
  std::string magic;
  getline(ifs, magic);
  if (magic.compare("P2")) { // P2 is magic number for pgm ascii format.
    std::cout << "Error: Image not in PGM ascii format!" << std::endl;
    return 1;
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
  input_image_int = new int[rows*cols];
  int i = 0;
  while (getline(ifs, inputLine)) {
    std::stringstream(inputLine) >> input_image_int[i++];
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

int writePGM(const std::string& filename, DMatrix<GS_Pixel>& out_image, int rows, int cols, int max_color)
{
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    std::cout << "Error: Cannot open image file " << filename << "!" << std::endl;
    return 1;
  }

  // Gather full image
  GS_Pixel** img = new GS_Pixel*[rows];
  for (int i = 0; i < rows; i++)
    img[i] = new GS_Pixel[cols];
  out_image.gather(img);

  // Write image header
  ofs << "P5\n" << cols << " " << rows << " " << std::endl << max_color << std::endl;

  // Write image
  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < cols; y++) {
      unsigned char intensity = static_cast<unsigned char> (img[x][y].intensity);
      ofs << intensity;
    }
  }

  if (ofs.fail()) {
    std::cout << "Cannot write file " << filename << "!" << std::endl;
    return 1;
  }

  return 0;
}

// Initialization functor.
// Returns grayscale pixel at (row, col)
class Init
{
  int cols;
public:

  Init(int c)
    : cols(c)
  {}

  int operator() (int row, int col) const
  {
    if (ascii) return input_image_int[row*cols+col];
    else return input_image_char[row*cols+col];
  }
};

// Gaussian blur functor.
class Gaussian : public MMapStencilFunctor<int, int>
{
  int offset;
public:

  Gaussian()
	: offset(2)
  {
    this->setTileWidth(TW);
    this->setStencilSize(offset);
  }

  MSL_USERFUNC
  int operator() (int row, int col, const PLMatrix<int>& input) const
  {
    // Gaussian kernel. See http://www.cse.iitd.ernet.in/~pkalra/csl783/canny.pdf.
    const int kw = 5;
    const int g_kernel[kw*kw] = { 2, 4,   5,  4, 2,
                                  4, 9,  12,  9, 4,
                                  5, 12, 15, 12, 5,
                                  4, 9,  12,  9, 4,
                                  2, 4,   5,  4, 2 };
//    const int g_kernel[kw*kw];
    float weight = 159.0f;

    // Convolution
    int sum = 0;
    for (int r = 0; r < kw; r++) {
      for (int c = 0; c < kw; c++) {
        sum += input.get(row+r-offset, col+c-offset) * g_kernel[r*kw+c];
      }
    }

    //float ret = ROUND((float)sum/weight);
    return (float)sum/weight;
  }
};

// Sobel filter functor.
class Sobel : public MMapStencilFunctor<int, GS_Pixel>
{
  const float PI_F=3.14159265358979f;
  int offset;
public:

  Sobel()
	  : offset(1)
  {
    this->setTileWidth(TW);
    this->setStencilSize(offset);
  }

  MSL_USERFUNC
  GS_Pixel operator() (int row, int col, const PLMatrix<int>& input) const
  {
    // Sobel kernels.
    const int sobel_kernel_x[3*3] = { -1, 0, 1,
                                      -2, 0, 2,
                                      -1, 0, 1 };
    const int sobel_kernel_y[3*3] = {  1,  2,  1,
                                       0,  0,  0,
                                      -1, -2, -1 };

    // Convolution.
    int gx = 0; int gy = 0;
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        gx += input.get(row+r-offset, col+c-offset) *
                          sobel_kernel_x[r*3+c];
        gy += input.get(row+r-offset, col+c-offset) *
                          sobel_kernel_y[r*3+c];
      }
    }
    int mag = (int) SQRT(gx*gx + gy*gy);
    float direction = (float) (FMOD((float)ATAN2(ABS(gy), ABS(gx)) + PI_F, PI_F) / PI_F) * 8;

    return GS_Pixel(mag, direction);
  }
};

// Non maximum suppression functor.
class NMS : public MMapStencilFunctor<GS_Pixel, int>
{
public:

  NMS()
  {
    this->setTileWidth(TW);
    this->setStencilSize(1);
  }

  MSL_USERFUNC
  int operator() (int row, int col, const PLMatrix<GS_Pixel>& G) const
  {
    GS_Pixel p = G.get(row, col);
    float dir = p.direction;
    int nms = 0;
    if (((dir <= 1.0f || dir > 7.0f) &&
                   p > G.get(row, col-1) && p > G.get(row, col+1)) ||     // 0 deg
           ((dir > 1.0f && dir <= 3.0f) &&
                   p > G.get(row-1, col+1) && p > G.get(row+1, col-1)) || // 45 deg
           ((dir > 3.0f && dir <= 5.0f) &&
                   p > G.get(row-1, col) && p > G.get(row+1, col)) ||     // 90 deg
           ((dir > 5.0f && dir <= 7.0f) &&
                   p > G.get(row-1, col-1) && p > G.get(row+1, col+1)))   // 135 deg
    {
      nms = p.intensity;
    }
    return nms;
  }
};

// Double Threshold functor.
class Threshold : public MMapFunctor<int, int>
{
  int low, high;
public:

  Threshold(int l, int h)
    : low(l), high(h)
  {}

  MSL_USERFUNC
  int operator() (int value) const
  {
    return value < low ? 0 : value > high ? 255 : value;
  }
};

class ThreshHyst : public MMapStencilFunctor<int, int>
{
  int low, high;
public:

  ThreshHyst(int l, int h)
    : low(l), high(h)
  {
    this->setTileWidth(TW);
    this->setStencilSize(1);
  }

  MSL_USERFUNC
  int operator() (int row, int col, const PLMatrix<int>& input) const
  {
    int val = input.get(row, col);
    if (val < low)
      return 0;
    else if (val > high)
      return 255;
    else {
      if (
          (input.get(row+1, col) > high)   || // west
          (input.get(row-1, col) > high)   || // east
          (input.get(row, col+1) > high)   || // south
          (input.get(row, col-1) > high)   || // north
          (input.get(row-1, col-1) > high) || // north east
          (input.get(row+1, col-1) > high) || // north west
          (input.get(row-1, col+1) > high) || // south east
          (input.get(row+1, col+1) > high)    // south west
         )
        return 255;
    }
    return val;
  }
};

void testCanny(std::string in_file, std::string out_file, bool output)
{
  int rows, cols, max_color;
  double canny_time = 0.0, t_upload = 0.0, t_padding =  0.0, t_kernel = 0.0;

  // Read image
  readPGM(in_file, rows, cols, max_color);
  msl::startTiming();
  for (int run = 0; run < Muesli::num_runs; ++run) {
    // Create distributed matrix to store the grey scale image.
    Init init(cols);
    DMatrix<int> gs_image(rows, cols, Muesli::num_local_procs, 1, init);
    //writePGM("original.pgm", gs_image, rows, cols, max_color);

    double t = MPI_Wtime();
    // Gaussian blur
    Gaussian g;
    gs_image.mapStencilInPlace(g, 0);
    //writePGM("afterGaussian.pgm", gs_image, rows, cols, max_color);

    // Sobel filter
    Sobel s;
    DMatrix<GS_Pixel> sobel_image = gs_image.mapStencil<GS_Pixel, Sobel>(s, 0);
    //writePGM("afterSobel.pgm", sobel_image, rows, cols, max_color);

    // Non-maximum suppression
    NMS nms;
    DMatrix<int> nms_image = sobel_image.mapStencil<int, NMS>(nms, GS_Pixel());
    //writePGM("afterNMS.pgm", nms_image, rows, cols, max_color);

    ThreshHyst th(20, 200);
    nms_image.mapStencilInPlace(th, 0);

    if (output && msl::isRootProcess())
      writePGM(out_file, nms_image, rows, cols, max_color);

    // timing
    canny_time += MPI_Wtime() - t;
    canny_time = getAvg(canny_time);
    t_upload += gs_image.getStencilTimes()[0] + sobel_image.getStencilTimes()[0] + nms_image.getStencilTimes()[0];
    t_padding +=  gs_image.getStencilTimes()[1] + sobel_image.getStencilTimes()[1] + nms_image.getStencilTimes()[1];
    t_kernel += gs_image.getStencilTimes()[2] + sobel_image.getStencilTimes()[2] + nms_image.getStencilTimes()[2];
    msl::splitTime(run);
  }
  msl::stopTiming();
  if (msl::isRootProcess()) {
    std::cout << "Canny time: " << canny_time/Muesli::num_runs << std::endl
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

  std::string in_file, out_file;
  bool output = false;
  if (argc == 4) {
    in_file = argv[1];
    size_t pos = in_file.find(".");
    out_file = in_file;
    out_file.insert(pos, "_edges");
    msl::setNumGpus(atoi(argv[2]));
    msl::setNumRuns(atoi(argv[3]));
  } else {
    in_file = "lena.pgm";
    out_file = "lena_edges.pgm";
    output = true;
  }

  msl::examples::canny::testCanny(in_file, out_file, output);

  if (msl::examples::canny::ascii)
    delete[] msl::examples::canny::input_image_int;
  else
    delete[] msl::examples::canny::input_image_char;

  msl::terminateSkeletons();
  return 0;
}

