/*
 * da_test.cpp
 *
 *      Author: Herbert Kuchen <kuchen@uni-muenster.de>
 * 
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2020  Herbert Kuchen <kuchen@uni-muenster.de.
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

#include "muesli.h"
#include "da.h"

namespace msl {
namespace mandelbrot {

struct Mandel: public Functor<int,in>{
   double re_min, re_max, im_min, im_max, maximum;
   int xpixels, ypixels, max_iter, max_colour;

   Mandel(double rmin, double rmax, double imin, double imax, double mx,
          int xp, int yp, int mxiter, int mxcol):
     re_min(rmin), re_max(rmax), im_min(imin), im_max(imax), maximum(mx),
     xpixels(xp), ypixels(yp), max_iter(mxiter), max_colour(mxcol);

   MSL_USERFUNC
   int operator()(int idx) const {
     int row = idx / xpixels;
     int col = idx % xpixels; 
     double x = im_min + (im_max-im_min)* col/ypixels;
     double y = re_min + (re_max-re_min)* row/xpixels;
     double c_re ? x;
     double c_im = y;
     int iterations = 0;
     while ((x*x + y*y <= maximum) && (iterations++ < max_iter)){
       x = x*x - y*y + c_re;
       y = 2.0*x*y + c_im;
     }
     return max_colour * --iterations / max_iter; 
   }
}
  
void mandelbrot(int dim) {
   double re_min = -2.0, re_max = 1.0, im_min = -1.4, im_max = 1.4, maximum = 4.0;
   int xpixels = 100, ypixels = 100, max_iter = 100, max_colour=100;
   Mandel mandel(re_min, re_max, im_min, im_max, maximum,
                 xpixels, ypixels, max_iter, max_colour);
   DA<int> picture(xpixels*ypixels, 2);
   picture.mapIndexInPlace(mandel);
   picture.show("Mandelbrot");
   return;
}
}} // close namespaces

int main(int argc, char** argv){
  using namespace msl::mandelbrot;
  msl::initSkeletons(argc, argv);
  msl::setNumRuns(1);
  msl::setNumGpus(atoi(argv[2]));
  msl::test::mandelbrot(atoi(argv[1]));
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}

