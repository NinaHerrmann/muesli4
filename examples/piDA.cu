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

void writebpm(int* picture, int w, int h){
  FILE *f;
  unsigned char *img = NULL;
  int filesize = 54 + 3*w*h;  //w is image width, h is image height

  img = (unsigned char *)malloc(3*w*h);
  int pixel; int idx;

  for(int i=0; i<h; i++){  
    for(int j=0; j<w; j++){
        idx = i*w+j;
        pixel = picture[idx];
        img[idx*3+2] = (unsigned char)(pixel       & 255); // red
        img[idx*3+1] = (unsigned char)((pixel>>8)  & 255); // green
        img[idx*3+0] = (unsigned char)((pixel>>16) & 255); // blue
    }
  }

  unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
  unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
  unsigned char bmppad[3] = {0,0,0};

  bmpfileheader[ 2] = (unsigned char)(filesize    );
  bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
  bmpfileheader[ 4] = (unsigned char)(filesize>>16);
  bmpfileheader[ 5] = (unsigned char)(filesize>>24);

  bmpinfoheader[ 4] = (unsigned char)(       w    );
  bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
  bmpinfoheader[ 6] = (unsigned char)(       w>>16);
  bmpinfoheader[ 7] = (unsigned char)(       w>>24);
  bmpinfoheader[ 8] = (unsigned char)(       h    );
  bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
  bmpinfoheader[10] = (unsigned char)(       h>>16);
  bmpinfoheader[11] = (unsigned char)(       h>>24);

  f = fopen("mandelbrot.bmp","wb");
  fwrite(bmpfileheader,1,14,f);
  fwrite(bmpinfoheader,1,40,f);
  for(int i=0; i<h; i++){
    fwrite(img+(w*(h-i-1)*3),3,w,f);
    fwrite(bmppad,1,(4-(w*3)%4)%4,f);
  }

  free(img);
  fclose(f);
}

struct Mandel: public Functor2<int,int,int>{
   double re_min, re_max, im_min, im_max, maximum;
   int xpixels, ypixels, max_iter, max_colour;

   Mandel(double rmin, double rmax, double imin, double imax, double mx,
          int xp, int yp, int mxiter, int mxcol):
     re_min(rmin), re_max(rmax), im_min(imin), im_max(imax), maximum(mx),
     xpixels(xp), ypixels(yp), max_iter(mxiter), max_colour(mxcol) {}

   MSL_USERFUNC
   int operator()(int idx, int dummy) const {
     int row = idx / xpixels;
     int col = idx % xpixels; 
     double x = im_min + (im_max-im_min)* col/ypixels;
     double y = re_min + (re_max-re_min)* row/xpixels;
     double c_re = x;
     double c_im = y;
     int iterations = 0;
     while ((x*x + y*y <= maximum) && (++iterations < max_iter)){
       x = x*x - y*y + c_re;
       y = 2.0*x*y + c_im;
     }
     return max_colour * iterations / max_iter; 
   }
};
  
void mandelbrot(int dim) {
   double re_min = -2.0;
   double re_max = 1.0, im_min = -1.4, im_max = 1.4, maximum = 4.0;
   int xpixels = sqrt(dim) + 0.1, 
       ypixels = xpixels, max_iter = 100, max_colour=1<<24; // 24 bit colors
   if (xpixels * ypixels != dim) throw 99; // no quadratic area
   Mandel mandel(re_min, re_max, im_min, im_max, maximum,
                 xpixels, ypixels, max_iter, max_colour);
   DA<int> picture(dim, 0);  // #pixels, (dummy) initial value 
   picture.mapIndexInPlace(mandel);

   picture.show("Mandelbrot");

   // gather and print on root
   int n = xpixels * ypixels;
   int* pic =  new int[n];
   picture.gather(pic);
   std::ostringstream s;
   if (msl::isRootProcess()) {
    s << "[";
    for (int i = 0; i < n - 1; i++) {
      s << pic[i];
      s << " ";
    }
    s << pic[n - 1] << "]" << std::endl;
    s << std::endl;
  }

   writebpm(pic,xpixels,ypixels);
   return;
}
}} // close namespaces

int main(int argc, char** argv){
  msl::initSkeletons(argc, argv);
  msl::setNumRuns(1);
  msl::setNumGpus(atoi(argv[2]));
  msl::mandelbrot::mandelbrot(atoi(argv[1]));
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}

