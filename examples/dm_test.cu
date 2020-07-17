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
#include "dm.h"

namespace msl {
namespace test {

class Square : public Functor<int, int> {
public: MSL_USERFUNC int operator() (int x) const {return x*x;}
};

void dm_test(int dim) {
  //printf("Starting dm_test...\n");
  DM<int> a(4,4, 2);
  /*a.show("a1");

  Square sq;
  a.mapInPlace(sq);
  a.show("a2");*/

  return;
}
}} // close namespaces

int main(int argc, char** argv){
  //printf("Starting Main...\n");
  msl::initSkeletons(argc, argv);
  msl::setNumRuns(1);
  msl::setNumGpus(atoi(argv[2]));
  msl::test::dm_test(16);
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}

