/*
 * dc_test.cpp
 *
 *      Author: Nina Hermann,
 *  	        Herbert Kuchen <kuchen@uni-muenster.de>
 * 
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2020  Herbert Kuchen <kuchen@uni-muenster.de>,
 *                 Nina Hermann
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
#include "dc.h"
#include "cuda_runtime.h"
#include "cuda.h"

int CHECK = 0;
int OUTPUT = 1;
namespace msl {
    namespace test {


        class Mult : public Functor<int, int> {
        private: int y;
        public:
            Mult(int factor):
                y(factor){}

            MSL_USERFUNC int operator() (int x) const {
                return y * x;
            }
        };
        class Fill : public Functor<int, int> {
        public:
            MSL_USERFUNC int operator() (int x) const {
                return 3;
            }
        };


        void dc_test(int dim, std::string nextfile, int reps, char * skeletons) {
            if (msl::isRootProcess()) {
                //printf("Starting dc_test...\n");
            }

            // ************* Init *********************** //
            int elements = dim * dim * dim;
            double fill_time = 0.0, const_time = 0.0, map0_time =  0.0, map1_time =  0.0, map2_time =  0.0, map3_time =  0.0, zip0_time =  0.0, zip1_time =  0.0, zip2_time =  0.0, zip3_time =  0.0, fold0_time = 0.0, fold1_time = 0.0;
            double t = MPI_Wtime();

            Mult mult(3);
            Fill fill;

            int * mapResults = new int [dim*dim*dim];
            int * manmapResults = new int [dim*dim*dim];
            DC<int> map_dest(dim, dim, dim, 5);
            DC<int> b(dim, dim, dim, 3);
            msl::syncStreams();

            b.mapInPlace(fill);
            if (strstr(skeletons, "map,") != NULL || strstr(skeletons, "all") != NULL) {

                t = MPI_Wtime();
                for (int i = 0; i<reps; i++) {
                    map_dest.map(mult, b);
                }
                mapResults = map_dest.gather();
                map0_time += MPI_Wtime() - t;
                if(CHECK && msl::isRootProcess()){
                    if (msl::isRootProcess()) {
                        for (int i = 0; i < elements; i++) {
                            if (mapResults[i] != 9){
                                printf("map \t\t\t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, mapResults[i], 9);
                                break;
                            }
                            if (i == (elements)-1) {
                                printf("map \t\t\t \xE2\x9C\x93\n");
                            }
                        }
                    }
                }
            }

            if (msl::isRootProcess()) {
                if (OUTPUT) {
                    std::ofstream outputFile;
                    outputFile.open(nextfile, std::ios_base::app);
                    outputFile << "" + std::to_string(fill_time) + ";" + std::to_string(const_time) +";" + std::to_string(map0_time) + ";" +
                    "" + std::to_string(map1_time) + ";" + std::to_string(map2_time) +";" + std::to_string(map3_time) + ";" +
                    "" + std::to_string(fold0_time) + ";" + std::to_string(zip0_time) +";" + std::to_string(zip1_time) + ";" +
                    std::to_string(zip2_time) + ";" + std::to_string(zip3_time)+ ";\n";
                    outputFile.close();
                }
               // printf("Filltime ; consttime ; Map ; Mapinplace ; mapindex ; mapindexinplace, fold, zip, zipinplace, zipindex, zipindexinplace\n");
               printf("%f;\n", map0_time);
            }

          return;
        }
    }} // close namespaces

int main(int argc, char** argv){
  //printf("Starting Main...\n");
  msl::setNumRuns(1);
  msl::initSkeletons(argc, argv);
  msl::Muesli::cpu_fraction = 0.2;
  int dim = 4;
  int nGPUs = 1;
  int reps = 1;
  if (argc >= 3) {
      dim = atoi(argv[1]);
      nGPUs = atoi(argv[2]);
  }
  if (argc >= 4) {
      msl::Muesli::cpu_fraction = atof(argv[3]);
      if (msl::Muesli::cpu_fraction > 1) {
          msl::Muesli::cpu_fraction = 1;
      }
  }
  if (argc >= 5) {
      CHECK = atoi(argv[4]);
  }
  if (argc >= 6) {
      reps = atoi(argv[5]);
  }
  char * skeletons;
  if (argc >= 7) {
      skeletons = argv[6];
  } else {
      skeletons = "all";
  }
  msl::setNumGpus(nGPUs);
  std::string nextfile;
  if (msl::isRootProcess()) {
      std::stringstream ss;
      ss << "dc2" << std::to_string(msl::Muesli::num_total_procs) << "_" << std::to_string(reps) << "_" << std::to_string(msl::Muesli::num_gpus) << std::to_string(msl::Muesli::cpu_fraction) ;
      nextfile = ss.str();
      if (OUTPUT) {
          std::ofstream outputFile;
          outputFile.open(nextfile, std::ios_base::app);
          outputFile << "" + std::to_string(msl::Muesli::num_total_procs) + ";" + std::to_string(msl::Muesli::num_gpus) + ";"
          + std::to_string(dim) + ";" + std::to_string(msl::Muesli::cpu_fraction) + ";";
          outputFile.close();
      }
      printf("%d; %d; %d; %d; %.2f\n", dim, msl::Muesli::num_total_procs,
             msl::Muesli::num_local_procs, msl::Muesli::num_gpus, msl::Muesli::cpu_fraction);
  }
  msl::test::dc_test(dim, nextfile, reps, skeletons);
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}
