/*
 * dm_test.cpp
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
#include "dm.h"

int CHECK = 0;
int OUTPUT = 0;
namespace msl {
    namespace test {

        class Square : public Functor<int, int> {
          public: MSL_USERFUNC int operator() (int y) const {return y*y;}
        };

        class Mult : public Functor<int, int> {
        private: int y;
        public:
            Mult(int factor):
                y(factor){}

            MSL_USERFUNC int operator() (int x) const {return x*y;}
        };

        struct Produkt : public Functor3<int, int, int, int>{
            MSL_USERFUNC int operator()(int i, int j, int Ai) const {return (i * j * Ai);}
        };

        class Mult4 : public Functor4<int, int, int, int, int> {
        private: int y;
        public:
            Mult4(int factor):
            y(factor){}

            MSL_USERFUNC int operator() (int i, int j, int Ai, int Bi) const {return (i * j * Ai * Bi * y);}
        };

        class Sum : public Functor2<int, int, int>{
        public: MSL_USERFUNC int operator() (int x, int y) const {return x+y;}
        };

        class Sum3 : public Functor3<int, int, int, int>{
        public: MSL_USERFUNC int operator() (int i, int j, int x) const {return i+j+x;}
        };

        class Index : public Functor3<int, int, int, int>{
        private: int y;
        public:
            Index(int cols):
            y(cols){}
        MSL_USERFUNC int operator() (int i, int j, int x) const {return (i*y)+j;}
        };


        class Sum4 : public Functor4<int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int i, int j, int x, int y) const {return i+j+x+y;}
        };

        void dm_test(int dim, std::string nextfile, int reps) {
            // ************* Init *********************** //
            double fill_time = 0.0, const_time = 0.0, map0_time =  0.0, map1_time =  0.0, map2_time =  0.0, map3_time =  0.0, zip0_time =  0.0, zip1_time =  0.0, zip2_time =  0.0, zip3_time =  0.0, fold0_time = 0.0, fold1_time = 0.0;

            DM<int> a(dim, dim);
            double t = MPI_Wtime();
            a.fill(2);
            int elements = dim * dim;
            int * fillResult = a.gather();
            fill_time += MPI_Wtime() - t;

            if(CHECK && msl::isRootProcess()){
                for (int i = 0; i < elements; i++){
                    if (fillResult[i] != 2){
                        printf("Fill \t\t\t\t \xE2\x9C\x97 At Index At Index %d - Value %d No further checking.\n", i, fillResult[i]);
                        break;
                    }
                    if (i == (elements)-1) {
                        printf("Fill \t\t\t \xE2\x9C\x93\n");
                    }
                }
            }
            t = MPI_Wtime();
            DM<int> b(dim, dim, 5);
            const_time += MPI_Wtime() - t;
            int * constResult = b.gather();

            if(CHECK && msl::isRootProcess()){
                for (int i = 0; i < elements; i++){
                    if (constResult[i] != 5){
                        printf("Initialize+fill \xE2\x9C\x97 At Index %d - Value %d No further checking.\n", i, constResult[i]);
                        break;
                    }
                    if (i == (elements)-1) {
                        printf("Initialize+fill \t \xE2\x9C\x93\n");
                    }
                }
            }
            Produkt pr;
            Mult4 mul4(3);
            Mult mult(3);
            Sum sum;
            Sum3 sum3;
            Sum4 sum4;

            /*DM<int> map(dim, dim, 3);
            int * mapResults;
            t = MPI_Wtime();
            DM<int> map_dest(dim, dim);
            for (int i = 0; i<reps; i++) {
                map_dest = map.map(mult);
            }
            mapResults = map_dest.gather();
            map0_time += MPI_Wtime() - t;
            if(CHECK && msl::isRootProcess()){
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
            DM<int> mapInPlace(dim, dim, 2);
            t = MPI_Wtime();

            for (int i = 0; i<reps; i++) {
                mapInPlace.mapInPlace(mult);
            }
            int * manmapResults = new int[elements];
            for (int j = 0; j < elements; j++) {
                manmapResults[j] = 2;
            }
            for (int i = 0; i<reps; i++) {
                for (int j = 0; j < elements; j++) {
                    manmapResults[j] = manmapResults[j] * 3;
                }
            }
            mapResults = mapInPlace.gather();
            map1_time += MPI_Wtime() - t;
            if(CHECK && msl::isRootProcess()){
                for (int i = 0; i < elements; i++) {
                    if (mapResults[i] != manmapResults[i]){
                        printf("mapInPlace \t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, mapResults[i], 6);
                        break;
                    }
                    if (i == (elements)-1) {
                        printf("mapInPlace \t\t \xE2\x9C\x93\n");
                    }
                }
            }
            DM<int> mapIndex(dim, dim, 6);
            for (int i = 0; i<reps; i++) {
                DM<int> mapIndex_dest = mapIndex.mapIndex(sum3);
                mapResults = mapIndex_dest.gather();
            }
            int *mapIndex_comp = new int[elements];
            for (int j = 0; j < elements; j++) {
                mapIndex_comp[j] = 6;
            }
            for (int j = 0; j < elements; j++) {
                mapIndex_comp[j] = int(j/dim) + (j%dim) + mapIndex_comp[j];
            }

            if (CHECK && msl::isRootProcess()){
                for (int i = 0; i < elements; i++) {
                    if (mapResults[i] != mapIndex_comp[i]){
                        printf("mapIndex \t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, mapResults[i], mapIndex_comp[i]);
                        break;
                    }
                    if (i == (elements)-1) {
                        printf("mapIndex \t\t \xE2\x9C\x93\n");
                    }
                }
            }
            DM<int> amap(dim, dim, 2);
            int *amap_comp = new int[elements];

            for (int i = 0; i<reps; i++) {
                amap.mapIndexInPlace(pr);
            }
            mapResults = amap.gather();
            for (int j = 0; j < elements; j++) {
                amap_comp[j] = 2;
            }
            for (int i = 0; i<reps; i++) {
                for (int j = 0; j < elements; j++) {
                    amap_comp[j] = amap_comp[j] * int(j/dim) * (j%dim);
                }
            }
            if(CHECK && msl::isRootProcess()) {
                for (int i = 0; i < elements; i++){
                    if (mapResults[i] != amap_comp[i]){
                        printf("MapIndexInPlace \t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, mapResults[i], amap_comp[i]);
                        break;
                    }
                    if (i == (elements)-1) {
                        printf("MapIndexInPlace \t \xE2\x9C\x93\n");
                    }
                }
            }
            // ************* Fold *********************** //
            DM<int> fold(dim, dim, 2);
            fold.mapIndexInPlace(pr);
            int foldresult;
            for (int i = 0; i<reps; i++) {
                foldresult = fold.fold(sum, true);
            }
            int compfoldresult = 0;
            int *fold_comp2 = new int[elements];
            for (int j = 0; j < elements; j++) {
                fold_comp2[j] = 2 * int(j/dim) * (j%dim);
            }

            for (int j = 0; j < elements; j++) {
                compfoldresult += fold_comp2[j];
            }
            if(CHECK && msl::isRootProcess()) {
                if (compfoldresult == foldresult) {
                    printf("Fold2  \t\t\t \xE2\x9C\x93\n");
                } else {
                    printf("Fold2 \t\t\t\t \xE2\x9C\x97  \t\t parallel = %d seq = %d! \n", foldresult, compfoldresult);
                }
            }
           
            // ************* Zip *********************** //

            DM<int> zip(dim, dim);
            DM<int> zip_param(dim, dim);
            DM<int> zip_dest(dim, dim);
            zip.fill(10);
            int * zipResults = new int[elements];
            int * manzipResults = new int[elements];

            zip_param.fill(20);
            for (int j = 0; j<elements; j++) {
                manzipResults[j] = 20;
            }
            for (int j = 0; j<elements; j++) {
                manzipResults[j] = 10 + manzipResults[j];
            }


            for (int i = 0; i<reps; i++) {
                zip_dest = zip.zip(zip_param,sum);
            }
            zipResults = zip_dest.gather();
            if(CHECK && msl::isRootProcess()) {
                for (int i = 0; i < elements; i++){
                    if (manzipResults[i] != zipResults[i]){
                        printf("Zip \t\t\t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, zipResults[i], manzipResults[i]);
                        break;
                    }
                    if (i == (elements)-1) {
                        printf("Zip \t\t\t \xE2\x9C\x93\n");
                    }
                }
            }
            DM<int> zipIndex(dim, dim, 7);
            DM<int> zipIndex_param(dim, dim, 8);
            for (int i = 0; i<reps; i++) {
                DM<int> zipIndex_dest = zipIndex.zipIndex(zipIndex_param, sum4);
                zipResults = zipIndex_dest.gather();
            }
            int *zipIndex_comp = new int[elements];
            // Independent from reps?
            for (int j = 0; j < elements; j++) {
                zipIndex_comp[j] = int(j/dim) + (j%dim) + 7 + 8;
            }

            if(CHECK && msl::isRootProcess()) {
                for (int i = 0; i < elements; i++){
                    if (zipResults[i] != zipIndex_comp[i]){
                        printf("ZipIndex \t\t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, zipResults[i], zipIndex_comp[i]);
                        break;
                    }
                    if (i == (elements)-1) {
                        printf("ZipIndex \t\t \xE2\x9C\x93\n");
                    }
                }
            }
            DM<int> zipInPlace(dim, dim);
            DM<int> zipInPlace_dest(dim, dim);
            zipInPlace.fill(10);
            zipInPlace_dest.fill(20);
            for (int i = 0; i<reps; i++) {
                zipInPlace_dest.zipInPlace(zipInPlace, sum);
            }
            zipResults = zipInPlace_dest.gather();

            for (int j = 0; j<elements; j++) {
                manzipResults[j] = 20;
            }
            for (int i = 0; i<reps; i++) {
                for (int j = 0; j<elements; j++) {
                    manzipResults[j] = 10 + manzipResults[j];
                }
            }
            if(CHECK && msl::isRootProcess()) {
                for (int i = 0; i < elements; i++){
                    if (zipResults[i] != manzipResults[i]){
                        printf("ZipInPlace \t\t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, zipInPlace_dest.get(i), 20);
                        break;
                    }
                    if (i == (elements)-1) {
                        printf("ZipInPlace \t\t \xE2\x9C\x93\n");
                    }
                }
            }
            DM<int> zipIndexInPlace(dim, dim, 4);
            DM<int> zipIndexInPlace_param(dim, dim, 2);
            int *zipIndexInPlace_comp = new int[elements];
            for (int j = 0; j < elements; j++) {
                zipIndexInPlace_comp[j] = 4;
            }
            for (int i = 0; i<reps; i++) {
                for (int j = 0; j < elements; j++) {
                    zipIndexInPlace_comp[j] = zipIndexInPlace_comp[j] * int(j/dim) * int(j%dim) * 3 * 2;
                }
            }
            for (int i = 0; i<reps; i++) {
                zipIndexInPlace.zipIndexInPlace(zipIndexInPlace_param, mul4);
            }
            zipResults = zipIndexInPlace.gather();

            if(CHECK && msl::isRootProcess()) {
                for (int i = 0; i < elements; i++){
                    if (zipResults[i] != zipIndexInPlace_comp[i]){
                        printf("ZipIndexInPlace \t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, zipResults[i], zipIndexInPlace_comp[i]);
                        break;
                    }
                    if (i == (elements)-1) {
                        printf("ZipIndexInPlace \t \xE2\x9C\x93\n");
                    }
                }
            }
            if (msl::isRootProcess()) {
                printf("%f; %f; %f; %f; %f; %f; %f; %f; %f; %f; %f\n", fill_time, const_time,
                       map0_time, map1_time, map2_time, map3_time, fold0_time, zip0_time, zip1_time, zip2_time, zip3_time);
            }*/
          return;
        }
    }} // close namespaces

int main(int argc, char** argv){
  //printf("Starting Main...\n");
  msl::setNumRuns(1);
  msl::setNumGpus(2);
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
  msl::setNumGpus(nGPUs);
  std::string nextfile;
  if (msl::isRootProcess()) {
      printf("Starting Program %s with %d nodes %d cpus and %d gpus\n", msl::Muesli::program_name, msl::Muesli::num_total_procs,
             msl::Muesli::num_local_procs, msl::Muesli::num_gpus);
  }
  msl::test::dm_test(dim, nextfile, reps);
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}
