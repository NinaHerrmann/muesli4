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
#include <cstring>

#include "muesli.h"
#include "dc.h"

int CHECK = 0;
int OUTPUT = 1;
namespace msl::test {

        class Mult : public Functor<int, int> {
        private: int y;
        public:
            Mult(int factor):
                y(factor){}

            MSL_USERFUNC int operator() (int x) const {
                return y * x;
            }
        };

        class Mult5 : public Functor5<int, int, int, int, int, int> {
        private: int y;
        public:
            Mult5(int factor):
            y(factor){}

            MSL_USERFUNC int operator() (int i, int j, int l, int Ai, int Bi) const {
                return i * j * l * Ai * Bi * y;
            }
        };

        class Sum : public Functor2<int, int, int>{
        public: MSL_USERFUNC int operator() (int x, int y) const {
            return y + x;
        }
        };


        class Sum4 : public Functor4<int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int i, int j, int x, int y) const {
            return i+j+x+y;}
        };

        class Sum5 : public Functor5<int, int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int i, int j, int x, int y, int l) const {
            return (i+j+x+y+l);}//(i+j+x) % 50;}
        };


        void dc_test(int dim, std::string nextfile, int reps, char * skeletons) {
            if (msl::isRootProcess()) {
                //printf("Starting dc_test...\n");
            }

            // ************* Init *********************** //
            int elements = dim * dim * dim;
            double fill_time = 0.0, const_time = 0.0, map0_time =  0.0, map1_time =  0.0, map2_time =  0.0, map3_time =  0.0, zip0_time =  0.0, zip1_time =  0.0, zip2_time =  0.0, zip3_time =  0.0, fold0_time = 0.0, fold1_time = 0.0;
            double t = MPI_Wtime();

            if (strstr(skeletons, "Fill,") != NULL || strstr(skeletons, "all") != NULL) {
                t = MPI_Wtime();
                DC<int> a(dim,dim,dim);
                a.fill(2);
                // TODO does not work for all sizes.
                int * fillResult = a.gather();

                if(CHECK && msl::isRootProcess()){
                    if (msl::isRootProcess()) {
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
                }

                fill_time += MPI_Wtime() - t;
                t = MPI_Wtime();
            }
            DC<int> b(dim, dim, dim, 3);

            if (strstr(skeletons, "Initfill,") != NULL || strstr(skeletons, "all") != NULL) {
                int * constructorResult = b.gather();

                if(CHECK && msl::isRootProcess()){
                    if (msl::isRootProcess()) {
                        for (int i = 0; i < elements; i++){
                            if (constructorResult[i] != 3){
                                printf("Initialize+fill \xE2\x9C\x97 At Index %d - Value %d No further checking.\n", i, constructorResult[i]);
                                break;
                            }
                            if (i == (elements)-1) {
                                printf("Initialize+fill \t \xE2\x9C\x93\n");
                            }
                        }
                    }
                }
                const_time += MPI_Wtime() - t;
            }
            Mult5 mul5(3);
            Mult mult(3);
            Sum sum;
            Sum4 sum4;
            Sum5 sum5;
            int * mapResults = new int [dim*dim*dim];
            int * manmapResults = new int [dim*dim*dim];
            DC<int> map_dest(dim, dim, dim, 5);
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

            b.fill(2);
	        t = MPI_Wtime();
	        if (strstr(skeletons, "mapInPlace,") != NULL || strstr(skeletons, "all") != NULL) {

	            for (int i = 0; i<reps; i++) {
	                b.mapInPlace(mult);
	            }
	            for (int j = 0; j < elements; j++) {
	                manmapResults[j] = 2;
	            }
	            for (int i = 0; i<reps; i++) {
	                for (int j = 0; j < elements; j++) {
	                    manmapResults[j] = manmapResults[j] * 3;
	                }
	            }
	            mapResults = b.gather();
	            map1_time += MPI_Wtime() - t;
	            if(CHECK && msl::isRootProcess()){
	                if (msl::isRootProcess()) {
	                    for (int j = 0; j < 30; j++){
	                        //printf("%d->%d=%d,", j, mapResults[j], manmapResults[j]);
	                    }
	                    for (int i = 0; i < elements; i++) {

	                        if (mapResults[i] != manmapResults[i]){
	                            printf("mapInPlace \t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, mapResults[i], manmapResults[i]);
	                            break;
	                        }
	                        if (i == (elements)-1) {
	                            printf("mapInPlace \t\t \xE2\x9C\x93\n");
	                        }
	                    }
	                }
	            }
	        }
	        if (strstr(skeletons, "mapIndex,") != NULL || strstr(skeletons, "all") != NULL) {

                DC<int> mapIndex(dim, dim, dim, 6);
                t = MPI_Wtime();
                for (int i = 0; i<reps; i++) {
                    b.mapIndex(sum4, mapIndex);
                }
                mapResults = b.gather();
                map2_time += MPI_Wtime() - t;

                if(CHECK && msl::isRootProcess()){
                    int *mapIndex_comp = new int[elements];
                    for (int j = 0; j < elements; j++) {
                        int depth = int(j/(dim*dim));
                        mapIndex_comp[j] = depth + int(j-(depth * dim*dim))/dim + (j%dim) + 6;
                    }
                    if (msl::isRootProcess()) {
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
                }
            }

            b.fill(3);
            t = MPI_Wtime();
            if (strstr(skeletons, "mapIndexInPlace,") != NULL || strstr(skeletons, "all") != NULL) {

                for (int i = 0; i<reps; i++) {
                    b.mapIndexInPlace(sum4);
                }
                mapResults = b.gather();
                map3_time += MPI_Wtime() - t;

                if(CHECK && msl::isRootProcess()){
                    int *mapIndexInPlace_comp = new int[elements];
                    for (int j = 0; j < elements; j++) {
                        mapIndexInPlace_comp[j] = 3;
                    }
                    for (int i = 0; i<reps; i++) {
                        for (int j = 0; j < elements; j++) {
                            int depth = int(j/(dim * dim));
                            mapIndexInPlace_comp[j] = depth + int(j-(depth * dim * dim))/dim + (j%dim) + mapIndexInPlace_comp[j];
                        }
                    }

                    if (msl::isRootProcess()) {
                        for (int i = 0; i < elements; i++){
                            if (mapResults[i] != mapIndexInPlace_comp[i]){
                                printf("MapIndexInPlace \t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, mapResults[i], mapIndexInPlace_comp[i]);
                                break;
                            }
                            if (i == (elements)-1) {
                                printf("MapIndexInPlace \t \xE2\x9C\x93\n");
                            }
                        }
                    }
                }
            }
            // ************* Fold *********************** //
            if (strstr(skeletons, "fold,") != NULL || strstr(skeletons, "all") != NULL) {

                b.fill(3);
                int result = 0;
                t = MPI_Wtime();
                for (int i = 0; i<reps; i++) {
                    result = b.fold(sum, true);
                }
                fold0_time += MPI_Wtime() - t;

                if(CHECK && msl::isRootProcess()){
                    int compresult = 0;
                    int *fold_comp = new int[elements];
                    for (int j = 0; j < elements; j++) {
                        int depth = int(j/(dim*dim));
                        fold_comp[j] = 3;//depth + int(j-(depth * dim * dim))/dim + (j%dim) + 3;
                    }
                    for (int j = 0; j < elements; j++) {
                        compresult += fold_comp[j];
                    }
                    if (msl::isRootProcess()) {
                        if (compresult == result) {
                            printf("Fold  \t\t\t \xE2\x9C\x93\n");
                        } else {
                            printf("Fold \t\t\t\t \xE2\x9C\x97  \t\t parallel = %d seq = %d! \n", result, compresult);
                        }
                    }
                }
            }
            // ************* Zip *********************** //
            DC<int> c(dim, dim, dim, 3);
            //DC<int>* d = new DC<int>(dim,dim,dim); delete d;
            DC<int> d(dim,dim,dim);
            int * zipResults = new int [dim*dim*dim];
            int * manzipResults = new int [dim*dim*dim];
            if (strstr(skeletons, "zip,") != NULL || strstr(skeletons, "all") != NULL) {
                b.fill(10);
                c.fill(20);

                t = MPI_Wtime();

                for (int i = 0; i<reps; i++) {
                    d.zip(c, b, sum);
                }
                zipResults = d.gather();
                zip0_time += MPI_Wtime() - t;

                if(CHECK && msl::isRootProcess()){
                    if (msl::isRootProcess()) {
                        for (int i = 0; i < elements; i++){
                            if (zipResults[i] != 30){
                                printf("Zip \t\t\t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, zipResults[i], 30);
                                break;
                            }
                            if (i == (elements)-1) {
                                printf("Zip \t\t\t \xE2\x9C\x93\n");
                            }
                        }
                    }
                }
            }
            if (strstr(skeletons, "zipInPlace,") != NULL || strstr(skeletons, "all") != NULL) {

                b.fill(10);
                c.fill(10);
                t = MPI_Wtime();

                for (int i = 0; i<reps; i++) {
                    b.zipInPlace(c, sum);
                }
                zipResults = b.gather();
                zip1_time += MPI_Wtime() - t;
                for (int j = 0; j < elements; j++){
                    manzipResults[j] = 10;
                }
                for (int i = 0; i<reps; i++) {
                    for (int j = 0; j < elements; j++){
                        manzipResults[j] = 10 + manzipResults[j];
                    }
                }
                if(CHECK && msl::isRootProcess()){
                    if (msl::isRootProcess()) {
                        for (int i = 0; i < elements; i++){
                            if (zipResults[i] != manzipResults[i]){
                                printf("ZipInPlace \t\t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, zipResults[i], 20);
                                break;
                            }
                            if (i == (elements)-1) {
                                printf("ZipInPlace \t\t \xE2\x9C\x93\n");
                            }
                        }
                    }
                }
            }
            if (strstr(skeletons, "zipIndex,") != NULL || strstr(skeletons, "all") != NULL) {

                b.fill(7);
                c.fill(5);
                t = MPI_Wtime();

                for (int i = 0; i<reps; i++) {
                    d.zipIndex(c, b, sum5);
                }
                zipResults = d.gather();
                zip2_time += MPI_Wtime() - t;

                if(CHECK && msl::isRootProcess()){
                    int *zipIndex_comp = new int[elements];
                    for (int j = 0; j < elements; j++) {
                        int depth = int(j/(dim*dim));
                        zipIndex_comp[j] = depth + int(j-(depth * dim*dim))/dim + (j%dim) + 7 + 5;
                    }
                    if (msl::isRootProcess()) {
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
                }
            }

            b.fill(3);
            c.fill(2);
            if (strstr(skeletons, "zipIndexInPlace,") != NULL || strstr(skeletons, "all") != NULL) {
                t = MPI_Wtime();

                for (int i = 0; i<reps; i++) {
                    b.zipIndexInPlace(c, mul5);
                }
                zipResults = b.gather();
                zip3_time += MPI_Wtime() - t;

                if(CHECK && msl::isRootProcess()){
                    int *zipIndexInPlace_comp = new int[elements];
                    for (int i = 0; i<elements; i++) {
                        zipIndexInPlace_comp[i] = 3;
                    }
                    for (int i = 0; i<reps; i++) {
                        for (int j = 0; j < elements; j++) {
                            int depth = int(j/(dim*dim));
                            zipIndexInPlace_comp[j] = zipIndexInPlace_comp[j] * depth * (int(j-(depth * dim*dim))/dim) * (j%dim) * 3 * 2;
                        }
                    }

                    if (msl::isRootProcess()) {
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
               printf("%f; %f; %f; %f; %f; %f; %f; %f; %f; %f; %f\n", fill_time, const_time,
                       map0_time, map1_time, map2_time, map3_time, fold0_time, zip0_time, zip1_time, zip2_time, zip3_time);}

          return;
        }
    } // close namespaces

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
