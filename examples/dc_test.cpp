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

int CHECK = 0;
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
        class Mult5 : public Functor5<int, int, int, int, int, int> {
        private: int y;
        public:
            Mult5(int factor):
            y(factor){}

            MSL_USERFUNC int operator() (int i, int j, int l, int Ai, int Bi) const {return (i * j * l * Ai * Bi * y);}
        };

        class Sum : public Functor2<int, int, int>{
        public: MSL_USERFUNC int operator() (int x, int y) const {return x+y;}
        };

        class Sum3 : public Functor3<int, int, int, int>{
        public: MSL_USERFUNC int operator() (int i, int j, int x) const {return i+j+x;}
        };


        class Sum4 : public Functor4<int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int i, int j, int x, int y) const {return i+j+x+y;}
        };

        class Sum5 : public Functor5<int, int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int i, int j, int x, int y, int l) const {return i+j+x+y+l;}
        };

        class CopyCond : public Functor4<int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int x, int v1, int v2, int y) const
                   {if ((v1 * v2) % 2 == 0) return x; else return y;}
        };


        class Proj1 : public Functor4<int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int x, int v1, int v2, int y) const {return x;}
        };


        class Proj2 : public Functor4<int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int x, int v1, int v2, int y) const {return v1;}
        };

        class Proj4 : public Functor4<int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int x, int v1, int v2, int y) const {return y;}
        };

        void dc_test(int dim, std::string nextfile, int reps) {
            if (msl::isRootProcess()) {
                //printf("Starting dc_test...\n");
            }

            // ************* Init *********************** //
            double fill_time = 0.0, const_time = 0.0, map0_time =  0.0, map1_time =  0.0, map2_time =  0.0, map3_time =  0.0, zip0_time =  0.0, zip1_time =  0.0, zip2_time =  0.0, zip3_time =  0.0, fold0_time = 0.0, fold1_time = 0.0;
            double t = MPI_Wtime();
            DC<int> a(dim,dim,dim);
            a.fill(2);
            // TODO does not work for all sizes.
            int elements = dim * dim * dim;
            if(CHECK){
                int * fillResult = a.gather();
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
            DC<int> b(dim, dim, dim, 3);
            if(CHECK){
                int * constructorResult = b.gather();
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

            Produkt pr;
            Mult4 mul4(3);
            Mult5 mul5(3);
            Mult mult(3);
            Sum sum;
            Sum3 sum3;
            Sum4 sum4;
            Sum5 sum5;
            int * mapResults = new int [dim*dim*dim];
            DC<int> map_dest(dim,dim,dim, 5);
	    for (int i = 0; i<reps; i++) {
                t = MPI_Wtime();
                map_dest = b.map(mult);
                int * mapResults = map_dest.gather();
                if(CHECK){
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
                map0_time += MPI_Wtime() - t;
            }

            b.fill(2);
            for (int i = 0; i<reps; i++) {
                t = MPI_Wtime();
                b.mapInPlace(mult);
             //   mapResults = b.gather();
                map1_time += MPI_Wtime() - t;
            }
            if(CHECK){
                if (msl::isRootProcess()) {
                    for (int i = 0; i < elements; i++) {
                        if (mapResults[i] != 6){
                            printf("mapInPlace \t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, mapResults[i], 6);
                            break;
                        }
                        if (i == (elements)-1) {
                            printf("mapInPlace \t\t \xE2\x9C\x93\n");
                        }
                    }
                }
            }

            DC<int> mapIndex(dim, dim, dim, 6);
            for (int i = 0; i<reps; i++) {
                t = MPI_Wtime();
                b = mapIndex.mapIndex(sum4);
             //   mapResults = b.gather();
                map2_time += MPI_Wtime() - t;
            }
            if(CHECK){
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


            b.fill(3);
            for (int i = 0; i<reps; i++) {
                t = MPI_Wtime();
                b.mapIndexInPlace(sum4);
             //   mapResults = b.gather();
                map3_time += MPI_Wtime() - t;
            }

            if(CHECK){
                int *mapIndexInPlace_comp = new int[elements];
                for (int j = 0; j < elements; j++) {
                    int depth = int(j/(dim * dim));
                    mapIndexInPlace_comp[j] = depth + int(j-(depth * dim * dim))/dim + (j%dim) + 3;
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
            // ************* Fold *********************** //
            b.fill(3);
            b.mapIndexInPlace(sum4);
            int result = 0;
            for (int i = 0; i<reps; i++) {
                t = MPI_Wtime();
                result = b.fold(sum, true);
                fold0_time += MPI_Wtime() - t;
            }
            if(CHECK){
                int compresult = 0;
                int *fold_comp = new int[elements];
                for (int j = 0; j < elements; j++) {
                    int depth = int(j/(dim*dim));
                    fold_comp[j] = depth + int(j-(depth * dim*dim))/dim + (j%dim) + 3;
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
            // ************* Zip *********************** //
            b.fill(10);
            DC<int> c(dim, dim, dim, 3);
            c.fill(20);
            int * zipResults = new int [dim*dim*dim];
            DC<int> d (dim,dim,dim);
            for (int i = 0; i<reps; i++) {
                t = MPI_Wtime();
                d = b.zip(c,sum);
                //zipResults = d.gather();
                zip0_time += MPI_Wtime() - t;
            }

            if(CHECK){
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
            b.fill(10);
            c.fill(10);
            for (int i = 0; i<reps; i++) {
                t = MPI_Wtime();
                b.zipInPlace(c, sum);
                //zipResults = b.gather();
                zip1_time += MPI_Wtime() - t;
            }

            if(CHECK){
                if (msl::isRootProcess()) {
                    for (int i = 0; i < elements; i++){
                        if (zipResults[i] != 20){
                            printf("ZipInPlace \t\t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, zipResults[i], 20);
                            break;
                        }
                        if (i == (elements)-1) {
                            printf("ZipInPlace \t\t \xE2\x9C\x93\n");
                        }
                    }
                }
            }
            b.fill(7);
            c.fill(5);
            for (int i = 0; i<reps; i++) {
                t = MPI_Wtime();
                d = b.zipIndex(c, sum5);
                //zipResults = d.gather();
                zip2_time += MPI_Wtime() - t;
            }

            if(CHECK){
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

            b.fill(3);
            c.fill(2);
            for (int i = 0; i<reps; i++) {
                t = MPI_Wtime();
                b.zipIndexInPlace(c, mul5);
                //zipResults = b.gather();
                zip3_time += MPI_Wtime() - t;
            }

            if(CHECK){
                int *zipIndexInPlace_comp = new int[elements];
                for (int j = 0; j < elements; j++) {
                    int depth = int(j/(dim*dim));
                    zipIndexInPlace_comp[j] = 3 * depth * (int(j-(depth * dim*dim))/dim) * (j%dim) * 3 * 2;
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
            if (msl::isRootProcess()) {
                std::ofstream outputFile;
                outputFile.open(nextfile, std::ios_base::app);
                outputFile << "" + std::to_string(fill_time) + ";" + std::to_string(const_time) +";" + std::to_string(map0_time) + ";" +
                "" + std::to_string(map1_time) + ";" + std::to_string(map2_time) +";" + std::to_string(map3_time) + ";" +
                "" + std::to_string(fold0_time) + ";" + std::to_string(zip0_time) +";" + std::to_string(zip1_time) + ";" +
                std::to_string(zip2_time) + ";" + std::to_string(zip3_time)+ ";\n";
                outputFile.close();
               // printf("Filltime ; consttime ; Map ; Mapinplace ; mapindex ; mapindexinplace, fold, zip, zipinplace, zipindex, zipindexinplace\n");
               printf("%f; %f; %f; %f; %f; %f; %f; %f; %f; %f; %f\n", fill_time, const_time,
                       map0_time, map1_time, map2_time, map3_time, fold0_time, zip0_time, zip1_time, zip2_time, zip3_time);}

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
  msl::setNumGpus(nGPUs);
  std::string nextfile;
  if (msl::isRootProcess()) {
      std::stringstream ss;
      ss << "without_gather_dc_" << std::to_string(msl::Muesli::num_total_procs) << "_" <<std::to_string(reps) << "_" << std::to_string(msl::Muesli::num_gpus);
      nextfile = ss.str();
      std::ofstream outputFile;
      outputFile.open(nextfile, std::ios_base::app);
      outputFile << "" + std::to_string(msl::Muesli::num_total_procs) + ";" + std::to_string(msl::Muesli::num_gpus) + ";"
      + std::to_string(dim) + ";" + std::to_string(msl::Muesli::cpu_fraction) + ";";
      outputFile.close();
      printf("%d; %d; %d;", msl::Muesli::num_total_procs,
        msl::Muesli::num_local_procs, msl::Muesli::num_gpus);
  }
  msl::test::dc_test(dim, nextfile, reps);
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}
