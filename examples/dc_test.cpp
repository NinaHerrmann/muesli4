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

        void dc_test(int dim) {
            printf("Starting dc_test...\n");

            // ************* Init *********************** //

            DC<int> a(5, 5, 5);
            a.fill(2);
            for (int i = 0; i < 5*5*5; i++){
                if (a.get(i) != 2){
                    printf("Fill \t\t\t\t \xE2\x9C\x97 At Index At Index %d - Value %d No further checking.\n", i, a.get(i));
                    break;
                }
                if (i == (5*5*5)-1) {
                    printf("Fill \t\t\t \xE2\x9C\x93\n");
                }
            }
            DC<int> b(5, 5, 5, 2);
            for (int i = 0; i < 5*5*5; i++){
                if (b.get(i) != 2){
                    printf("Initialize+fill \xE2\x9C\x97 At Index %d - Value %d No further checking.\n", i, b.get(i));
                    break;
                }
                if (i == (5*5*5)-1) {
                    printf("Initialize+fill \t \xE2\x9C\x93\n");
                }
            }
            Produkt pr;
            Mult4 mul4(3);
            Mult5 mul5(3);
            Mult mult(3);
            Sum sum;
            Sum3 sum3;
            Sum4 sum4;
            Sum5 sum5;
            int elements = 5 * 5 * 5;
            DC<int> map(5, 5, 5,3);
            DC<int> map_dest = map.map(mult);
            for (int i = 0; i < elements; i++) {
                if (map_dest.get(i) != 9){
                    printf("map \t\t\t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, map_dest.get(i), 9);
                    break;
                }
                if (i == (elements)-1) {
                    printf("map \t\t\t \xE2\x9C\x93\n");
                }
            }
            DC<int> mapInPlace(5, 5, 5, 2);
            mapInPlace.mapInPlace(mult);
            for (int i = 0; i < elements; i++) {
                if (mapInPlace.get(i) != 6){
                    printf("mapInPlace \t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, mapInPlace.get(i), 6);
                    break;
                }
                if (i == (elements)-1) {
                    printf("mapInPlace \t\t \xE2\x9C\x93\n");
                }
            }
            DC<int> mapIndex(5, 5, 5, 6);
            DC<int> mapIndex_dest = mapIndex.mapIndex(sum4);
            int *mapIndex_comp = new int[elements];
            for (int j = 0; j < elements; j++) {
                int depth = int(j/(5*5));
                mapIndex_comp[j] = depth + int(j-(depth * 5*5))/5 + (j%5) + 6;
            }
            for (int i = 0; i < elements; i++) {
                if (mapIndex_dest.get(i) != mapIndex_comp[i]){
                    printf("mapIndex \t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, mapIndex_dest.get(i), mapIndex_comp[i]);
                    break;
                }
                if (i == (elements)-1) {
                    printf("mapIndex \t\t \xE2\x9C\x93\n");
                }
            }

            DC<int> mapIndexInPlace(5, 5, 5, 3);
            int *mapIndexInPlace_comp = new int[elements];
            for (int j = 0; j < elements; j++) {
                int depth = int(j/(5*5));
                mapIndexInPlace_comp[j] = depth + int(j-(depth * 5*5))/5 + (j%5) + 3;
            }
            mapIndexInPlace.mapIndexInPlace(sum4);
            for (int i = 0; i < elements; i++){
                if (mapIndexInPlace.get(i) != mapIndexInPlace_comp[i]){
                    printf("MapIndexInPlace \t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, mapIndexInPlace.get(i), mapIndexInPlace_comp[i]);
                    break;
                }
                if (i == (elements)-1) {
                    printf("MapIndexInPlace \t \xE2\x9C\x93\n");
                }
            }
            // ************* Fold *********************** //
            DC<int> fold(5, 5, 5, 3);
            DC<int> fold2(10, 10, 10, 5);
            fold.mapIndexInPlace(sum4);
            fold2.mapIndexInPlace(sum4);
            int result = fold.fold(sum, true);
            //int result2 = 0;
            int result2 = fold2.fold(sum, true);
            int compresult2 = 0;
            int *fold_comp2 = new int[10*10*10];
            for (int j = 0; j < 10*10*10; j++) {
                int depth = int(j/(10*10));
                fold_comp2[j] = depth + int(j-(depth * 10*10))/10 + (j%10) + 5;
            }
            for (int j = 0; j < 10*10*10; j++) {
                compresult2 += fold_comp2[j];
            }
            if (compresult2 == result2) {
                printf("Fold2  \t\t\t \xE2\x9C\x93\n");
            } else {
                printf("Fold2 \t\t\t\t \xE2\x9C\x97  \t\t parallel = %d seq = %d! \n", result2, compresult2);
            }
            int compresult = 0;
            int *fold_comp = new int[5*5*5];
            for (int j = 0; j < elements; j++) {
                int depth = int(j/(5*5));
                fold_comp[j] = depth + int(j-(depth * 5*5))/5 + (j%5) + 3;
            }
            for (int j = 0; j < elements; j++) {
                compresult += fold_comp[j];
            }
            if (compresult == result) {
                printf("Fold  \t\t\t \xE2\x9C\x93\n");
            } else {
                printf("Fold \t\t\t\t \xE2\x9C\x97  \t\t parallel = %d seq = %d! \n", result, compresult);
            }
            // ************* Zip *********************** //

            DC<int> zip(5, 5, 5, 3);
            DC<int> zip_param(5, 5, 5, 3);
            zip.fill(10);
            zip_param.fill(20);
            DC<int> zip_dest = zip.zip(zip_param,sum);
            for (int i = 0; i < elements; i++){
                if (zip_dest.get(i) != 30){
                    printf("Zip \t\t\t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, zip_dest.get(i), 30);
                    break;
                }
                if (i == (10*10)-1) {
                    printf("Zip \t\t\t \xE2\x9C\x93\n");
                }
            }
            DC<int> zipInPlace(5, 5, 5);
            DC<int> zipInPlace_dest(5, 5, 5);
            zipInPlace.fill(10);
            zipInPlace_dest.fill(10);
            zipInPlace_dest.zipInPlace(zipInPlace, sum);
            for (int i = 0; i < elements; i++){
                if (zipInPlace_dest.get(i) != 20){
                    printf("ZipInPlace \t\t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, zipInPlace_dest.get(i), 20);
                    break;
                }
                if (i == (10*10)-1) {
                    printf("ZipInPlace \t\t \xE2\x9C\x93\n");
                }
            }
            DC<int> zipIndex(5, 5, 5, 7);
            DC<int> zipIndex_param(5, 5, 5, 5);
            DC<int> zipIndex_dest = zipIndex.zipIndex(zipIndex_param, sum5);
            int *zipIndex_comp = new int[elements];
            for (int j = 0; j < elements; j++) {
                int depth = int(j/(5*5));
                zipIndex_comp[j] = depth + int(j-(depth * 5*5))/5 + (j%5) + 7 + 5;
            }
            for (int i = 0; i < elements; i++){
                if (zipIndex_dest.get(i) != zipIndex_comp[i]){
                    printf("ZipIndex \t\t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, zipIndex_dest.get(i), zipIndex_comp[i]);
                    break;
                }
                if (i == (elements)-1) {
                    printf("ZipIndex \t\t \xE2\x9C\x93\n");
                }
            }

            DC<int> zipIndexInPlace(5, 5, 5, 3);
            DC<int> zipIndexInPlace_param(5, 5, 5, 2);
            int *zipIndexInPlace_comp = new int[elements];
            for (int j = 0; j < elements; j++) {
                int depth = int(j/(5*5));
                zipIndexInPlace_comp[j] = 3 * depth * (int(j-(depth * 5*5))/5) * (j%5) * 3 * 2;
            }
            zipIndexInPlace.zipIndexInPlace(zipIndexInPlace_param, mul5);
            for (int i = 0; i < elements; i++){
                if (zipIndexInPlace.get(i) != zipIndexInPlace_comp[i]){
                    printf("ZipIndexInPlace \t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, zipIndexInPlace.get(i), zipIndexInPlace_comp[i]);
                    break;
                }
                if (i == (10*10)-1) {
                    printf("ZipIndexInPlace \t \xE2\x9C\x93\n");
                }
            }

          return;
        }
    }} // close namespaces

int main(int argc, char** argv){
  //printf("Starting Main...\n");
  msl::setNumRuns(1);
  msl::setNumGpus(1);
  msl::initSkeletons(argc, argv);
  msl::Muesli::cpu_fraction = 0.2;

  printf("Starting Program %s with %d nodes %d cpus and %d gpus\n", msl::Muesli::program_name, msl::Muesli::num_total_procs,
  msl::Muesli::num_local_procs, msl::Muesli::num_gpus);
  msl::test::dc_test(16);
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}
