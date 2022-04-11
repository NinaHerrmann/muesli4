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


        class Sum4 : public Functor4<int, int, int, int, int>{
        public: MSL_USERFUNC int operator() (int i, int j, int x, int y) const {return i+j+x+y;}
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

        void dm_test(int dim) {
            printf("Starting dm_test...\n");

            // ************* Init *********************** //

            DM<int> a(10, 10);
            a.fill(2);
            for (int i = 0; i < 10*10; i++){
                if (a.get(i) != 2){
                    printf("Fill \t\t\t\t \xE2\x9C\x97 At Index At Index %d - Value %d No further checking.\n", i, a.get(i));
                    break;
                }
                if (i == (10*10)-1) {
                    printf("Fill \t\t\t \xE2\x9C\x93\n");
                }
            }
            DM<int> b(10, 10, 5);
            for (int i = 0; i < 10*10; i++){
                if (b.get(i) != 5){
                    printf("Initialize+fill \xE2\x9C\x97 At Index %d - Value %d No further checking.\n", i, b.get(i));
                    break;
                }
                if (i == (10*10)-1) {
                    printf("Initialize+fill \t \xE2\x9C\x93\n");
                }
            }
            Produkt pr;
            Mult4 mul4(3);
            Mult mult(3);
            Sum sum;
            Sum3 sum3;
            Sum4 sum4;

            DM<int> map(10, 10, 3);
            DM<int> map_dest = map.map(mult);
            for (int i = 0; i < 10*10; i++) {
                if (map_dest.get(i) != 9){
                    printf("map \t\t\t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, map_dest.get(i), 9);
                    break;
                }
                if (i == (10*10)-1) {
                    printf("map \t\t\t \xE2\x9C\x93\n");
                }
            }
            DM<int> mapInPlace(10, 10, 2);
            mapInPlace.mapInPlace(mult);
            for (int i = 0; i < 10*10; i++) {
                if (mapInPlace.get(i) != 6){
                    printf("mapInPlace \t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, mapInPlace.get(i), 6);
                    break;
                }
                if (i == (10*10)-1) {
                    printf("mapInPlace \t\t \xE2\x9C\x93\n");
                }
            }
            DM<int> mapIndex(10, 10, 6);
            DM<int> mapIndex_dest = mapIndex.mapIndex(sum3);
            int *mapIndex_comp = new int[10*10];
            for (int j = 0; j < 10 * 10; j++) {
                mapIndex_comp[j] = int(j/10) + (j%10) + 6;
            }
            for (int i = 0; i < 10*10; i++) {
                if (mapIndex_dest.get(i) != mapIndex_comp[i]){
                    printf("mapIndex \t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, mapIndex_dest.get(i), mapIndex_comp[i]);
                    break;
                }
                if (i == (10*10)-1) {
                    printf("mapIndex \t\t \xE2\x9C\x93\n");
                }
            }
            DM<int> amap(10, 10, 2);
            int *amap_comp = new int[10*10];
            for (int j = 0; j < 10 * 10; j++) {
                amap_comp[j] = 2 * int(j/10) * (j%10);
            }
            amap.mapIndexInPlace(pr);
            for (int i = 0; i < 10*10; i++){
                if (amap.get(i) != amap_comp[i]){
                    printf("MapIndexInPlace \t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, amap.get(i), amap_comp[i]);
                    break;
                }
                if (i == (10*10)-1) {
                    printf("MapIndexInPlace \t \xE2\x9C\x93\n");
                }
            }
            // TODO ************* Fold *********************** //
            int result = amap.fold(sum, true);
            int compresult = 0;
            for (int j = 0; j < 10 * 10; j++) {
                compresult += amap_comp[j];
            }
            if (compresult == result) {
                printf("Fold  \t\t\t \xE2\x9C\x93\n");
            } else {
                printf("Fold \t\t\t\t \xE2\x9C\x97  \t\t parallel = %d seq = %d! \n", result, compresult);
            }
            // ************* Zip *********************** //

            DM<int> zip(10, 10);
            DM<int> zip_param(10, 10);
            zip.fill(10);
            zip_param.fill(20);
            DM<int> zip_dest = zip.zip(zip_param,sum);
            for (int i = 0; i < 10*10; i++){
                if (zip_dest.get(i) != 30){
                    printf("Zip \t\t\t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, zip_dest.get(i), 30);
                    break;
                }
                if (i == (10*10)-1) {
                    printf("Zip \t\t\t \xE2\x9C\x93\n");
                }
            }
            DM<int> zipIndex(10, 10, 7);
            DM<int> zipIndex_param(10, 10, 8);
            DM<int> zipIndex_dest = zipIndex.zipIndex(zipIndex_param, sum4);
            int *zipIndex_comp = new int[10*10];
            for (int j = 0; j < 10 * 10; j++) {
                zipIndex_comp[j] = int(j/10) + (j%10) + 7 + 8;
            }
            for (int i = 0; i < 10*10; i++){
                if (zipIndex_dest.get(i) != zipIndex_comp[i]){
                    printf("ZipIndex \t\t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, zipIndex_dest.get(i), zipIndex_comp[i]);
                    break;
                }
                if (i == (10*10)-1) {
                    printf("ZipIndex \t\t \xE2\x9C\x93\n");
                }
            }
            DM<int> zipInPlace(10, 10);
            DM<int> zipInPlace_dest(10, 10);
            zipInPlace.fill(10);
            zipInPlace_dest.fill(10);
            zipInPlace_dest.zipInPlace(zipInPlace, sum);
            for (int i = 0; i < 10*10; i++){
                if (zipInPlace_dest.get(i) != 20){
                    printf("ZipInPlace \t\t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, zipInPlace_dest.get(i), 20);
                    break;
                }
                if (i == (10*10)-1) {
                    printf("ZipInPlace \t\t \xE2\x9C\x93\n");
                }
            }
            DM<int> zipIndexInPlace(10, 10, 4);
            DM<int> zipIndexInPlace_param(10, 10, 2);
            int *zipIndexInPlace_comp = new int[10*10];
            for (int j = 0; j < 10 * 10; j++) {
                zipIndexInPlace_comp[j] = 4 * int(j/10) * (j%10) * 3 * 2;
            }
            zipIndexInPlace.zipIndexInPlace(zipIndexInPlace_param, mul4);
            for (int i = 0; i < 10*10; i++){
                if (zipIndexInPlace.get(i) != zipIndexInPlace_comp[i]){
                    printf("ZipIndexInPlace \t\t \xE2\x9C\x97 At Index %d: Valuep %d != Valueseq %d No further checking.\n", i, zipIndexInPlace.get(i), zipIndexInPlace_comp[i]);
                    break;
                }
                if (i == (10*10)-1) {
                    printf("ZipIndexInPlace \t \xE2\x9C\x93\n");
                }
            }
            /*Proj1 pr1;
            a.zipInPlaceAAM(ar1,ar2,b,pr1);
            a.show("a5");

            Proj2 pr2;
            a.zipInPlaceAAM(ar1,ar2,b,pr2);
            a.show("a6");

            Proj4 pr4;
            a.zipInPlaceAAM(ar1,ar2,b,pr4);
            a.show("a7");

            CopyCond copyCond;
            a.zipInPlaceAAM(ar1,ar2,c,copyCond);
            a.show("a8");*/

          return;
        }
    }} // close namespaces

int main(int argc, char** argv){
  //printf("Starting Main...\n");
  msl::setNumRuns(1);
  msl::setNumGpus(2);
  msl::initSkeletons(argc, argv);
  msl::Muesli::cpu_fraction = 0.2;

  printf("Starting Program %s with %d nodes %d cpus and %d gpus\n", msl::Muesli::program_name, msl::Muesli::num_total_procs,
  msl::Muesli::num_local_procs, msl::Muesli::num_gpus);
  msl::test::dm_test(16);
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}
