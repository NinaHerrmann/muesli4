
/*
 * dm_test.cpp
 *
 *      Author: Nina Hermann,
 *  		Herbert Kuchen <kuchen@uni-muenster.de>
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

#include <cmath>
#include "dm.h"
#include "cdm.h"
#include "da.h"
#include "muesli.h"
#include <algorithm>

namespace msl {
    namespace test {
        class Probabilities : public Functor2<double, double, double>{
            // j is distance, x is phero
        public: MSL_USERFUNC double operator() (double j, double x) const {
              double result = x * j;
              // Prevent to high values.
              if(result > 1.0){
                result = 1.0;
              }
              return result;
            }
        };
        class Initialize : public Functor3<int, int, int, int>{
        public: MSL_USERFUNC int operator() (int i, int j, int x) const { if (i==j){return 0;}return ((i + 1) * (j+1))%30 +1;}
        };
        class InitializeDouble : public Functor3<int, int, double, double>{
        public: MSL_USERFUNC double operator() (int i, int j, double x) const {
              double phero = (double)((i + 1) * (j+1)%9+1) * 0.01;
              if (i==j){return 0.0;}
              if (phero > 1) {
                phero = 1.0;
              }
              return phero;
            }
        };
        class RouteCalculation : public Functor3<int, int, double, double>{
        private: DM<double> probabilites;
        public:
            RouteCalculation(DM<double> probs):
                probabilites(probs){}
            MSL_USERFUNC double operator() (int i, int j, double x, int ncities) const {
              double phero = (double)((i + 1) * (j+1)%9+1) * 0.01;
              // TODO update
              /*int newroute = 0;
              int route[ncities] = {0};
              int visited[ncities] = {0};
              double sum = 0.0;
              int next_city = -1;
              double ETA = 0.0;
              double TAU = 0.0;
              double random = 0.0;
              int initial_city = 0;
              route[0] = initial_city;
              for (int i=0; i < ncities-1; i++) {
                int cityi = route[0];
                int count = 0;
                // Find the shortest not visited city.

                for (int c = 0; c < ncities; c++) {
                  next_city = d_iroulette[c];
                  int visited = 0;
                  for (int l=0; l <= i; l++) {
                    if (route[l] == next_city){visited[]}
                  }
                  if (!visited){
                    int indexpath = (cityi * ncities) + next_city;
                    double firstnumber = 1 / distance[indexpath];
                    ETA = (double) mkt::pow(firstnumber, BETA);
                    TAU = (double) mkt::pow(phero[indexpath], ALPHA);
                    sum += ETA * TAU;
                  }
                }
                for (int c = 0; c < IROULETE; c++) {
                  next_city = d_iroulette[(cityi * IROULETE) + c];
                  int visited = 0;
                  for (int l=0; l <= i; l++) {
                    if (d_routes[ant_index*ncities+l] == next_city) {visited = 1;}
                  }
                  if (visited) {
                    d_probabilities[ant_index*ncities+c] = 0.0;
                  } else {
                    double dista = (double)distance[cityi*ncities+next_city];
                    double ETAij = (double) mkt::pow(1 / dista , BETA);
                    double TAUij = (double) mkt::pow(phero[(cityi * ncities) + next_city], ALPHA);
                    d_probabilities[ant_index*ncities+c] = (ETAij * TAUij) / sum;
                    count = count + 1;
                  }
                }

                if (0 == count) {
                  int breaknumber = 0;
                  for (int nc = 0; nc < (ncities); nc++) {
                    int visited = 0;
                    for (int l = 0; l <= i; l++) {
                      if (d_routes[(ant_index * ncities) + l] == nc) { visited = 1;}
                    }
                    if (!(visited)) {
                      breaknumber = (nc);
                      nc = ncities;
                    }
                  }
                  newroute = breaknumber;
                } else {
                  random = mkt::rand(0.0, (double)ncities);
                  int ii = -1;
                  double summ = d_probabilities[ant_index*ncities];

                  for(int check = 1; check > 0; check++){
                    if (summ >= random){
                      check = -2;
                    } else {
                      i = i+1;
                      summ += d_probabilities[ant_index*ncities+ii];
                    }
                  }
                  int chosen_city = ii;
                  newroute = d_iroulette[cityi*IROULETE+chosen_city];
                }
                d_routes[(ant_index * ncities) + (i + 1)] = newroute;
                sum = 0.0;
              }


              return value;*/
              return phero;
            }
        };
        void aco(int dim) {
          //printf("Starting dm_test...\n");
          int ncities = 10;
          int ants = 10;
          // TODO distance could also be array ncities*ncities/2
          DM<int> distance(ncities, ncities, 1);
          DA<int> routelength(ants, 0);
          DM<double> phero(ncities, ncities, 2.0);
          DM<double> probabilities(ncities, ncities, 3.0);
          DM<double> ant(1, ants, 0);
          CDM<double> antes(1, ants, 0);
          Initialize init;
          distance.mapIndexInPlace(init);
          InitializeDouble init_d;
          phero.mapIndexInPlace(init_d);

          distance.show("distance");
          phero.show("phero");

          Probabilities prob;
          probabilities = phero.zip(distance, prob);
          probabilities.show();
          //RouteCalculation route(probabilities);
          //phero = ant.combine(probabilities, route, prob, ncities);
          phero.show();
          //pheromon = pheromom.zip(newPheromon,add);
          //int result = a.fold(sum,true);
          //printf("result: %i\n",result);
          return;
        }
    }
} // close namespaces

int main(int argc, char** argv){
  //printf("Starting Main...\n");
  msl::setNumRuns(1);
  msl::setNumGpus(2);
  msl::initSkeletons(argc, argv);
  printf("Starting Program %c with %d nodes %d cpus and %d gpus\n", msl::Muesli::program_name, msl::Muesli::num_total_procs,
         msl::Muesli::num_local_procs, msl::Muesli::num_gpus);
  msl::test::aco(16);
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}
