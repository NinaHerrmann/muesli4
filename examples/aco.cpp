
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
#include "da.h"
#include "dc.h"
#include "muesli.h"
#include <algorithm>
#include <utility>
#include "array.h"
#include "vec2.h"

const size_t nants = 256;
const size_t ncities = 256;
typedef array<float, ncities> tour;
typedef array<tour, nants> tours;
typedef array<int, 2> city;
// typedef vec2<int> city;
std::ostream& operator<< (std::ostream& os, const city f) {
    os << "(" << f[0] << ", " << f[1] << ", " << "...)";
    return os;
}
namespace msl::aco {
    class Probabilities : public Functor2<double, double, double> {
        // j is distance, x is phero
    public:
        MSL_USERFUNC double operator()(double j, double x) const {
            double result = x * j;
            // Prevent to high values.
            if (result > 1.0) {
                result = 1.0;
            }
            return result;
        }
    };

    class Initialize : public Functor3<int, int, int, int> {
    public:
        MSL_USERFUNC int operator()(int i, int j, int x) const {
            if (i == j) { return 0; }
            return ((i + 1) * (j + 1)) % 30 + 1;
        }
    };

    class InitializeDouble : public Functor3<int, int, double, double> {
    public:
        MSL_USERFUNC double operator()(int i, int j, double x) const {
            double phero = (double) ((i + 1) * (j + 1) % 9 + 1) * 0.01;
            if (i == j) { return 0.0; }
            if (phero > 1) {
                phero = 1.0;
            }
            return phero;
        }
    };

    class RouteCalculation : public Functor3<int, int, double, double> {
    private:
        DM<double> probabilites;
    public:
        explicit RouteCalculation(DM<double> probs) :
                probabilites(std::move(probs)) {}

        MSL_USERFUNC double operator()(int i, int j, double x, int ncities) const {
            //double phero = (double) ((i + 1) * (j + 1) % 9 + 1) * 0.01;
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
            return 0.0;
        }
    };

    void aco(int iterations, std::string importFile) {
        // TODO distance could also be array ncities*ncities/2
        DM<int> distance(ncities, ncities, 1);
        DM<double> phero(ncities, ncities, 2.0);
        DA<int> routelength(nants, 0);
        DA<city> cities(ncities);
        DM<double> probabilities(ncities, ncities, 3.0);
        Initialize init;
        distance.mapIndexInPlace(init);
        InitializeDouble init_d;
        phero.mapIndexInPlace(init_d);
        // Calculate the closest x cities.

        Probabilities prob;
        msl::startTiming();

        for (int i = 0; i < iterations; i++) {
            //RouteCalculation route(probabilities);
            //phero = ant.combine(probabilities, route, prob, ncities);
        }
        msl::stopTiming();
    }
} // close namespaces

void exitWithUsage() {
    std::cerr
            << "Usage: ./gassimulation_test [-g <nGPUs>] [-n <iterations>] [-i <importFile>] [-e <exportFile>] [-t <threads>] [-c <cities>] [-a <ants>] [-r <runs>]"
            << "Default 1 GPU 1 Iteration No import File No Export File threads omp_get_max_threads cities 10 random generated cities ants 16 runs 1" <<std::endl;
    exit(-1);
}

int getIntArg(char *s, bool allowZero = false) {
    int i = std::atoi(s);
    if (i < 0 || (i == 0 && !allowZero)) {
        exitWithUsage();
    }
    return i;
}

int main(int argc, char **argv) {
    msl::initSkeletons(argc, argv);
    int gpus = 1, iterations = 1, cities = 0, ants = 16, runs = 1;
    std::string importFile, exportFile;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            exitWithUsage();
        }
        switch (argv[i++][1]) {
            case 'g':
                gpus = getIntArg(argv[i]);
                break;
            case 'n':
                iterations = getIntArg(argv[i], true);
                break;
            case 'i':
                importFile = std::string(argv[i]);
                break;
            case 'e':
                exportFile = std::string(argv[i]);
                break;
            case 't':
                msl::setNumThreads(getIntArg(argv[i]));
                break;
            case 'r':
                runs = (getIntArg(argv[i]));
                break;
            default:
                exitWithUsage();
        }
    }
    msl::setNumGpus(gpus);
    msl::setNumRuns(runs);
    printf("Starting with %d nodes %d cpus and %d gpus\n", msl::Muesli::num_total_procs,
           msl::Muesli::num_local_procs, msl::Muesli::num_gpus);
    msl::aco::aco(iterations, importFile);
    if (!exportFile.empty()) {
        // TODO Export result
    }
    msl::terminateSkeletons();
    return EXIT_SUCCESS;
}
