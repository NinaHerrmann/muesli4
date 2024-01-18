
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
#include "Randoms.h"

typedef array<double, 2> city;
Randoms *randoms;

#define PHERINIT 0.005
#define EVAPORATION 0.5
#define ALPHA 1
#define BETA 2
#define TAUMAX 2
std::ostream& operator<< (std::ostream& os, const city f) {
    os << "(" << f[0] << ", " << f[1] << ")";
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

    class InitializeDistance : public Functor3<int, int, int, int> {
    public:
        InitializeDistance(DA<city> cities) : Functor3(){
            this->cities = cities;
        }
            MSL_USERFUNC int operator()(int i, int j, int x) const {

        }
    private:
        DA<city> cities;
    };

    class RouteCalculation : public Functor3<int, int, double, double> {
    private:
        DM<double> probabilites;
    public:
        explicit RouteCalculation(DM<double> probs) :
                probabilites(std::move(probs)) {}

        MSL_USERFUNC double operator()(int xindex, int yindex, double value, int ncities) const {
            int newroute = 0;
            int route[ncities] = {0};
            int visited[ncities] = {0};
            double sum = 0.0;
            int next_city = -1;
            double ETA = 0.0;
            double TAU = 0.0;
            double random = 0.0;
            int initial_city = 0;
            route[0] = initial_city;
            /*for (int i = 0; i < ncities-1; i++) {
              int cityi = route[0];
              int count = 0;
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

    DA <city> readsize(const std::string &basicString);

    void readData(const std::string &basicString, int ncities, DA <city> &da, DM<double> &phero);

    void aco(int iterations, const std::string& importFile, int nants) {
        // TODO distance could also be array ncities*ncities/2

        DA<city> cities = readsize(importFile);
        int ncities = cities.getSize();
        DM<double> phero(ncities, ncities, 0.0);
        readData(importFile, ncities, cities, phero);
        // cities.showLocal("cities");
        DM<double> distance(ncities, ncities, 1);
        DA<int> routelength(nants, 0);
        DM<double> probabilities(ncities, ncities, 0.0);
        // TODO parallelization not possible since passing of DA does not work.
        // distance = size cities x cities
        for (int i = 0; i < ncities; i++) {
            for (int j = 0; j < ncities; j++) {
                if (i == j) {
                    distance.setLocal(i*ncities + j, 0);
                } else {
                    distance.setLocal(i*ncities + j, sqrt(pow(cities.get(j)[0] - cities.get(i)[0], 2) +
                                pow(cities.get(j)[1] - cities.get(i)[1], 2)));
                }
            }
        }
        // distance.show("distance");
        phero.show("phero");

        Probabilities prob;
        msl::startTiming();

        for (int i = 0; i < iterations; i++) {
            // Was sollte raus kommen? Möglich ant x cities (tour für jede ant) - DA ants länge jeder tour
            RouteCalculation route(probabilities);
            //phero = ant.combine(probabilities, route, prob, ncities);
        }
        msl::stopTiming();
    }

    void readData(const std::string &basicString, int ncities, DA <city> &cities, DM<double> &phero) {
        std::ifstream data;
        data.open("/home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/data/" + basicString + ".txt", std::ifstream::in);
        randoms = new Randoms(15);
        if (data.is_open()){
            double randn = 0.0;
            for(int j = 0;j<ncities;j++){
                for(int k = 0;k<ncities;k++){
                    if(j!=k){
                        randn = randoms -> Uniforme() * TAUMAX;
                        phero.setLocal((j*ncities) + k, randn);
                        phero.setLocal((k*ncities) + j, randn);
                    }
                    else{
                        phero.setLocal((j*ncities) + k, 0);
                        phero.setLocal((k*ncities) + j, 0);
                    }
                }
            }
            int i = 0;
            double index, x, y;
            index = 0.0; x = 0.0; y = 0.0;
            city city = {0,0};
            while(i < ncities){
                data >> index;
                data >> x;
                data >> y;

                city[0] = (double)x;
                city[1] = (double)y;
                cities.setLocal(i, city);
                i += 1;
            }
            data.close();
            printf("%.2f \t", cities.localPartition[0][0]);
        } else{
            printf(" File not opened\n");
        }
    }

    DA <city> readsize(const std::string &basicString) {
        int n_cities = 0;

        if (basicString == "djibouti") {
            n_cities = 38;
        } else if (basicString == "luxembourg") {
            n_cities = 980;
        } else if (basicString == "catar") {
            n_cities = 194;
        } else if (basicString == "a280") {
            n_cities = 280;
        } else if (basicString == "d198") {
            n_cities = 198;
        } else if (basicString == "d1291") {
            n_cities = 1291;
        } else if (basicString == "lin318") {
            n_cities = 318;
        } else if (basicString == "pcb442") {
            n_cities = 442;
        } else if (basicString == "pcb1173") {
            n_cities = 1173;
        } else if (basicString == "pr1002") {
            n_cities = 1002;
        } else if (basicString == "pr2392") {
            n_cities = 2392;
        } else if (basicString == "rat783") {
            n_cities = 783;
        } else {
            std::cout << "No valid import file provided. Please provide a valid import file." << std::endl;
            exit(-1);
        }
        DA<city> cities(n_cities, {});
        return cities;
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
    int gpus = 1, iterations = 1, cities = 0, ants = 16, runs = 1, nants = 256;
    std::string importFile, exportFile;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            exitWithUsage();
        }
        switch (argv[i++][1]) {
            case 'g':
                gpus = getIntArg(argv[i]);
                break;
            case 'a':
                nants = getIntArg(argv[i]);
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
    if (!importFile.empty()) {
        msl::aco::aco(iterations, importFile, nants);
    } else {
        printf("Providing an import file is mandatory. \n");
        exit(-1);
    }
    if (!exportFile.empty()) {
        // TODO Export result
    }
    msl::terminateSkeletons();
    return EXIT_SUCCESS;
}
