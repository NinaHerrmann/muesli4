
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#include <cmath>
#include "dm.h"
#include "da.h"
#include "muesli.h"
#include <algorithm>
#include "array.h"
#include "Randoms.h"

#ifdef __CUDACC__
#define MSL_MANAGED __managed__
#define MSL_CONSTANT __constant__
#define POW(a, b)      powf(a, b)
#define EXP(a)      exp(a)
__device__ double d_PHERINIT;
__device__ double d_EVAPORATION;
__device__ double d_ALPHA;
__device__ double d_BETA ;
__device__ double d_TAUMAX;
__device__ int d_BLOCK_SIZE;
__device__ int d_GRAPH_SIZE;
#else
#define MSL_MANAGED
#define MSL_CONSTANT
#define POW(a, b)      std::pow(a, b)
#define EXP(a)      std::exp(a)
#endif

const int Q = 38;
typedef array<double, 2> city;
Randoms *randoms;


#define TAUMAX 2
#define IROULETE 32
#define CHECKCORRECTNESS 1

std::ostream& operator<< (std::ostream& os, const city t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
namespace msl::aco {

    class Reset : public Functor<double, double> {
    public:
        MSL_USERFUNC double operator()(double x) const override {
            return 0.0;
        }
    };

    class EtaTauCalc : public Functor3<int, int, city, city> {
    private:
        int * iroulette{}, * routes{};
        double * phero{}, * dist{};
        int cities, iteration;
    public:
        EtaTauCalc(int cities, int i) {
            this->cities = cities;
            this->iteration = i;
        }
        void setIterationsParams(int * iroulet, int i, double * distance, int * tours, double * pheromones) {
            this->iroulette = iroulet;
            this->iteration = i;
            this->dist = distance;
            this->routes = tours;
            this->phero = pheromones;
        }

        MSL_USERFUNC city operator()(int xindex, int yindex, city eta_tau) const override {
            int d_ALPHA = 1;
            int d_BETA = 2;
            int fromcity = routes[xindex * cities + iteration];
            int next_city = iroulette[(fromcity * 32) + yindex];
            bool bvisited = false;
            city eta_tau_return = {0,0};
            for (int y = 0; y < this->cities; y++) {
                if (routes[xindex * cities + y] == next_city) {
                    bvisited = true;
                    break;
                }
            }
            // For every city which can be visited, calculate the eta and tau value.
            if (fromcity != next_city && !bvisited) {
                // Looks like zero but is just very small.
                eta_tau_return[0] = (double) pow(1 /  dist[fromcity * this->cities + next_city], d_BETA);
                eta_tau_return[1] = (double) pow(phero[fromcity * this->cities + next_city], d_ALPHA);
            }
            return eta_tau_return;
        }
    };

    class CalcProbs : public Functor5<int, int, double, city, double, double> {
    private:
        int i{}, ncities{};
        int * routes{}, * iroulette{};
    public:
        CalcProbs(int i, int ncities, int * iroulet) {
            this->i = i;
            this->ncities = ncities;
            this->iroulette = iroulet;
        }
        void setIterationsParams(int ii, int * tours) {
            this->i = ii;
            this->routes = tours;
        }

        MSL_USERFUNC double operator()(int x, int y, double t, city etatau, double sum) const override {
            // Calculates the probability of an ant going to the city at index x in the 32 closest cities.
            int cityi = routes[x*ncities+i];
            int next_city = iroulette[cityi*IROULETE+y];
            if (cityi == next_city || visited(x, next_city, routes, ncities, i+1)) {
                return 0;
            } else {
                if (sum == 0.0) {
                    return 0;
                } else {
                    return (etatau[0] * etatau[1]) / sum;
                }
            }
        }
        MSL_USERFUNC static bool visited(int antk, int c, const int* routes, int n_cities, int step) {
            for (int l=0; l <= step; l++) {
                if (routes[antk*n_cities+l] == c) {
                    return true;
                }
            }
            return false;
        }
    };
    class SUM : public Functor2<city, double, double> {
    public:
        MSL_USERFUNC double operator()(city x, double y) const override {
            return (x[0] * x[1]) + y;
        }
    };
    class Min : public Functor2<double, double, double> {
    public:
        MSL_USERFUNC double operator()(double x, double y) const override {
            if (x < y) { return x; }
            return y;}
    };
    class ZipSum : public Functor3<int, int, double, double> {
    private:
        double * dist{};
        int ncities{};
    public:
        explicit ZipSum(int cities) {
            this->ncities = cities;
        }
        void setDist(double * distance) {
            this->dist = distance;
        }
        MSL_USERFUNC double operator()(int x, int x2, double y) const override {
            double newdist = dist[x*ncities + x2];
            return y + newdist;
        }
    };
    class CalcRlength : public Functor2<int, double, double> {
    private:
        int * routes{};
        double * distances{};
        int ncities;
    public:
        explicit CalcRlength(int cities) {
            this->ncities = cities;
        }
        void setIterationsParams(int * tours, double * distance) {
            this->routes = tours;
            this->distances = distance;
        }
        MSL_USERFUNC double operator()(int x, double value) const override {
            double sum = 0.0;
            for (int j=0; j < ncities - 1; j++) {
                int cityi = routes[x * ncities + j];
                int cityj = routes[x * ncities + j + 1];
                sum += distances[cityi * ncities + cityj];
            }

            int cityi = routes[x * ncities + ncities - 1];
            int cityj = routes[x * ncities];
            sum += distances[cityi * ncities + cityj];

            return sum;
        }
    };
    class nextStep : public Functor3<int, int, int, int> {
    private:
        int * iroulette{};
        double * d_sum{};
        int * routes{};
        double * d_probs{};
        int ncities, i;
        double * randoms;
    public:
        nextStep(int i, int ncities, double * randomp) {
            this->i = i;
            this->ncities = ncities;
            this->randoms = randomp;
        }
        void setIterationsParams(int * iroulet, int ii, double * dprobs, double * dSum, int * tours) {
            this->iroulette = iroulet;
            this->i = ii;
            this->d_probs = dprobs;
            this->d_sum = dSum;
            this->routes = tours;
        }

        MSL_USERFUNC int operator()(int ant_index, int column, int citytobe) const override {
            if (column != i) {
                return citytobe;
            } else {
                int cityi = routes[ant_index * ncities + i];
                if (d_sum[ant_index] > 0.0) {
                    int nextCity = city(ant_index, d_probs, randoms, 32);
                    if (nextCity == -1) {
                        int nc;
                        for (nc = 0; nc < ncities; nc++) {
                            if (!visited(ant_index, nc, routes, ncities, i+1)) {
                                break;
                            }
                        }
                        return nc;
                    }
                    return iroulette[cityi*IROULETE+nextCity];
                } else {
                    int nc;
                    for (nc = 0; nc < ncities; nc++) {
                        if (!visited(ant_index, nc, routes, ncities, i+1)) {
                            break;
                        }
                    }
                    return nc;
                }
            }
        }

        MSL_USERFUNC static int city(int antK, const double *probabilities, const double *rand_states, int iroulette) {
            double random = rand_states[antK];
            int i = 0;
            // In case a city was already visited the probability is zero therefore the while loop should not terminate.
            double sum = probabilities[antK * iroulette];
            while (sum < random) {
                i++;
                sum += probabilities[antK * iroulette + i];
            }
            if (i > iroulette) {
                return -1;
            }
            return (int) i;
        }

        MSL_USERFUNC static bool visited(int antk, int c, const int* routes, int n_cities, int step) {
            for (int l=0; l <= step; l++) {
                if (routes[antk*n_cities+l] == c) {
                    return true;
                }
            }
            return false;
        }
    };


    class UpdateDelta : public Functor3<int, int, double, double> {
    private:
        int *routes{};
        double *dist_routes{};
        int nants, ncities;
    public:
        UpdateDelta(int cities, int ants) {
            this->nants = ants;
            this->ncities = cities;
        }

        void setIterationsParams(int *tours, double *distroutes) {
            this->routes = tours;
            this->dist_routes = distroutes;
        }

        MSL_USERFUNC double operator()(int row, int column, double phero) const override {
            int Q = 11340;
            int cityi = row+1;
            int cityj = column+1;
            double result = 0.0;
            for (int k = 0; k < nants; k++) {
                double rlength = dist_routes[k];
                dist_routes[k] = rlength;
                for (int r = 0; r < ncities-1; r++) {
                    if ((routes[k * ncities + r] == cityi && routes[k * ncities + r + 1] == cityj) ||
                            (routes[k * ncities + r] == cityj && routes[k * ncities + r + 1] == cityi)) {
                        result += Q / rlength;
                    }
                }
            }
            return result;
        }
    };
    class UpdatePhero : public Functor4<int, int, double, double, double> {
    public:
        MSL_USERFUNC double operator()(int x, int y, double prevphero, double deltaphero) const override {
            double RO = 0.5;
            return (1 - RO) * (prevphero + deltaphero);
        }
    };
    int readsize(const std::string &basicString) {
        int n_cities;
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
        return n_cities;
    }


    void readData(const std::string &basicString, int ncities, DA <city> &cities, DM<double> &phero) {
        std::ifstream data;
        data.open("/home/n_herr03@WIWI.UNI-MUENSTER.DE/Schreibtisch/muesli/data/" + basicString + ".txt", std::ifstream::in);
        randoms = new Randoms(15);

        if (data.is_open()){
            double randn;
            for(int j = 0;j<ncities;j++){
                for(int k = 0;k<ncities;k++){
                    if(j!=k){
                        randn = randoms -> Uniforme() * TAUMAX;
                        phero.setLocal((j * ncities) + k, randn);
                        phero.setLocal((k * ncities) + j, randn);
                    }
                    else{
                        phero.setLocal((j * ncities) + k, 0.0);
                        phero.setLocal((k * ncities) + j, 0.0);
                    }
                }
            }
            int i = 0;
            double index, x, y;
            index = 0.0; x = 0.0; y = 0.0;
            city city = {0.0,0.0};
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
        } else{
            printf(" File not opened\n");
        }
    }
    void checkminroute(int nants, double minroute, const DA<double>& dist_routes) {
        for (int ii = 0; ii < nants; ii++) {
            if (minroute > dist_routes.localPartition[ii]) {
                printf("minimum route %.2f bigger than distance at %d %.2f\n", minroute, ii, dist_routes.localPartition[ii]);
                exit(1);
            }
        }
    }
    void checkvalidroute(const DM<int>& routes, int ncities, int nants) {
        for (int ii = 0; ii < nants; ii++) {
            for (int jj = 0; jj < ncities; jj++) {
                int currentcity = routes.localPartition[ii * ncities + jj];
                for (int kk = 0; kk < ncities; kk++) {
                    if (kk != jj && currentcity == routes.localPartition[ii * ncities + kk]) {
                        printf("city %d visited twice in ant %d\n", currentcity, ii);
                        exit(1);
                    }
                }
            }
        }
    }
    void aco(int iterations, const std::string& importFile, int nants) {
        int niroulet = 32;
        int ncities = readsize(importFile);
        DA<city> cities(ncities, {});
        DM<double> phero(ncities, ncities, {});
        DM<double> distance(ncities, ncities, {});
        readData(importFile, ncities, cities, phero);
        DA<int> routelength(nants, 0);
        DM<int> iroulet(ncities, niroulet, 0);
        DM<double> probabilities(nants, niroulet, 0.0);
        for (int i = 0; i < ncities; i++) {
            for (int j = 0; j < ncities; j++) {
                if (i == j) {
                    distance.setLocal(i*ncities + j, 0.0);
                } else {
                    double dist = sqrt(pow(cities.get(j)[0] - cities.get(i)[0], 2) +
                                   pow(cities.get(j)[1] - cities.get(i)[1], 2));
                    distance.setLocal(i*ncities + j, dist);
                }
            }
        }
        for (int i = 0; i < ncities; i++) {
            for (int y = 0; y < IROULETE; y++) {

                double maxdistance = 999999.9;
                double c_dist;
                int city = -1;
                for (int j = 0; j < ncities; j++) {
                    bool check = true;
                    for (int k = 0; k < y; k++) {
                        if (iroulet.get(i * IROULETE + k) == j) {
                            check = false;
                        }
                    }

                    if (i != j && check) {
                        c_dist = distance.get(i*ncities + j);
                        if (c_dist < maxdistance) {
                            maxdistance = c_dist;
                            city = j;
                        }
                    }
                }
                iroulet.set(i * IROULETE + y, city);
            }
        }
        msl::syncStreams();
        DM<int> tours(nants, ncities, 0);
        DM<double> deltaphero(ncities, ncities, 0);
        DM<city> etatau(nants, niroulet, {});
        DA<double> sum(nants, 0.0);
        DA<double> dist_routes(nants, 0.0);
        DA<double> r_length(nants, 0.0);
        SUM summe;
        EtaTauCalc etataucalc(ncities, 0);
        CalcProbs calcprobs(0, ncities, iroulet.getGpuData());
        DA<double> randompointer(nants, 0.0);
        for (int i = 0; i < nants; i++) {
            randompointer.set(i, randoms->Uniforme());
        }
        nextStep nextstep(0, ncities, randompointer.getGpuData());
        Reset reset;
        ZipSum zipsum(ncities);
        Min min;
        double minroute;
        zipsum.setDist(distance.getLocalPartition());
        UpdateDelta updatedelta(ncities, nants);
        UpdatePhero updatephero;
        CalcRlength calcrlength(ncities);
        msl::startTiming();
        double alltimeminroute = 999999.9;
        for (int i = 0; i < iterations; i++) {
            for (int j = 0; j < ncities; j++) {
                etataucalc.setIterationsParams(iroulet.getGpuData(), j, distance.getGpuData(), tours.getGpuData(), phero.getGpuData());
                // Write the eta tau value to the data structure.
                etatau.mapIndexInPlace(etataucalc);
                // Write the sum of the etatau value for each ant to the sum datastructure.
                etatau.reduceColumns(sum, summe);
                calcprobs.setIterationsParams(j, tours.getGpuData());
                // Set the probabilites to visit city x next.
                probabilities.zipIndexInPlaceMA(etatau, sum, calcprobs);
                nextstep.setIterationsParams(iroulet.getGpuData(), j, probabilities.getGpuData(), sum.getGpuData(), tours.getGpuData());
                // Getting to the heart of it. Either we want to randomly choose one of the next 32 closest cities ...
                // ... or we want to take a city not visited.
                tours.mapIndexInPlace(nextstep);
                sum.mapInPlace(reset);
            }
            calcrlength.setIterationsParams(tours.getGpuData(), distance.getGpuData());
            // Calculate the length of the route.
            dist_routes.mapIndexInPlace(calcrlength);
            // Get the best route.
            minroute = dist_routes.foldCPU(min);
            if (minroute < alltimeminroute) {
                alltimeminroute = minroute;
            }
            printf("Minroute %.2f \n", minroute);
            updatedelta.setIterationsParams(tours.getGpuData(), dist_routes.getGpuData());
            // Calculate the delta pheromone.
            deltaphero.mapIndexInPlace(updatedelta);
            // Update the pheromone.
            phero.zipIndexInPlace(deltaphero, updatephero);
        }
        // Idee Save as {int, int} sehr viele Dopplungen...?
        // tours.reducetwoColumns(dist_routes, zipsum);
        // minroute = dist_routes.foldCPU(min);
        printf("AlltimeminRoute: %.2f \n", alltimeminroute);

        if (CHECKCORRECTNESS) {
            checkminroute(nants, minroute, dist_routes);
            checkvalidroute(tours, ncities, nants);
            printf("Made it!\n");
        }
        msl::stopTiming();
        printf("Minimum Route %.2f \n", minroute);
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
    // TODO does not work for less
    int gpus = 1, iterations = 1, runs = 1, nants = 256;
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
