
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
#include <random>
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

    class EtaTauCalc : public Functor3<int, int, double, double> {
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

        MSL_USERFUNC double operator()(int xindex, int yindex, double eta_tau) const override {
            int d_ALPHA = 1;
            int d_BETA = 2;
            int fromcity = routes[xindex * cities + iteration];
            int next_city = iroulette[(fromcity * IROULETE) + yindex];
            bool bvisited = false;
            city eta_tau_return = {0,0};
            for (int y = 0; y < this->cities; y++) {
                if (routes[xindex * cities + y] == next_city) {
                    bvisited = true;
                    break;
                }
            }
            eta_tau = 0;
            // For every city which can be visited, calculate the eta and tau value.
            if (fromcity != next_city && !bvisited) {
                // Looks like zero but is just very small.
                eta_tau_return[0] = (double) pow(1/dist[fromcity * this->cities + next_city], d_BETA);
                eta_tau_return[1] = (double) pow(phero[fromcity * this->cities + next_city], d_ALPHA);
                eta_tau = eta_tau_return[0] * eta_tau_return[1];
            }
            return eta_tau;
        }
    };

    class CalcProbs : public Functor5<int, int, double, double, double, double> {
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

        MSL_USERFUNC double operator()(int x, int y, double t, double etatau, double sum) const override {
            // Calculates the probability of an ant going to the city at index x in the IROULETE closest cities.
            int cityi = routes[x*ncities+i];
            int next_city = iroulette[cityi*IROULETE+y];
            if (cityi == next_city || visited(x, next_city, routes, ncities, i+1)) {
                return 0;
            } else {
                if (sum == 0.0) {
                    return 0;
                } else {
                    return etatau / sum;
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
    class Mult : public Functor<city, double> {
    public:
        MSL_USERFUNC double operator()(city x) const override {
            return (x[0] * x[1]);
        }
    };
    class SUM : public Functor2<double, double, double> {
    public:
        MSL_USERFUNC double operator()(double x, double y) const override {
            return x + y;
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
    class initialCity : public Functor3<int, int, int, int> {
    private:
        int *randoms;
    public:
        initialCity(int * randomp) {
            this->randoms = randomp;
        }
        MSL_USERFUNC int operator()(int antindex, int city, int value) const override {
            if (city != 0) {
                return value;
            } else {
                return randoms[antindex];
            }
        }
    };

    class calcDistance : public Functor3<int, int, double, double> {
    private:
        city *cities;
    public:
        explicit calcDistance(city * cities) {
            this->cities = cities;
        }
        MSL_USERFUNC double operator()(int cityi, int cityj, double value) const override {
            if (cityi == cityj) {
                return 0.0;
            } else {
                return sqrt(pow(cities[cityj][0] - cities[cityi][0], 2) +
                                          pow(cities[cityj][1] - cities[cityi][1], 2));;
            }
        }
    };
    class nextStep2 : public Functor2<int, int, int> {
    private:
        int * iroulette{};
        double * d_sum{};
        int * routes{};
        double * d_probs{};
        int ncities, i;
        double * randoms;
    public:
        nextStep2(int i, int ncities, double * randomp) {
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

        MSL_USERFUNC int operator()(int ant_index, int citytobe) const override {
            int cityi = routes[ant_index * ncities + i];
            if (d_sum[ant_index] > 0.0) {
                double random = randoms[ant_index];
                int j = 0;
                // In case a city was already visited the probability is zero therefore the while loop should not terminate.
                double sum = d_probs[ant_index * IROULETE];
                while (sum < random & j <= IROULETE) {
                    j++;
                    sum += d_probs[ant_index * IROULETE + i];
                }
                if (j < IROULETE) {
                    return iroulette[cityi*IROULETE+j];
                } else {
                    int nc;
                    for (nc = 0; nc < ncities; nc++) {
                        bool visited = false;
                        for (int l=0; l <= i+1; l++) {
                            if (routes[ant_index*ncities+l] == nc) {
                                visited = true;
                            }
                        }
                        if (!visited) {
                            return nc;
                        }
                    }
                }
            } else {
                int nc;
                for (nc = 0; nc < ncities; nc++) {
                    bool visited = false;
                    for (int l=0; l <= i+1; l++) {
                        if (routes[ant_index*ncities+l] == nc) {
                            visited = true;
                        }
                    }
                    if (!visited) {
                        return nc;
                    }
                }
            }
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
            int cityi = row;
            int cityj = column;
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
    bool check_ant(int ncities, const DM<int>& routes, int ant) {
        bool returnvalid = true;
        for (int jj = 0; jj < ncities; jj++) {
            int currentcity = routes.localPartition[ant * ncities + jj];
            for (int kk = 0; kk < ncities; kk++) {
                if (kk != jj && currentcity == routes.localPartition[ant * ncities + kk]) {
                    returnvalid = false;
                }
            }
        }
        return returnvalid;
    }
    void checkvalidroute(const DM<int>& routes, int ncities, int nants) {
        int * invalidants = new int[nants];
        int j = 0;

        for (int ii = 0; ii < nants; ii++) {
            if (!check_ant(ncities, routes, ii)) {
                printf("Ant %d is invalid;", ii);
                exit(1);
                invalidants[j] = ii;
                j++;
            }
        }
       /* for (int jj=0; jj <= j; jj++) {
            printf("[ ");
            for (int kk = 0; kk < ncities; kk++) {
                printf("%d ", routes.localPartition[invalidants[jj] * ncities + kk]);
            }
            printf("]\n");
        }*/
    }
    void aco(int iterations, const std::string& importFile, int nants) {
        int ncities = readsize(importFile);
        double ds2fill, calctime;
        msl::startTiming();
        DA<city> cities(ncities, {});
        DM<double> phero(ncities, ncities, {});
        DM<double> distance(ncities, ncities, {});
        readData(importFile, ncities, cities, phero);
        DM<int> iroulet(ncities, IROULETE, 0);
        DM<double> probabilities(nants, IROULETE, 0.0);
        calcDistance calcdistance(cities.getUserFunctionData());
        distance.mapIndexInPlace(calcdistance);
        distance.updateHost();

        for (int i = 0; i < ncities; i++) {
            for (int y = 0; y < IROULETE; y++) {
                double maxdistance = 999999.9;
                double c_dist;
                int city = -1;
                for (int j = 0; j < ncities; j++) {
                    bool check = true;
                    for (int k = 0; k < y; k++) {
                        if (iroulet.getLocal(i * IROULETE + k) == j) {
                            check = false;
                        }
                    }

                    if (i != j && check) {
                        c_dist = distance.getLocal(i*ncities + j);
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
        DA<int> nextcities(nants, 0);
        DM<double> deltaphero(ncities, ncities, 0);
        DM<double> etatau(nants, IROULETE, {});
        DA<double> sum(nants, 0.0);
        DA<double> dist_routes(nants, 0.0);
        SUM summe;
        EtaTauCalc etataucalc(ncities, 0);
        CalcProbs calcprobs(0, ncities, iroulet.getUserFunctionData());
        DA<double> randompointer(nants, 0.0);
        for (int i = 0; i < nants; i++) {
            randompointer.set(i, randoms->Uniforme());
        }
        DA<int> randomstartcity(nants, 0);
        std::random_device rd;
        // Use Mersenne Twister engine for random number generation
        std::mt19937 gen(rd());
        // Define the distribution for integers between 1 and x
        std::uniform_int_distribution<> dis(1, ncities);
        for (int i = 0; i < nants; i++) {
            randomstartcity.set(i, dis(gen));
        }
        nextStep2 nextstep2(0, ncities, randompointer.getUserFunctionData());
        initialCity initialcity(randomstartcity.getUserFunctionData());
        tours.mapIndexInPlace(initialcity);
        Reset reset;
        ZipSum zipsum(ncities);
        Min min;
        double minroute;
        zipsum.setDist(distance.getLocalPartition());
        UpdateDelta updatedelta(ncities, nants);
        UpdatePhero updatephero;
        CalcRlength calcrlength(ncities);
        ds2fill = msl::stopTiming();

        double alltimeminroute = 999999.9;
        msl::startTiming();

        for (int i = 0; i < iterations; i++) {
            for (int j = 1; j < ncities; j++) {
                etataucalc.setIterationsParams(iroulet.getUserFunctionData(), j, distance.getUserFunctionData(),
                                               tours.getUserFunctionData(),
                                               phero.getUserFunctionData());
                // Write the eta tau value to the data structure.
                etatau.mapIndexInPlace(etataucalc);
                // Write the sum of the etatau value for each ant to the sum datastructure.
                etatau.reduceRows(sum, summe);
                calcprobs.setIterationsParams(j, tours.getUserFunctionData());
                probabilities.zipIndexInPlaceMA(etatau, sum, calcprobs);

                // TODO Rather create an array and a set column method?
                // Getting to the heart of it. Either we want to "randomly" choose one of the next IROULETE closest cities ...
                // ... or we want to take a city not visited.
                nextstep2.setIterationsParams(iroulet.getUserFunctionData(), j, probabilities.getUserFunctionData(),
                                              sum.getUserFunctionData(),
                                              tours.getUserFunctionData());
                nextcities.mapIndexInPlace(nextstep2);

                tours.setColumn(nextcities, j);

                msl::syncStreams();
                sum.mapInPlace(reset);
            }
            calcrlength.setIterationsParams(tours.getUserFunctionData(), distance.getUserFunctionData());
            // Calculate the length of the route.
            dist_routes.mapIndexInPlace(calcrlength);
            // Get the best route.
            //dist_routes.show();
            //double minroutegpu = dist_routes.fold(min, true);
            minroute = dist_routes.foldCPU(min);
            //dist_routes.show();
            if (minroute < alltimeminroute) {
                alltimeminroute = minroute;
            }
            updatedelta.setIterationsParams(tours.getUserFunctionData(), dist_routes.getUserFunctionData());
            // Calculate the delta pheromone.
            deltaphero.mapIndexInPlace(updatedelta);
            // Update the pheromone.
            phero.zipIndexInPlace(deltaphero, updatephero);
        }
        printf("minroute %f\n", minroute);

        calctime = msl::stopTiming();
        printf("%s;%d;%.6f;%.4f;\n",  importFile.c_str(), nants, ds2fill, calctime);
               // %.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f  etataucalctime, reduceRowstime, calcprobstime, nextsteptime, deltapherotime, updatepherotime, calcrlengthtime, minroutetime, alltimeminroute);
        dist_routes.updateHost();
        tours.updateHost();
        if (CHECKCORRECTNESS) {
            checkminroute(nants, minroute, dist_routes);
            checkvalidroute(tours, ncities, nants);
            printf("Made it!\n");
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
        printf("get Int arg %d %d\n", i, allowZero);
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
            printf("entering ! = - \n");
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
                printf("entering default\n");
                exitWithUsage();
        }
    }
    msl::setNumGpus(gpus);
    msl::setNumRuns(runs);
    if (!importFile.empty()) {
        msl::aco::aco(iterations, importFile, nants);
    } else {
        printf("Providing an import file is mandatory. \n");
        exit(-1);
    }
    msl::terminateSkeletons();
    return EXIT_SUCCESS;
}
