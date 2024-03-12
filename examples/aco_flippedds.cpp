
/*
 * aco_flippedds.cpp
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

#define TAUMAX 2
#define IROULETE 32
#define CHECKCORRECTNESS 1

std::ostream& operator<< (std::ostream& os, const city t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
namespace msl::aco {

    class Fill : public Functor<int, int> {
        int d;
    public:
        explicit Fill(int d) : d(d) {}

        MSL_USERFUNC int operator()(int x) const override {
            return d;
        }
    };

    class EtaTauCalc : public Functor5<int, int, double, double, double, double> {
    public:
        MSL_USERFUNC double operator()(int xindex, int yindex, double eta_tau, double dist, double phero) const override {
            double d_ALPHA = 1;
            double d_BETA = 2;
            int fromcity = xindex;
            int next_city = yindex;

            eta_tau = 0;
            // For every city which can be visited, calculate the eta and tau value.
            if (fromcity != next_city) {
                // Looks like zero but is just very small.
                double eta = pow(1/dist, d_BETA);
                double tau = pow(phero, d_ALPHA);
                eta_tau = eta * tau;
            }
            return eta_tau;
        }
    };
    class Min : public Functor2<double, double, double> {
    public:
        MSL_USERFUNC double operator()(double x, double y) const override {
            if (x < y) { return x; }
            return y;}
    };

    class CalcDistance : public Functor3<int, int, double, double> {
    private:
        city *cities;
    public:
        explicit CalcDistance(city * cities) {
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
    class WorkHorseAllHandler : public Functor2<int, double, double> {
    private:
        int width;
        int seed;
        int *iroulette{};
        double *etataus{};
        double *distances{};
        int *tours{};
    public:
        WorkHorseAllHandler(int width, int seed) : width(width), seed(seed) {
        }

        void setIterationParams(int *_iroulette, double *_etataus, double *_distances, int *_tours) {
            this->iroulette = _iroulette;
            this->etataus = _etataus;
            this->distances = _distances;
            this->tours = _tours;
        }

        MSL_USERFUNC double operator()(int row, double unused) const override {
            MSL_RANDOM_STATE randomState = msl::generateRandomState(this->seed, row);

            int *rowdata = &this->tours[row * width];

            int fromCity = msl::randInt(0, width - 1, randomState);
            rowdata[fromCity] = 0;

            double distance = 0;

            for (int i = 1; i < width; i++) {
                int nextCity = -1;
                double etaTauSum = 0;
                for (int j = 0; j < IROULETE; j++) {
                    int toCity = iroulette[row * IROULETE + j];
                    // Not visited yet.
                    if (rowdata[toCity] == -1) {
                        // x = fromCity, y = toCity.
                        double etatau = etataus[toCity * width + fromCity];
                        // If there is a city with infinite etatau => in the same place. Just go directly to it.
                        if (std::isinf(etatau)) {
                            nextCity = toCity;
                            break;
                        }
                        etaTauSum += etatau;
                    }
                }

                if (nextCity == -1) {
                    if (etaTauSum != 0) {
                        double rand = msl::randDouble(0.0, etaTauSum, randomState);
                        double etaTauSum2 = 0;

                        for (int j = 0; j < IROULETE; j++) {
                            nextCity = iroulette[row * IROULETE + j];
                            if (rowdata[nextCity] == -1) {
                                etaTauSum2 += etataus[nextCity * width + fromCity];
                            }
                            if (rand < etaTauSum2)
                                break;
                        }
                    } else {
                        // Select any city at random
                        int startCity = msl::randInt(0, width - 1, randomState);
                        for (int j = 0; j < width; j++) {
                            if (rowdata[(startCity + j) % width] == -1) {
                                nextCity = (startCity + j) % width;
                                break;
                            }
                        }
                    }
                }

                rowdata[nextCity] = i;
                distance += distances[nextCity * width + fromCity];
                fromCity = nextCity;
            }
            return distance;
        }
    };
    class TourConstruction: public Functor3<int, int *, double, double> {
    private:
        int width;
        int seed;
        int * iroulette{};
        double * etataus{};
        double * distances{};
    public:
        TourConstruction(int width, int seed) : width(width), seed(seed) {
        }

        void setIterationParams(int * _iroulette, double * _etataus, double * _distances) {
            this->iroulette = _iroulette;
            this->etataus = _etataus;
            this->distances = _distances;
        }

        MSL_USERFUNC double operator()(int ant_index, int * tour, double current_distance) const override {
            MSL_RANDOM_STATE randomState = msl::generateRandomState(this->seed, ant_index);
            int fromCity = msl::randInt(0, width - 1, randomState);

            tour[fromCity] = 0;
            double distance = 0;
            for (int i = 1; i < width; i++) {
                int nextCity = -1;
                double etaTauSum = 0;
                for (int j = 0; j < IROULETE; j++) {
                    int toCity = iroulette[fromCity * IROULETE + j];
                    if (tour[toCity] == -1) {
                        etaTauSum += etataus[toCity * width + fromCity];
                    }
                }
                if (etaTauSum != 0) {
                    double rand = msl::randDouble(0.0, etaTauSum, randomState);
                    double etaTauSum2 = 0;

                    for (int j = 0; j < IROULETE; j++) {
                        nextCity = iroulette[fromCity * IROULETE + j];
                        if (tour[nextCity] == -1) {
                            etaTauSum2 += etataus[nextCity * width + fromCity];
                        }
                        if (rand < etaTauSum2)
                            break;
                    }
                } else {
                    int startCity = msl::randInt(0, width - 1, randomState);
                    for (int j = 0; j < width; j++) {
                        if (tour[(startCity + j) % width] == -1) {
                            nextCity = (startCity + j) % width;
                        }
                    }
                }
                tour[nextCity] = i;
                distance += distances[nextCity * width + fromCity];
                fromCity = nextCity;
            }
            return distance;
        }
    };

    class UpdateDelta : public Functor3<int, int, double, double> {
    private:
        int *flipped_routes{};
        double *dist_routes{};
        int nants, ncities;
    public:
        UpdateDelta(int cities, int ants) {
            this->nants = ants;
            this->ncities = cities;
        }

        void setIterationsParams(int *tours, double *distroutes) {
            this->flipped_routes = tours;
            this->dist_routes = distroutes;
        }

        MSL_USERFUNC double operator()(int row, int column, double phero) const override {
            int Q = 11340;
            double result = 0.0;
            for (int k = 0; k < nants; k++) {
                double rlength = dist_routes[k];
                int city1VisitIndex = flipped_routes[k * ncities + row];
                int city2VisitIndex = flipped_routes[k * ncities + column];
                if (abs(city1VisitIndex - city2VisitIndex) == 1) {
                    result += Q / rlength;
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

    void checkminroute(int nants, double minroute, const DA<double>& dist_routes) {
        for (int ii = 0; ii < nants; ii++) {
            if (minroute > dist_routes.localPartition[ii]) {
                printf("minimum route %.2f bigger than distance at %d %.2f\n", minroute, ii, dist_routes.localPartition[ii]);
                exit(1);
            }
        }
    }

    void checkvalidroute(const DM<int>& routes, int ncities, int nants) {
        for (int ant = 0; ant < nants; ant++) {
            for (int n = 0; n < ncities; n++) {
                int found = 0;
                for (int i = 0; i < ncities; i++) {
                    if (routes.localPartition[ant * ncities + i] == n)
                        found++;
                }
                if (found != 1) {
                    printf("Ant %d: Visit %d was found %d times\n", ant, n, found);
                    exit(1);
                }
            }
        }
    }

    DA<city> readCities(const std::string &problem) {
        std::ifstream data("/home/n_herr03@WIWI.UNI-MUENSTER.DE/research/aco-project/programs/lowlevel/tsp/tsplib/" + problem + ".txt");

        if (!data.is_open()) {
            std::cerr << "File could not be opened!" << std::endl;
            exit(-1);
        }

        std::vector<city> cities;

        while(true) {
            int index;
            city city;
            data >> index;
            if (data.fail())
                break;
            data >> city[0];
            data >> city[1];
            cities.push_back(city);
        }
        data.close();

        DA<city> cityArray(cities.size());
        for (int i = 0; i < cities.size(); i++) {
            cityArray.localPartition[i] = cities[i];
        }
        cityArray.updateDevice(true);

        return cityArray;
    }

    DM<double> createPheroMatrix(int ncities) {
        Randoms randoms(15);
        DM<double> phero(ncities, ncities, 0);
        for (int j = 0; j < ncities; j++) {
            for (int k = 0; k <= j; k++) {
                if (j != k) {
                    double randn = randoms.Uniforme() * TAUMAX;
                    phero.set2D(j, k, randn);
                    phero.set2D(k, j, randn);
                }
            }
        }
        phero.updateDevice(true);
        return phero;
    }

    DM<int> createIRoulette(const DM<double> &distances, int ncities) {
        DM<int> iroulette(ncities, IROULETE, 0);
        for (int i = 0; i < ncities; i++) {
            for (int y = 0; y < IROULETE; y++) {
                double maxdistance = std::numeric_limits<double>::infinity();
                int city = -1;
                for (int j = 0; j < ncities; j++) {
                    bool check = true;
                    for (int k = 0; k < y; k++) {
                        if (iroulette.localPartition[i * IROULETE + k] == j) {
                            check = false;
                        }
                    }
                    if (i != j && check) {
                        double c_dist = distances.localPartition[i * ncities + j];
                        if (c_dist < maxdistance) {
                            maxdistance = c_dist;
                            city = j;
                        }
                    }
                }
                iroulette.set2D(i, y, city);
            }
        }
        iroulette.updateDevice(true);
        return iroulette;
    }

    void aco(int iterations, const std::string& importFile, int nants) {
        msl::startTiming();
        DA<city> cities = readCities(importFile);
        int ncities = cities.getSize();
        DM<double> phero = createPheroMatrix(ncities);
        DM<double> distance(ncities, ncities, 0);
        CalcDistance calcdistance(cities.getUserFunctionData());
        distance.mapIndexInPlace(calcdistance);
        distance.updateHost();
        DM<int> iroulette = createIRoulette(distance, ncities);
        DM<int> flipped_tours(nants, ncities, -1);
        DM<double> deltaphero(ncities, ncities, 0);
        DM<double> etatau(ncities, ncities, 0);
        DA<double> dist_routes(nants, 0);

        Fill fill(-1);
        Min min;
        double minroute;
        EtaTauCalc etataucalc;
        int veryGoodSeed = (int) time(nullptr);
        TourConstruction tourConstruction(ncities, veryGoodSeed);
        UpdateDelta updatedelta(ncities, nants);
        UpdatePhero updatephero;
        double dsinit = msl::stopTiming();

        double alltimeminroute = std::numeric_limits<double>::infinity();
        for (int i = 0; i < iterations; i++) {
            flipped_tours.mapInPlace(fill);
            // Write the eta tau value to the data structure.
            etatau.zipIndexInPlace3(distance, phero, etataucalc);
            tourConstruction.setIterationParams(iroulette.getUserFunctionData(),
                                                etatau.getUserFunctionData(),
                                                distance.getUserFunctionData());
            dist_routes.mapIndexInPlaceDMRows(flipped_tours, tourConstruction);
            // Get the best route.
            minroute = dist_routes.foldCPU(min);
            if (minroute < alltimeminroute) {
                alltimeminroute = minroute;
            }
            updatedelta.setIterationsParams(flipped_tours.getUserFunctionData(),
                                            dist_routes.getUserFunctionData());
            // Calculate the delta pheromone.
            deltaphero.mapIndexInPlace(updatedelta);
            // Update the pheromone.
            phero.zipIndexInPlace(deltaphero, updatephero);
            if (i != iterations - 1) {
                flipped_tours.fill(-1);
            }
        }
        double calctime = msl::stopTiming();// etataucalctime + constructtime + deltapherotime + updatepherotime + resettime + minroutetime;
        printf("%s;%d;%s;%f;%f;%f;", importFile.c_str(), nants, "flippedds",
               calctime, dsinit+calctime, alltimeminroute);// %.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f etataucalctime, constructtime, minroutetime, deltapherotime, updatepherotime, resettime, alltimeminroute);

        dist_routes.updateHost();
        flipped_tours.updateHost();
        if (CHECKCORRECTNESS) {
            checkminroute(nants, minroute, dist_routes);
            checkvalidroute(flipped_tours, ncities, nants);
            // printf("Made it!");
        }
        msl::stopTiming();
    }
} // close namespaces

void exitWithUsage() {
    std::cerr
            << "Usage: ./gassimulation_test [-g <nGPUs>] [-n <iterations>] [-i <importFile>] [-t <threads>] [-c <cities>] [-a <ants>] [-r <runs>]"
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
    if (!exportFile.empty()) {
        // TODO Export result
    }
    msl::terminateSkeletons();
    return EXIT_SUCCESS;
}
