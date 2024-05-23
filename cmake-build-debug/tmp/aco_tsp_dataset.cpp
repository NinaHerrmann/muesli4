
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
#include "tour.h"
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
#define DJIBOUTI 38
#define CATAR 194
#define A280 280
#define DJ198 198
#define L318 318
#define PCB442 442
#define RAT783 783
#define LUXEMBOURG 980
#define PR1002 1002
#define PCB1173 1173
#define D1291 1291
#define PR2392 2392

std::ostream& operator<< (std::ostream& os, const city t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
typedef tour<int, DJIBOUTI> djroute;
std::ostream& operator<< (std::ostream& os, djroute t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
typedef tour<int, CATAR> caroute;
std::ostream& operator<< (std::ostream& os, caroute t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
typedef tour<int, DJ198> d198route;
std::ostream& operator<< (std::ostream& os, d198route t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
typedef tour<int, A280> a280route;
std::ostream& operator<< (std::ostream& os, a280route t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
typedef tour<int, L318> l318route;
std::ostream& operator<< (std::ostream& os, l318route t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
typedef tour<int, PCB442> pcb442route;
std::ostream& operator<< (std::ostream& os, pcb442route t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
typedef tour<int, RAT783> rat783route;
std::ostream& operator<< (std::ostream& os, rat783route t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
typedef tour<int, LUXEMBOURG> luxembourgroute;
std::ostream& operator<< (std::ostream& os, luxembourgroute t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}typedef tour<int, PR1002> pr1002route;
std::ostream& operator<< (std::ostream& os, pr1002route t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
typedef tour<int, PCB1173> pcb1173route;
std::ostream& operator<< (std::ostream& os, pcb1173route t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
typedef tour<int, D1291> d1291route;
std::ostream& operator<< (std::ostream& os, d1291route t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
typedef tour<int, PR2392> pr2392route;
std::ostream& operator<< (std::ostream& os, pr2392route t) {
    os << "(" << t[0] << ", " << t[1] << ")";
    return os;
}
#define CHECKCORRECTNESS 1
namespace msl::aco {
    template<typename T>
    class Fill : public Functor<T, T> {
        int d, ncities;
    public:
        explicit Fill(int d, int ncities) : d(d), ncities(ncities) {}

        MSL_USERFUNC T operator()(T x) const override {
            for (int i = 0; i < ncities; i++) {
                x[i] = d;
            }
            return x;
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

    template<typename T>
    class Min : public Functor2<T, T, T> {
    private:
        int width;
    public:
        explicit Min(int width) : width(width) {}
        MSL_USERFUNC T operator()(T x, T y) const override {
            if (x.getDist() < y.getDist()) { return x; }
            return y;}
    };
    class CalcDistance : public Functor3<int, int, double, double> {
    private:
        city *cities;
    public:
        explicit CalcDistance(city *cities) {
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
    template<typename T>
    class TourConstruction: public Functor2<int, T, T> {
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

        MSL_USERFUNC T operator()(int ant_index, T build_tour) const override {
            MSL_RANDOM_STATE randomState = msl::generateRandomState(this->seed, ant_index);
            int fromCity = msl::randInt(0, width - 1, randomState);

            build_tour[fromCity] = 0;
            double distance = 0;
            for (int i = 1; i < width; i++) {
                int nextCity = -1;
                double etaTauSum = 0;
                for (int j = 0; j < IROULETE; j++) {
                    int toCity = iroulette[fromCity * IROULETE + j];
                    if (build_tour[toCity] == -1) {
                        etaTauSum += etataus[toCity * width + fromCity];
                    }
                }
                if (etaTauSum != 0) {
                    double rand = msl::randDouble(0.0, etaTauSum, randomState);
                    double etaTauSum2 = 0;

                    for (int j = 0; j < IROULETE; j++) {
                        nextCity = iroulette[fromCity * IROULETE + j];
                        if (build_tour[nextCity] == -1) {
                            etaTauSum2 += etataus[nextCity * width + fromCity];
                        }
                        if (rand < etaTauSum2)
                            break;
                    }
                } else {
                    int startCity = msl::randInt(0, width - 1, randomState);
                    for (int j = 0; j < width; j++) {
                        if (build_tour[(startCity + j) % width] == -1) {
                            nextCity = (startCity + j) % width;
                        }
                    }
                }
                build_tour[nextCity] = i;
                distance += distances[nextCity * width + fromCity];
                fromCity = nextCity;
            }
            build_tour.setDist(distance);
            return build_tour;
        }
    };
    /*template<typename T>
    class UpdateDelta : public Functor3<int, int, double, double> {
    private:
        T *dist_routes{};
        int nants, ncities;
    public:
        UpdateDelta(int cities, int ants) {
            this->nants = ants;
            this->ncities = cities;
        }
        // template<typename T, unsigned int ncities>
        void setIterationsParams(T *distroutes) {
            this->dist_routes = distroutes;
        }

        MSL_USERFUNC double operator()(int row, int column, double phero) const override {
            int Q = 11340;
            double result = 0.0;
            for (int k = 0; k < nants; k++) {
                auto rlength = dist_routes[k].getDist();
                int city1VisitIndex = dist_routes[k][row];
                int city2VisitIndex = dist_routes[k][column];
                if (abs(city1VisitIndex - city2VisitIndex) == 1) {
                    result += Q / rlength;
                }
            }
            return result;
        }
    };*/
    template<typename T>
    class UpdatePhero : public Functor3<int, int, double, double> {
    private:
        T *dist_routes{};
        int nants;
    public:
        UpdatePhero(int ants) {
            this->nants = ants;
        }
        void setIterationsParams(T *distroutes) {
            this->dist_routes = distroutes;
        }
    public:
        MSL_USERFUNC double operator()(int row, int column, double prevphero) const override {
            double RO = 0.5;
            int Q = 11340;
            double result = 0.0;
            for (int k = 0; k < nants; k++) {
                auto rlength = dist_routes[k].getDist();
                int city1VisitIndex = dist_routes[k][row];
                int city2VisitIndex = dist_routes[k][column];
                if (abs(city1VisitIndex - city2VisitIndex) == 1) {
                    result += Q / rlength;
                }
            }
            return (1 - RO) * (prevphero + result);
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

    template<typename T2>
    void aco(int iterations, int nants, DA<city> &cities, int ncities, const std::string& importFile ) {
        msl::startTiming();
        msl::DM<double> phero = createPheroMatrix(ncities);
        msl::DM<double> distance(ncities, ncities, 0);
        CalcDistance calcdistance(cities.getUserFunctionData());
        distance.mapIndexInPlace(calcdistance);
        distance.updateHost();
        msl::DM<int> iroulette = createIRoulette(distance, ncities);
        msl::DM<double> etatau(ncities, ncities, 0);
        DM<T2> routes(nants, {});
        Fill<T2> fill(-1, ncities);
        Min<T2> min(ncities);
        T2 minroute;
        EtaTauCalc etataucalc;
        int veryGoodSeed = (int) time(nullptr);
        TourConstruction<T2> tourConstruction(ncities, veryGoodSeed);
        UpdatePhero<T2> updatephero(nants);
        double dsinit = msl::stopTiming();
        T2 alltimeminroute = {};
        for (int i = 0; i < iterations; i++) {
            routes.mapInPlace(fill);
            // Write the eta tau value to the data structure.
            etatau.zipIndexInPlace3(distance, phero, etataucalc);
            tourConstruction.setIterationParams(iroulette.getUserFunctionData(),
                                                etatau.getUserFunctionData(),
                                                distance.getUserFunctionData());
            routes.mapIndexInPlace(tourConstruction);
            // Get the best route.
            minroute = routes.foldCPU(min);
            if (i == 0) {
                alltimeminroute = minroute;
            } else {
                if (minroute[ncities+1] < alltimeminroute[ncities+1]) {
                    alltimeminroute = minroute;
                }
            }
            updatephero.setIterationsParams(routes.getUserFunctionData());
            // Update the pheromone.
            phero.mapIndexInPlace(updatephero);
        }

        double calctime = msl::stopTiming();// etataucalctime + constructtime + deltapherotime + updatepherotime + resettime + minroutetime;
        printf("%s;%d;%s;%f;%f;%f;", importFile.c_str(), nants, "singlekernel",
               calctime, dsinit+calctime, alltimeminroute.getDist());// %.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f etataucalctime, constructtime, minroutetime, deltapherotime, updatepherotime, resettime, alltimeminroute);

        routes.updateHost();
        if (CHECKCORRECTNESS) {
            checkminroute(nants, minroute.getDist(), static_cast<const DA<double>>(routes));
            checkvalidroute(routes, ncities, nants);
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

}
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
        msl::DA<city> cities = msl::aco::readCities(importFile);
        const int ncities = cities.getSize();
        if (ncities == 38) {
            typedef tour<int, DJIBOUTI> route;
            msl::aco::aco<route>(iterations, nants, cities, ncities, importFile);
        } else if (ncities == 194) {
            typedef tour<int, CATAR> route;
            msl::aco::aco<route>(iterations, nants, cities, ncities, importFile);
        } else if (ncities == 198) {
            typedef tour<int, DJ198> route;
            msl::aco::aco<route>(iterations, nants, cities, ncities, importFile);
        } else if (ncities == 280) {
            typedef tour<int, A280> route;
            msl::aco::aco<route>(iterations, nants, cities, ncities, importFile);
        } else if (ncities == 318){
            typedef tour<int, L318> route;
            msl::aco::aco<route>(iterations, nants, cities, ncities, importFile);
        } else if (ncities == 442){
            typedef tour<int, PCB442> route;
            msl::aco::aco<route>(iterations, nants, cities, ncities, importFile);
        } else if (ncities == 783){
            typedef tour<int, RAT783> route;
            msl::aco::aco<route>(iterations, nants, cities, ncities, importFile);
        } else if (ncities == 980){
            typedef tour<int, LUXEMBOURG> route;
            msl::aco::aco<route>(iterations, nants, cities, ncities, importFile);
        } else if (ncities == 1002){
            typedef tour<int, PR1002> route;
            msl::aco::aco<route>(iterations, nants, cities, ncities, importFile);
        } else if (ncities == 1173){
            typedef tour<int, PCB1173> route;
            msl::aco::aco<route>(iterations, nants, cities, ncities, importFile);
        } else if (ncities == 1291){
            typedef tour<int, D1291> route;
            msl::aco::aco<route>(iterations, nants, cities, ncities, importFile);
        } else if (ncities == 2392){
            typedef tour<int, PR2392> route;
            msl::aco::aco<route>(iterations, nants, cities, ncities, importFile);
        }
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
