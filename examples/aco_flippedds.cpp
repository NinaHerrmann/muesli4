
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
        double * phero{}, * dist{};
        int cities;
    public:
        EtaTauCalc(int cities) {
            this->cities = cities;
        }
        void setIterationsParams(double * distance, double * pheromones) {
            this->dist = distance;
            this->phero = pheromones;
        }

        MSL_USERFUNC double operator()(int xindex, int yindex, double eta_tau) const override {
            double d_ALPHA = 1;
            double d_BETA = 2;
            int fromcity = xindex;
            int next_city = yindex;

            eta_tau = 0;
            // For every city which can be visited, calculate the eta and tau value.
            if (fromcity != next_city) {
                // Looks like zero but is just very small.
                double eta = pow(1 /  dist[fromcity * this->cities + next_city], d_BETA);
                double tau = pow(phero[fromcity * this->cities + next_city], d_ALPHA);
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

    class WorkHorse : public Functor2<int, double, double> {
    private:
        int width;
        int seed;
        int * iroulette;
        double * etataus;
        double * distances;
        int * tours;
    public:
        WorkHorse(int width, int seed) : width(width), seed(seed) {
        }

        void setIterationParams(int * _iroulette, double * _etataus, double * _distances, int * _tours) {
            this->iroulette = _iroulette;
            this->etataus = _etataus;
            this->distances = _distances;
            this->tours = _tours;
        }

        MSL_USERFUNC double operator()(int row, double unused) const override {
            MSL_RANDOM_STATE randomState = msl::generateRandomState(this->seed, row);

            int* rowdata = &this->tours[row * width];

            int fromCity = msl::randInt(0, width - 1, randomState);
            rowdata[fromCity] = 0;

            double distance = 0;

            for (int i = 1; i < width; i++) {
                double etaTauSum = 0;
                for (int j = 0; j < IROULETE; j++) {
                    int toCity = iroulette[row * IROULETE + j];
                    // Not visited yet.
                    if (rowdata[toCity] == -1) {
                        // x = fromCity, y = toCity.
                        etaTauSum += etataus[toCity * width + fromCity];
                    }
                }

                int nextCity;
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
                        }
                        if (j == width - 1) {
                            printf("Somehow, ant %d found no free city in step %d\n", row, i);
                        }
                    }
                }

                rowdata[nextCity] = i;
                distance += distances[nextCity * width + fromCity];
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
                dist_routes[k] = rlength;
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
        for (int ant = 0; ant < nants; ant++) {
            for (int n = 0; n < ncities; n++) {
                int found = 0;
                for (int i = 0; i < ncities; i++) {
                    if (routes.localPartition[ant * ncities + i] == n)
                        found++;
                }
                if (found != 1) {
                    printf("Ant %d: Visit %d was found %d times\n", ant, n, found);
                }
            }
        }
    }

    void aco(int iterations, const std::string& importFile, int nants) {
        int niroulet = IROULETE;
        int ncities = readsize(importFile);
        double dsinit = 0.0;
        msl::startTiming();
        DA<city> cities(ncities, {});
        DM<double> phero(ncities, ncities, {});
        DM<double> distance(ncities, ncities, {});
        readData(importFile, ncities, cities, phero);
        DM<int> iroulet(ncities, niroulet, 0);
        CalcDistance calcdistance(cities.getUserFunctionData());
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
        DM<int> flipped_tours(nants, ncities, -1);
        DM<double> deltaphero(ncities, ncities, 0);
        DM<double> etatau(ncities, ncities, {});
        DA<double> dist_routes(nants, 0);

        Reset reset;
        Min min;
        double minroute;
        EtaTauCalc etataucalc(ncities);
        int veryGoodSeed = (int) time(nullptr);
        WorkHorse workHorse(ncities, veryGoodSeed);
        UpdateDelta updatedelta(ncities, nants);
        UpdatePhero updatephero;
        dsinit = msl::stopTiming();
        double etataucalctime = 0.0, constructtime = 0.0, deltapherotime = 0.0, updatepherotime = 0.0, resettime = 0.0,
            minroutetime = 0.0;

        double alltimeminroute = 999999.9;
        for (int i = 0; i < iterations; i++) {
            msl::startTiming();
            etataucalc.setIterationsParams(distance.getUserFunctionData(), phero.getUserFunctionData());
            // Write the eta tau value to the data structure.
            etatau.mapIndexInPlace(etataucalc);
            etataucalctime += msl::stopTiming();
            etatau.show("dist routes", 6);

            msl::startTiming();
            workHorse.setIterationParams(iroulet.getUserFunctionData(), etatau.getUserFunctionData(), distance.getUserFunctionData(), flipped_tours.getUserFunctionData());
            dist_routes.mapIndexInPlace(workHorse);
            constructtime += msl::stopTiming();
            dist_routes.show("dist routes", 6);
            // Get the best route.
            msl::startTiming();
            msl::syncStreams();
            minroute = dist_routes.foldCPU(min);
            minroutetime += msl::stopTiming();
            printf("minroute: %.2f\n", minroute);
            if (minroute < alltimeminroute) {
                alltimeminroute = minroute;
            }

            msl::startTiming();
            updatedelta.setIterationsParams(flipped_tours.getUserFunctionData(), dist_routes.getUserFunctionData());
            // Calculate the delta pheromone.
            deltaphero.mapIndexInPlace(updatedelta);
            deltapherotime += msl::stopTiming();
            // Update the pheromone.
            msl::startTiming();
            phero.zipIndexInPlace(deltaphero, updatephero);
            updatepherotime += msl::stopTiming();

            msl::startTiming();
            if (i != iterations - 1) {
                flipped_tours.fill(-1);
            }
            resettime += msl::stopTiming();
        }
        double calctime = etataucalctime + constructtime + deltapherotime + updatepherotime + resettime + minroutetime;
        printf("%s;%d;%.6f;%.6f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f \n", importFile.c_str(), nants,
               dsinit, calctime, etataucalctime, constructtime, minroutetime, deltapherotime, updatepherotime,
               resettime, alltimeminroute);
        // importFile.c_str(), nants,   dsinit, fill,   ds2fill     etataucalctime, reduceRowstime, calcprobstime, nextsteptime, deltapherotime, updatepherotime, calcrlengthtime, minroutetime, alltimeminroute);
        // pcb442;              256;    2.9862; 0.1416; 0.0058;     6.4798;         3.7537;         0.6522;        76.8972;        2.1370;             0.0008;         0.0169;     0.0033,          217790.4655
        dist_routes.updateHost();
        flipped_tours.updateHost();
        flipped_tours.show("flipped tours", 76);
        if (CHECKCORRECTNESS) {
            checkminroute(nants, minroute, dist_routes);
            checkvalidroute(flipped_tours, ncities, nants);
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
